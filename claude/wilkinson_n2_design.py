#!/usr/bin/env python3
"""
Two-section lumped Wilkinson divider: component selection + Monte Carlo.

Pipeline
  1. Continuous optimization of the 7 design values (real-part models included)
  2. Candidate lists: the E-series neighbours of each optimized value
  3. Exhaustive discrete search on the nominal circuit (minimax cost over the band)
  4. Monte Carlo on the shortlist with common random numbers -> yield
  5. Report for the winner: sensitivity, dissipation / currents at P_IN, fault cases

The circuit is solved with plain NumPy nodal analysis (vectorized over Monte
Carlo runs and frequency), so no scikit-rf is needed here. scikit-rf is still
the right tool to cross-check the winner and to compare against VNA data.

Topology = your scikit-rf netlist (section B at the input, section A at the outputs):

    port1 --LB_1--(JA)--LA_1-- port2     shunt to ground:  CB_0 (=2*CB) at the input node,
          \\-LB_2--(JB)--LA_2-- port3                      CB_1, CA_11 at JA ; CB_2, CA_21 at JB,
                                                         CA_12 at port 2 ; CA_22 at port 3
    RB between JA and JB, RA between the two outputs

ALL part models and tolerances below are ASSUMPTIONS. Replace them with
datasheet / vendor data for the parts you can actually buy and assemble.
"""
import itertools
import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Configuration (edit these)
# --------------------------------------------------------------------------
Z0 = 50.0
F_REF = 165e6                                  # centre of the band
F_BAND = np.linspace(140e6, 190e6, 51)         # spec band
F_WIDE = np.linspace(100e6, 250e6, 301)        # for plots only
P_IN = 2.0                                     # available source power, W

# Pass/fail spec over F_BAND (dB). il = worst insertion loss, imb = S21/S31 imbalance
SPEC = dict(rl=-20.0, iso=-20.0, il=3.5, imb=0.3)

# Cohn table values (normalised to Z0 = 50 ohm) from your netlist
COHN = dict(z1_n=1.1998, r1_n=5.3163,      # section A (outputs)
            z2_n=1.6670, r2_n=1.8643)      # section B (input)
# Your code uses RB = r2_n*Z0 but RA = r1_n*ZA. "ZA" keeps that; "Z0" normalises both to Z0.
RA_NORM = "ZA"
RUN_CONTINUOUS = True   # False: skip the optimizer, take E-series neighbours of the Cohn values

# Part models (assumed). Q is taken at F_REF; Cp is the inductor's parallel capacitance.
MODEL = dict(L_Q=60.0, L_cp=0.15e-12, C_esr=0.15, C_esl=0.6e-9)

# Tolerances: relative for L and R; for C the larger of a relative and an absolute tolerance
TOL = dict(L=0.02, R=0.01, C_rel=0.02, C_abs=0.1e-12)

N_RUNS = 2000          # Monte Carlo runs per candidate
N_EACH = 1             # E-series neighbours kept per side (1 -> 2 candidates per value)
SHORTLIST = 10         # nominal-cost winners passed to the Monte Carlo
DIST = "gauss"         # "gauss" (sigma = tol/2, clipped at +-tol) or "uniform"
SEED = 1

# --------------------------------------------------------------------------
# Circuit definition
# --------------------------------------------------------------------------
GROUPS = ["LB", "LA", "C0", "CB", "CA", "RB", "RA"]     # design variables
INSTANCES = {                                           # physical parts per variable
    "LB": ["LB_1", "LB_2"], "LA": ["LA_1", "LA_2"],
    "C0": ["CB_0"],                                      # input shunt cap, nominally 2*CB
    "CB": ["CB_1", "CB_2"],
    "CA": ["CA_11", "CA_12", "CA_21", "CA_22"],
    "RB": ["RB"], "RA": ["RA"],
}
INST_LIST = [i for g in GROUPS for i in INSTANCES[g]]
GROUP_OF = {i: g for g, insts in INSTANCES.items() for i in insts}

IN, JA, JB, O2, O3 = range(5)       # nodes (ground is implicit)
PORTS = [IN, O2, O3]
BRANCHES = [                         # (instance, kind, node_i, node_j or None=ground)
    ("LB_1", "L", IN, JA), ("LB_2", "L", IN, JB),
    ("LA_1", "L", JA, O2), ("LA_2", "L", JB, O3),
    ("CB_0", "C", IN, None),
    ("CB_1", "C", JA, None), ("CA_11", "C", JA, None),
    ("CB_2", "C", JB, None), ("CA_21", "C", JB, None),
    ("CA_12", "C", O2, None), ("CA_22", "C", O3, None),
    ("RB", "R", JA, JB), ("RA", "R", O2, O3),
]

E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9,
       4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
E96 = list(np.round(10 ** (np.arange(96) / 96), 2))      # verify against the standard table
SERIES = {"L": (E24, 1e-9), "C": (E24, 1e-12), "R": (E96, 1.0)}   # (series, unit)


# --------------------------------------------------------------------------
# Circuit solver
# --------------------------------------------------------------------------
def admittance(kind, value, w, model):
    """Branch admittance. value: (N,1), w: (1,F) -> (N,F)."""
    if model is None:                                    # ideal parts
        q_inv, cp, esr, esl = 0.0, 0.0, 0.0, 0.0
    else:
        q_inv, cp = 1.0 / model["L_Q"], model["L_cp"]
        esr, esl = model["C_esr"], model["C_esl"]
    if kind == "L":
        zs = 2 * np.pi * F_REF * value * q_inv + 1j * w * value
        return 1.0 / zs + 1j * w * cp
    if kind == "C":
        return 1.0 / (esr + 1j * w * esl + 1.0 / (1j * w * value))
    return (1.0 / value) * np.ones_like(w) + 0j          # resistor


def build_Y(vals, f, model):
    """Nodal admittance matrix, shape (N, F, 5, 5). vals[inst] has shape (N,)."""
    w = 2 * np.pi * np.asarray(f)[None, :]
    n = len(next(iter(vals.values())))
    Y = np.zeros((n, len(f), 5, 5), complex)
    for inst, kind, i, j in BRANCHES:
        y = admittance(kind, np.asarray(vals[inst], float)[:, None], w, model)
        Y[..., i, i] += y
        if j is not None:
            Y[..., j, j] += y
            Y[..., i, j] -= y
            Y[..., j, i] -= y
    return Y


def s_params(vals, f, model):
    """3-port S-parameters, shape (N, F, 3, 3), reference impedance Z0."""
    Y = build_Y(vals, f, model)
    for p in PORTS:
        Y[..., p, p] += 1.0 / Z0
    B = np.zeros((5, 3), complex)
    for k, p in enumerate(PORTS):
        B[p, k] = 1.0 / Z0                               # unit-EMF source behind Z0
    V = np.linalg.solve(Y, np.broadcast_to(B, Y.shape[:-2] + (5, 3)))
    return 2.0 * V[..., PORTS, :] - np.eye(3)


def metrics(S):
    db = 20 * np.log10(np.abs(S) + 1e-12)
    rl = np.maximum.reduce([db[..., 0, 0], db[..., 1, 1], db[..., 2, 2]]).max(axis=-1)
    iso = db[..., 1, 2].max(axis=-1)
    il = (-np.minimum(db[..., 1, 0], db[..., 2, 0])).max(axis=-1)
    imb = np.abs(db[..., 1, 0] - db[..., 2, 0]).max(axis=-1)
    return dict(rl=rl, iso=iso, il=il, imb=imb)


def cost(m):
    """Minimax cost in dB: worst return loss / isolation plus a penalty when the
    insertion loss exceeds its spec (loss set by inductor Q is common to all candidates)."""
    return np.maximum(m["rl"], m["iso"]) + 5.0 * np.maximum(0.0, m["il"] - SPEC["il"])


def passes(m):
    return (m["rl"] <= SPEC["rl"]) & (m["iso"] <= SPEC["iso"]) & \
           (m["il"] <= SPEC["il"]) & (m["imb"] <= SPEC["imb"])


def expand(design):
    """design[group] -> per-instance values (same nominal on both arms)."""
    return {inst: design[g] for g, insts in INSTANCES.items() for inst in insts}


# --------------------------------------------------------------------------
# Step 1: continuous optimization
# --------------------------------------------------------------------------
def cohn_start():
    """Pi-section conversion of the Cohn table values at F_REF (same maths as your netlist)."""
    w0 = 2 * np.pi * F_REF
    zb, za = COHN["z2_n"] * Z0, COHN["z1_n"] * Z0
    cb, ca = 1 / (w0 * zb), 1 / (w0 * za)
    lb, la = zb / w0, za / w0
    rb = COHN["r2_n"] * Z0
    ra = COHN["r1_n"] * Z0
    return np.array([lb, la, 2 * cb, cb, ca, rb, ra])     # order of GROUPS


def optimize_continuous(x0, model, restarts=3, seed=0):
    rng = np.random.default_rng(seed)

    def f(x):
        d = {g: np.array([v]) for g, v in zip(GROUPS, x0 * np.exp(x))}
        return float(cost(metrics(s_params(expand(d), F_BAND, model)))[0])

    best = None
    for r in range(restarts):
        start = np.zeros(len(x0)) if r == 0 else rng.normal(0, 0.1, len(x0))
        res = minimize(f, start, method="Nelder-Mead",
                       options=dict(maxiter=1500, xatol=1e-4, fatol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    return x0 * np.exp(best.x), best.fun


# --------------------------------------------------------------------------
# Step 2: E-series candidate lists
# --------------------------------------------------------------------------
def neighbours(value, kind, n_each=N_EACH):
    series, unit = SERIES[kind]
    decade = int(np.floor(np.log10(value / unit)))
    allv = np.array(sorted(s * unit * 10.0 ** d for d in range(decade - 1, decade + 2)
                           for s in series))
    below = allv[allv <= value * (1 + 1e-9)][-n_each:]
    above = allv[allv > value * (1 + 1e-9)][:n_each]
    return np.concatenate([below, above])


def tol_rel(group, value):
    if group.startswith("L"):
        return TOL["L"]
    if group.startswith("R"):
        return TOL["R"]
    return max(TOL["C_rel"], TOL["C_abs"] / value)


# --------------------------------------------------------------------------
# Step 3: exhaustive discrete search (nominal, real-part models)
# --------------------------------------------------------------------------
def discrete_search(x_opt, model, chunk=4096):
    cands = [neighbours(v, g[0]) for g, v in zip(GROUPS, x_opt)]   # g[0] in {L, C, R}
    combos = np.array(list(itertools.product(*cands)))
    costs = np.empty(len(combos))
    for s in range(0, len(combos), chunk):
        blk = combos[s:s + chunk]
        d = {g: blk[:, i] for i, g in enumerate(GROUPS)}
        costs[s:s + chunk] = cost(metrics(s_params(expand(d), F_BAND, model)))
    order = np.argsort(costs)
    return combos, costs, order


# --------------------------------------------------------------------------
# Step 4: Monte Carlo (common random numbers)
# --------------------------------------------------------------------------
def draw_U(n, seed=SEED, dist=DIST):
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        return rng.uniform(-1, 1, (n, len(INST_LIST)))
    return np.clip(rng.normal(0, 0.5, (n, len(INST_LIST))), -1, 1)


def mc_vals(nominal, U):
    """nominal: dict group -> float. Each physical part gets its own random deviation."""
    return {inst: nominal[GROUP_OF[inst]] *
            (1 + tol_rel(GROUP_OF[inst], nominal[GROUP_OF[inst]]) * U[:, k])
            for k, inst in enumerate(INST_LIST)}


def monte_carlo(nominal, U, model, f=F_BAND):
    m = metrics(s_params(mc_vals(nominal, U), f, model))
    return m, passes(m)


# --------------------------------------------------------------------------
# Step 5: reports for the winner
# --------------------------------------------------------------------------
def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def sensitivity(U, m):
    worst = np.maximum(m["rl"], m["iso"])
    return sorted(((spearman(U[:, k], worst), inst) for k, inst in enumerate(INST_LIST)),
                  key=lambda t: -abs(t[0]))


def power_report(nominal, model, f=F_BAND):
    """Dissipation / currents / voltages at P_IN (RMS) with matched and faulted port 3."""
    vals = {inst: np.array([nominal[GROUP_OF[inst]]]) for inst in INST_LIST}
    w = 2 * np.pi * f
    zs1 = 2 * np.pi * F_REF * nominal["LB"] / model["L_Q"] + 1j * w * nominal["LB"]
    zs2 = 2 * np.pi * F_REF * nominal["LA"] / model["L_Q"] + 1j * w * nominal["LA"]
    scale = 2 * np.sqrt(Z0 * P_IN)                       # source EMF for P_IN available
    print(f"\nWorst case over the band at P_IN = {P_IN} W (RMS values)")
    print(f"{'case':<14}{'P(RB) W':>9}{'P(RA) W':>9}{'I LB A':>9}{'I LA A':>9}{'Vpk max V':>11}")
    for name, term3 in [("matched", 1 / Z0), ("port 3 open", 0.0), ("port 3 short", 1e6)]:
        Y = build_Y(vals, f, model)
        Y[..., IN, IN] += 1 / Z0
        Y[..., O2, O2] += 1 / Z0
        Y[..., O3, O3] += term3
        b = np.zeros(5, complex)
        b[IN] = 1 / Z0
        V = np.linalg.solve(Y, np.broadcast_to(b, Y.shape[:-1])[..., None])[0, :, :, 0] * scale
        pr1 = (abs(V[:, JA] - V[:, JB]) ** 2 / nominal["RB"]).max()
        pr2 = (abs(V[:, O2] - V[:, O3]) ** 2 / nominal["RA"]).max()
        il1 = (np.maximum(abs(V[:, IN] - V[:, JA]), abs(V[:, IN] - V[:, JB])) / abs(zs1)).max()
        il2 = (np.maximum(abs(V[:, JA] - V[:, O2]), abs(V[:, JB] - V[:, O3])) / abs(zs2)).max()
        vpk = np.sqrt(2) * abs(V).max()
        print(f"{name:<14}{pr1:9.3f}{pr2:9.3f}{il1:9.3f}{il2:9.3f}{vpk:11.1f}")


def plot_envelope(nominal, U, model, fname="mc_envelope.png", n_curves=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = s_params(mc_vals(nominal, U[:n_curves]), F_WIDE, model)
    S0 = s_params({i: np.array([nominal[GROUP_OF[i]]]) for i in INST_LIST}, F_WIDE, model)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (i, j), c, lab in [((0, 0), "C0", "S11"), ((1, 1), "C3", "S22"), ((1, 2), "C2", "S23")]:
        ax.plot(F_WIDE / 1e6, 20 * np.log10(abs(S[:, :, i, j])).T, c, alpha=0.04)
        ax.plot(F_WIDE / 1e6, 20 * np.log10(abs(S0[0, :, i, j])), c, lw=2, label=lab)
    ax.axvspan(F_BAND[0] / 1e6, F_BAND[-1] / 1e6, color="c", alpha=0.12)
    ax.axhline(SPEC["rl"], color="k", ls="--", lw=1)
    ax.set_ylim(-70, 0)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend()
    ax.set_title("Monte Carlo envelope (nominal in bold)")
    fig.savefig(fname, dpi=150, bbox_inches="tight")


# --------------------------------------------------------------------------
def fmt(g, v):
    unit, scale = {"L": ("nH", 1e9), "C": ("pF", 1e12), "R": ("ohm", 1.0)}[g[0]]
    return f"{g}={v * scale:.4g} {unit}"


def print_bom(nom):
    print("\nBOM (value, tolerance used in the Monte Carlo):")
    for inst in INST_LIST:
        g = GROUP_OF[inst]
        print(f"  {inst:<6} {fmt(g, nom[g]).split('=')[1]:<12} +-{100 * tol_rel(g, nom[g]):.1f}%")
    print("\nValues to paste into your scikit-rf netlist (SI units):")
    for g in GROUPS:
        print(f"  {g} = {nom[g]:.4g}")


def main():
    # 0. the Cohn design converted to lumped parts, as in your netlist
    x_cohn = cohn_start()
    print("Cohn start: " + ", ".join(fmt(g, v) for g, v in zip(GROUPS, x_cohn)))
    d0 = {g: np.array([v]) for g, v in zip(GROUPS, x_cohn)}
    for label, mdl in [("ideal parts", None), ("real-part model", MODEL)]:
        m0 = metrics(s_params(expand(d0), F_BAND, mdl))
        print(f"  {label:<16} worst RL {m0['rl'][0]:6.1f} dB | iso {m0['iso'][0]:6.1f} dB | "
              f"IL {m0['il'][0]:.2f} dB")

    # 1. continuous optimum (or stay on the Cohn values)
    if RUN_CONTINUOUS:
        x_opt, c_opt = optimize_continuous(x_cohn, MODEL)
        print("\nContinuous optimum (cost %.1f dB):" % c_opt)
    else:
        x_opt = x_cohn
        print("\nContinuous optimization skipped; using the Cohn values")
    print("  " + ", ".join(fmt(g, v) for g, v in zip(GROUPS, x_opt)))

    # 2-3. discrete search on E-series neighbours
    combos, costs, order = discrete_search(x_opt, MODEL)
    print(f"\n{len(combos)} combinations evaluated; best nominal cost {costs[order[0]]:.1f} dB")

    # 4. Monte Carlo on the shortlist with the SAME random draws for every candidate
    U = draw_U(N_RUNS)
    rows = []
    for idx in order[:SHORTLIST]:
        nom = dict(zip(GROUPS, combos[idx]))
        m, ok = monte_carlo(nom, U, MODEL)
        rows.append((ok.mean(), -costs[idx], nom, m))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print(f"\nShortlist ({N_RUNS} runs, spec {SPEC}):")
    for y, negc, nom, m in rows:
        print(f"  yield {100 * y:5.1f}%  nominal {-negc:6.1f} dB  "
              f"p95 RL {np.percentile(m['rl'], 95):6.1f}  p95 iso {np.percentile(m['iso'], 95):6.1f}  | "
              + ", ".join(fmt(g, nom[g]) for g in GROUPS))

    # 5. reports for the winner
    y, _, nom, m = rows[0]
    print("\nWinner:", ", ".join(fmt(g, nom[g]) for g in GROUPS))
    print("Sensitivity (Spearman correlation with worst of RL/isolation, top 6):")
    for rho, inst in sensitivity(U, m)[:6]:
        print(f"  {inst:<6} {rho:+.2f}")
    power_report(nom, MODEL)
    print_bom(nom)
    plot_envelope(nom, U, MODEL)
    print("\nSaved mc_envelope.png")


if __name__ == "__main__":
    main()
