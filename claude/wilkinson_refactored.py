#!/usr/bin/env python3
r"""
Two-section lumped Wilkinson divider: part selection and Monte Carlo tolerance study.

What the script does
--------------------
  1. Start from the Cohn-table design converted to lumped L and C (pi sections).
  2. Optionally fine-tune the seven design values with an optimizer.
  3. Look at the neighbouring E-series values of each design value and score every
     combination on the nominal circuit.
  4. Run a Monte Carlo tolerance study on the best-scoring combinations and rank
     them by yield (fraction of random builds that meet the spec).
  5. Report the winner: part sensitivity, power and current at the input power
     (matched and with a faulted output), and a bill of materials.

How the circuit is solved
-------------------------
The script does not call scikit-rf. It solves the same circuit with nodal analysis in
NumPy, written so that all Monte Carlo runs and all frequencies are solved in one
vectorised call. Use scikit-rf or LTspice to cross-check the winning design.

Topology (same as the scikit-rf netlist; section B at the input, section A at the outputs):

    port 1 --LB_1--(JA)--LA_1-- port 2      shunt capacitors to ground:
           `-LB_2--(JB)--LA_2-- port 3        CB_0 at the input node,
                                              CB_1 and CA_11 at JA, CB_2 and CA_21 at JB,
    RB sits between JA and JB                 CA_12 at port 2, CA_22 at port 3
    RA sits between port 2 and port 3

All part models and tolerances below are ASSUMPTIONS. Replace them with datasheet
values for the parts you can actually buy and assemble.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize


# ============================================================================
# 1. SETTINGS (edit these)
# ============================================================================
Z0 = 50.0                                         # port impedance, ohm
CENTER_FREQ_HZ = 165e6                            # design centre frequency
BAND_FREQS_HZ = np.linspace(140e6, 190e6, 51)     # band the spec applies to
PLOT_FREQS_HZ = np.linspace(100e6, 250e6, 301)    # wider sweep, plots only
INPUT_POWER_W = 2.0                               # available source power


@dataclass(frozen=True)
class Spec:
    """Pass/fail limits, applied to the worst case over the band (all values in dB)."""
    reflection_db: float = -20.0       # S11, S22 and S33 must be at or below this
    isolation_db: float = -20.0        # S23 must be at or below this
    insertion_loss_db: float = 3.5     # worst of -S21 and -S31 must be at or below this
    imbalance_db: float = 0.3          # |S21 - S31| must be at or below this


@dataclass(frozen=True)
class PartModel:
    """Non-ideal behaviour of real parts (ASSUMED values, replace with datasheet data)."""
    inductor_q: float = 60.0                  # quality factor at CENTER_FREQ_HZ
    inductor_parallel_c: float = 0.15e-12     # F, winding capacitance across the inductor
    capacitor_esr: float = 0.15               # ohm, series resistance
    capacitor_esl: float = 0.6e-9             # H, series inductance

    @classmethod
    def ideal(cls) -> "PartModel":
        return cls(inductor_q=float("inf"), inductor_parallel_c=0.0,
                   capacitor_esr=0.0, capacitor_esl=0.0)


@dataclass(frozen=True)
class Tolerances:
    """Part tolerances. Capacitors use the larger of a relative and an absolute tolerance,
    because small capacitors are specified in pF (for example +-0.1 pF), not in percent."""
    inductor: float = 0.02
    resistor: float = 0.01
    capacitor_relative: float = 0.02
    capacitor_absolute: float = 0.1e-12

    def relative(self, kind: str, value: float) -> float:
        """Relative tolerance (fraction) of one part of the given kind and nominal value."""
        if kind == "inductor":
            return self.inductor
        if kind == "resistor":
            return self.resistor
        return max(self.capacitor_relative, self.capacitor_absolute / value)


@dataclass(frozen=True)
class CohnTable:
    """Normalised Cohn table values (divided by Z0), as in the scikit-rf netlist."""
    z1_n: float = 1.1998     # section A (outputs): line impedance
    r1_n: float = 5.3163     # section A: isolation resistor
    z2_n: float = 1.6670     # section B (input): line impedance
    r2_n: float = 1.8643     # section B: isolation resistor


@dataclass(frozen=True)
class SearchSettings:
    optimize_first: bool = True      # False: take E-series neighbours of the Cohn values directly
    optimizer: str = 'minimax'
    neighbours_per_side: int = 1     # E-series values kept below and above each design value
    shortlist_size: int = 10         # best nominal combinations sent to the Monte Carlo


@dataclass(frozen=True)
class MonteCarloSettings:
    n_runs: int = 2000
    seed: int = 1
    distribution: str = "gauss"      # "gauss" (sigma = tolerance / 2, clipped) or "uniform"


SPEC = Spec()
PART_MODEL = PartModel()
TOLERANCES = Tolerances()
COHN = CohnTable()
SEARCH = SearchSettings()
MONTE_CARLO = MonteCarloSettings()

# The scikit-rf netlist uses RB = r2_n * Z0 but RA = r1_n * ZA. "ZA" keeps that,
# "Z0" normalises both resistors to Z0.
RA_NORMALISATION = "Z0"


# ============================================================================
# 2. CIRCUIT DEFINITION
# ============================================================================
class Node(IntEnum):
    """Circuit nodes (ground is the implicit reference)."""
    IN = 0      # port 1
    JA = 1      # junction between the two sections, arm toward port 2
    JB = 2      # junction between the two sections, arm toward port 3
    O2 = 3      # port 2
    O3 = 4      # port 3


N_NODES = 5
PORT_NODES = [Node.IN, Node.O2, Node.O3]       # nodes of port 1, 2 and 3


class Branch(NamedTuple):
    part: str
    kind: str                  # "inductor", "capacitor" or "resistor"
    node_a: int
    node_b: int | None         # None means ground


BRANCHES = [
    Branch("LB_1", "inductor", Node.IN, Node.JA),
    Branch("LB_2", "inductor", Node.IN, Node.JB),
    Branch("LA_1", "inductor", Node.JA, Node.O2),
    Branch("LA_2", "inductor", Node.JB, Node.O3),
    Branch("CB_0", "capacitor", Node.IN, None),
    Branch("CB_1", "capacitor", Node.JA, None),
    Branch("CA_11", "capacitor", Node.JA, None),
    Branch("CB_2", "capacitor", Node.JB, None),
    Branch("CA_21", "capacitor", Node.JB, None),
    Branch("CA_12", "capacitor", Node.O2, None),
    Branch("CA_22", "capacitor", Node.O3, None),
    Branch("RB", "resistor", Node.JA, Node.JB),
    Branch("RA", "resistor", Node.O2, Node.O3),
]


class DesignVariable(NamedTuple):
    kind: str
    parts: list[str]           # physical parts that share this nominal value


# The seven values the design is described by. In the Monte Carlo every physical
# part still gets its own random deviation.
DESIGN_VARIABLES = {
    "LB": DesignVariable("inductor", ["LB_1", "LB_2"]),
    "LA": DesignVariable("inductor", ["LA_1", "LA_2"]),
    "C0": DesignVariable("capacitor", ["CB_0"]),            # input capacitor, nominally 2 * CB
    "CB": DesignVariable("capacitor", ["CB_1", "CB_2"]),
    "CA": DesignVariable("capacitor", ["CA_11", "CA_12", "CA_21", "CA_22"]),
    "RB": DesignVariable("resistor", ["RB"]),
    "RA": DesignVariable("resistor", ["RA"]),
}
VARIABLES = list(DESIGN_VARIABLES)
PART_NAMES = [part for var in VARIABLES for part in DESIGN_VARIABLES[var].parts]
VARIABLE_OF_PART = {part: var for var in VARIABLES for part in DESIGN_VARIABLES[var].parts}
KIND_OF_PART = {b.part: b.kind for b in BRANCHES}

# Standard value series per part kind: (series in decade 1-10, unit)
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9,
       4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
E96 = list(np.round(10 ** (np.arange(96) / 96), 2))       # check against the standard table
STANDARD_SERIES = {"inductor": (E24, 1e-9), "capacitor": (E24, 1e-12), "resistor": (E96, 1.0)}

# A "design" maps each design variable to a value in SI units. The value is a float for a
# single design, or an array when many designs are evaluated in one batch.


def part_values_from_design(design: dict) -> dict:
    """Give every physical part the nominal value of its design variable (arrays of shape (N,))."""
    return {part: np.atleast_1d(design[var])
            for var in VARIABLES for part in DESIGN_VARIABLES[var].parts}


# ============================================================================
# 3. CIRCUIT SOLVER (nodal analysis)
# ============================================================================
def inductor_series_impedance(inductance, omega, model: PartModel):
    """Inductor with a series loss resistance set by Q (specified at the centre frequency)."""
    series_resistance = 2 * np.pi * CENTER_FREQ_HZ * inductance / model.inductor_q
    return series_resistance + 1j * omega * inductance


def branch_admittance(kind: str, value, omega, model: PartModel):
    """Admittance of one part. value has shape (N, 1) and omega (1, F); result is (N, F)."""
    if kind == "inductor":
        return 1.0 / inductor_series_impedance(value, omega, model) \
               + 1j * omega * model.inductor_parallel_c
    if kind == "capacitor":
        impedance = model.capacitor_esr + 1j * omega * model.capacitor_esl \
                    + 1.0 / (1j * omega * value)
        return 1.0 / impedance
    return (1.0 / value) * np.ones_like(omega) + 0j           # resistor


def add_branch_to_matrix(Y, node_a: int, node_b: int | None, y) -> None:
    """Stamp one branch admittance into the nodal matrix (node_b None = connected to ground)."""
    Y[..., node_a, node_a] += y
    if node_b is not None:
        Y[..., node_b, node_b] += y
        Y[..., node_a, node_b] -= y
        Y[..., node_b, node_a] -= y


def build_admittance_matrix(part_values: dict, freqs_hz, model: PartModel):
    """Nodal admittance matrix, shape (runs, frequencies, nodes, nodes)."""
    omega = 2 * np.pi * np.asarray(freqs_hz)[None, :]                    # (1, F)
    n_runs = len(next(iter(part_values.values())))
    Y = np.zeros((n_runs, omega.shape[1], N_NODES, N_NODES), dtype=complex)
    for branch in BRANCHES:
        values = np.asarray(part_values[branch.part], dtype=float)[:, None]    # (N, 1)
        y = branch_admittance(branch.kind, values, omega, model)               # (N, F)
        add_branch_to_matrix(Y, branch.node_a, branch.node_b, y)
    return Y


def solve_s_parameters(part_values: dict, freqs_hz, model: PartModel):
    """3-port S-parameters with Z0 on every port, shape (runs, frequencies, 3, 3)."""
    Y = build_admittance_matrix(part_values, freqs_hz, model)
    for node in PORT_NODES:
        Y[..., node, node] += 1.0 / Z0                                   # port termination

    # Column k drives port k with a 1 V source behind Z0, i.e. a current of 1/Z0 into its node.
    sources = np.zeros((N_NODES, 3), dtype=complex)
    for port, node in enumerate(PORT_NODES):
        sources[node, port] = 1.0 / Z0

    voltages = np.linalg.solve(Y, np.broadcast_to(sources, Y.shape[:-2] + sources.shape))
    return 2.0 * voltages[..., PORT_NODES, :] - np.eye(3)


# ============================================================================
# 4. METRICS, SPEC CHECK AND COST
# ============================================================================
@dataclass
class Metrics:
    """Worst case over the band for each run (arrays with one entry per run, in dB)."""
    reflection_db: np.ndarray        # worst of S11, S22, S33
    isolation_db: np.ndarray         # S23
    insertion_loss_db: np.ndarray    # worst of -S21, -S31
    imbalance_db: np.ndarray         # |S21 - S31|


def compute_metrics(S) -> Metrics:
    magnitude_db = 20 * np.log10(np.abs(S) + 1e-12)
    s11, s22, s33 = (magnitude_db[..., k, k] for k in range(3))
    s21, s31, s23 = magnitude_db[..., 1, 0], magnitude_db[..., 2, 0], magnitude_db[..., 1, 2]
    return Metrics(
        reflection_db=np.maximum.reduce([s11, s22, s33]).max(axis=-1),
        isolation_db=s23.max(axis=-1),
        insertion_loss_db=(-np.minimum(s21, s31)).max(axis=-1),
        imbalance_db=np.abs(s21 - s31).max(axis=-1),
    )


def meets_spec(metrics: Metrics, spec: Spec):
    """Boolean array: True for the runs that pass every limit."""
    return ((metrics.reflection_db <= spec.reflection_db)
            & (metrics.isolation_db <= spec.isolation_db)
            & (metrics.insertion_loss_db <= spec.insertion_loss_db)
            & (metrics.imbalance_db <= spec.imbalance_db))


def nominal_cost(metrics: Metrics, spec: Spec):
    """Score for ranking designs (dB, lower is better): the worst reflection or isolation,
    plus a penalty when insertion loss exceeds its limit. Loss from finite inductor Q is a
    floor of roughly 0.3 dB that is similar for near-optimal candidates (it still varies
    with the L values and with mismatch), so loss is only penalised beyond the spec.
    The weight 5.0 is a judgement call: 0.2 dB of excess loss counts as much as 1 dB of
    worse reflection or isolation."""
    excess_loss = np.maximum(0.0, metrics.insertion_loss_db - spec.insertion_loss_db)
    return np.maximum(metrics.reflection_db, metrics.isolation_db) + 5.0 * excess_loss


def evaluate_designs(design: dict, model: PartModel, freqs_hz=BAND_FREQS_HZ) -> Metrics:
    """Metrics of one design (floats) or a batch of designs (arrays) with nominal part values."""
    S = solve_s_parameters(part_values_from_design(design), freqs_hz, model)
    return compute_metrics(S)


# ============================================================================
# 5. STEP 1: STARTING POINT AND CONTINUOUS OPTIMIZATION
# ============================================================================
def cohn_starting_design(table: CohnTable = COHN) -> dict:
    """Convert the Cohn table values into pi-section L and C values at the centre frequency
    (same maths as the scikit-rf netlist)."""
    omega0 = 2 * np.pi * CENTER_FREQ_HZ
    z_b, z_a = table.z2_n * Z0, table.z1_n * Z0                  # line impedances, B then A
    c_b, c_a = 1 / (omega0 * z_b), 1 / (omega0 * z_a)
    return {
        "LB": z_b / omega0,
        "LA": z_a / omega0,
        "C0": 2 * c_b,                                           # input capacitor serves both arms
        "CB": c_b,
        "CA": c_a,
        "RB": table.r2_n * Z0,
        "RA": table.r1_n * (z_a if RA_NORMALISATION == "ZA" else Z0),
    }


def optimize_design(start: dict, model: PartModel, spec: Spec, restarts: int = 3, seed: int = 0):
    """Nelder-Mead search for the seven values, varied as multiplicative factors of `start`.
    The first run starts at `start`; the others start from small random perturbations."""
    rng = np.random.default_rng(seed)
    base = np.array([start[var] for var in VARIABLES])

    def design_from(log_factors):
        return dict(zip(VARIABLES, base * np.exp(log_factors)))

    def score(log_factors):
        return float(nominal_cost(evaluate_designs(design_from(log_factors), model), spec)[0])

    best = None
    for attempt in range(restarts):
        first = np.zeros(len(base)) if attempt == 0 else rng.normal(0, 0.1, len(base))
        result = minimize(score, first, method="Nelder-Mead",
                          options=dict(maxiter=1500, xatol=1e-4, fatol=1e-4))
        if best is None or result.fun < best.fun:
            best = result
    return design_from(best.x), best.fun


def optimize_design_minimax(start: dict, model: PartModel, spec: Spec, restarts: int = 5,
                            seed: int = 0, max_log_change: float = 1.0):
    """Gradient-based alternative to optimize_design (epigraph form of the minimax problem).

    Minimise t subject to: S11, S22, S33 and S23 (dB) <= t at every band frequency, and
    -S21, -S31 <= the insertion-loss limit at every band frequency. All functions are
    smooth, so SLSQP can be used. Values are varied as exp(x) factors of `start`, with
    |x| <= max_log_change. Returns (design, t), where t is the worst reflection/isolation in dB."""
    base = np.array([start[var] for var in VARIABLES])

    def band_magnitudes_db(x):                                   # shape (frequencies, 3, 3)
        design = dict(zip(VARIABLES, base * np.exp(x)))
        S = solve_s_parameters(part_values_from_design(design), BAND_FREQS_HZ, model)[0]
        return 20 * np.log10(np.abs(S) + 1e-12)

    def reflection_isolation_db(x):
        db = band_magnitudes_db(x)
        return np.concatenate([db[:, 0, 0], db[:, 1, 1], db[:, 2, 2], db[:, 1, 2]])

    def constraints(z):                                          # every entry must be >= 0
        x, t = z[:-1], z[-1]
        db = band_magnitudes_db(x)
        return np.concatenate([
            t - db[:, 0, 0], t - db[:, 1, 1], t - db[:, 2, 2], t - db[:, 1, 2],
            spec.insertion_loss_db + db[:, 1, 0],                # -S21 <= loss limit
            spec.insertion_loss_db + db[:, 2, 0],                # -S31 <= loss limit
        ])

    rng = np.random.default_rng(seed)
    best = None
    for attempt in range(restarts):
        x0 = np.zeros(len(base)) if attempt == 0 else rng.normal(0, 0.1, len(base))
        z0 = np.append(x0, reflection_isolation_db(x0).max())
        result = minimize(lambda z: z[-1], z0, method="SLSQP",
                          constraints=[{"type": "ineq", "fun": constraints}],
                          bounds=[(-max_log_change, max_log_change)] * len(base) + [(-80, 0)],
                          options=dict(maxiter=300, ftol=1e-7))
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:                                             # no restart converged
        print("WARNING: minimax optimizer did not converge; falling back to Nelder-Mead")
        return optimize_design(start, model, spec)
    return dict(zip(VARIABLES, base * np.exp(best.x[:-1]))), best.fun


# ============================================================================
# 6. STEP 2 AND 3: E-SERIES CANDIDATES AND EXHAUSTIVE NOMINAL SEARCH
# ============================================================================
def eseries_neighbours(value: float, kind: str, per_side: int) -> np.ndarray:
    """The `per_side` standard values at or below `value` and the `per_side` values above it."""
    series, unit = STANDARD_SERIES[kind]
    decade = int(np.floor(np.log10(value / unit)))
    candidates = np.array(sorted(s * unit * 10.0 ** d
                                 for d in range(decade - 1, decade + 2) for s in series))
    below = candidates[candidates <= value * (1 + 1e-9)][-per_side:]
    above = candidates[candidates > value * (1 + 1e-9)][:per_side]
    return np.concatenate([below, above])


def rank_combinations_by_nominal_cost(center: dict, model: PartModel, spec: Spec,
                                      per_side: int, chunk_size: int = 4096):
    """Evaluate every combination of E-series neighbours of `center` on the nominal circuit.
    Returns (combinations, costs), both sorted from best to worst; each combination is an
    array of values in the order of VARIABLES."""
    value_lists = [eseries_neighbours(center[var], DESIGN_VARIABLES[var].kind, per_side)
                   for var in VARIABLES]
    combinations = np.array(list(itertools.product(*value_lists)))
    costs = np.empty(len(combinations))
    for start in range(0, len(combinations), chunk_size):         # chunks limit memory use
        block = combinations[start:start + chunk_size]
        design = {var: block[:, i] for i, var in enumerate(VARIABLES)}
        costs[start:start + chunk_size] = nominal_cost(evaluate_designs(design, model), spec)
    order = np.argsort(costs)
    return combinations[order], costs[order]


# ============================================================================
# 7. STEP 4: MONTE CARLO
# ============================================================================
def draw_deviations(settings: MonteCarloSettings, n_parts: int) -> np.ndarray:
    """Random numbers in [-1, 1], shape (runs, parts). +-1 means a part at its tolerance limit.
    Draw them once and reuse them for every candidate, so that differences in yield come
    from the designs and not from luck."""
    rng = np.random.default_rng(settings.seed)
    shape = (settings.n_runs, n_parts)
    if settings.distribution == "uniform":
        return rng.uniform(-1, 1, shape)
    return np.clip(rng.normal(0, 0.5, shape), -1, 1)


def sample_part_values(design: dict, deviations: np.ndarray, tolerances: Tolerances) -> dict:
    """Value of every physical part in every run: nominal * (1 + tolerance * deviation)."""
    values = {}
    for column, part in enumerate(PART_NAMES):
        nominal = design[VARIABLE_OF_PART[part]]
        tolerance = tolerances.relative(KIND_OF_PART[part], nominal)
        values[part] = nominal * (1 + tolerance * deviations[:, column])
    return values


@dataclass
class Candidate:
    """One E-series combination together with its Monte Carlo outcome."""
    design: dict
    nominal_cost: float
    metrics: Metrics
    passed: np.ndarray                  # boolean per run

    @property
    def yield_fraction(self) -> float:
        return float(self.passed.mean())


def run_monte_carlo(design: dict, nominal_cost_db: float, deviations: np.ndarray,
                    model: PartModel, spec: Spec, tolerances: Tolerances) -> Candidate:
    part_values = sample_part_values(design, deviations, tolerances)
    S = solve_s_parameters(part_values, BAND_FREQS_HZ, model)
    metrics = compute_metrics(S)
    return Candidate(design, nominal_cost_db, metrics, meets_spec(metrics, spec))


def rank_candidates_by_yield(combinations, costs, deviations, model, spec, tolerances,
                             shortlist_size: int) -> list[Candidate]:
    """Monte Carlo on the best nominal combinations; sorted by yield, then by nominal cost."""
    candidates = []
    for values, cost in zip(combinations[:shortlist_size], costs[:shortlist_size]):
        design = dict(zip(VARIABLES, values))
        candidates.append(run_monte_carlo(design, cost, deviations, model, spec, tolerances))
    candidates.sort(key=lambda c: (c.yield_fraction, -c.nominal_cost), reverse=True)
    return candidates


# ============================================================================
# 8. STEP 5: REPORTS FOR THE WINNING DESIGN
# ============================================================================
def spearman_correlation(x, y) -> float:
    """Rank correlation: how consistently y rises or falls with x."""
    return np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]


def part_sensitivity(deviations: np.ndarray, metrics: Metrics) -> list[tuple[float, str]]:
    """(correlation, part) pairs, strongest influence first. The correlation is between a
    part's random deviation and the worst of reflection and isolation."""
    worst = np.maximum(metrics.reflection_db, metrics.isolation_db)
    pairs = [(spearman_correlation(deviations[:, column], worst), part)
             for column, part in enumerate(PART_NAMES)]
    return sorted(pairs, key=lambda pair: -abs(pair[0]))


def node_voltages_with_faulty_port3(part_values: dict, freqs_hz, model: PartModel,
                                    port3_admittance: float):
    """RMS node voltages (frequencies x nodes) when port 1 delivers INPUT_POWER_W, port 2 has a
    Z0 load and port 3 is terminated with the given admittance (open: 0, short: very large)."""
    Y = build_admittance_matrix(part_values, freqs_hz, model)
    Y[..., Node.IN, Node.IN] += 1 / Z0                           # source impedance
    Y[..., Node.O2, Node.O2] += 1 / Z0
    Y[..., Node.O3, Node.O3] += port3_admittance
    drive = np.zeros(N_NODES, dtype=complex)
    drive[Node.IN] = 1 / Z0                                      # 1 V source behind Z0
    rhs = np.broadcast_to(drive, Y.shape[:-1])[..., None]
    unit_source_voltages = np.linalg.solve(Y, rhs)[0, :, :, 0]
    source_emf = 2 * np.sqrt(Z0 * INPUT_POWER_W)                 # EMF that delivers the power
    return unit_source_voltages * source_emf


def print_power_report(design: dict, model: PartModel, freqs_hz=BAND_FREQS_HZ) -> None:
    """Worst case over the band of resistor power, inductor current and voltage."""
    part_values = {part: np.array([design[VARIABLE_OF_PART[part]]]) for part in PART_NAMES}
    omega = 2 * np.pi * freqs_hz
    z_inductor_b = inductor_series_impedance(design["LB"], omega, model)
    z_inductor_a = inductor_series_impedance(design["LA"], omega, model)

    print(f"\nWorst case over the band at {INPUT_POWER_W} W input power (RMS values)")
    print(f"{'case':<14}{'P(RB) W':>9}{'P(RA) W':>9}{'I LB A':>9}{'I LA A':>9}{'Vpk max V':>11}")
    cases = [("matched", 1 / Z0), ("port 3 open", 0.0), ("port 3 short", 1e6)]
    for name, port3_admittance in cases:
        V = node_voltages_with_faulty_port3(part_values, freqs_hz, model, port3_admittance)
        power_rb = (abs(V[:, Node.JA] - V[:, Node.JB]) ** 2 / design["RB"]).max()
        power_ra = (abs(V[:, Node.O2] - V[:, Node.O3]) ** 2 / design["RA"]).max()
        arm_voltage_b = np.maximum(abs(V[:, Node.IN] - V[:, Node.JA]),
                                   abs(V[:, Node.IN] - V[:, Node.JB]))
        arm_voltage_a = np.maximum(abs(V[:, Node.JA] - V[:, Node.O2]),
                                   abs(V[:, Node.JB] - V[:, Node.O3]))
        current_lb = (arm_voltage_b / abs(z_inductor_b)).max()
        current_la = (arm_voltage_a / abs(z_inductor_a)).max()
        peak_voltage = np.sqrt(2) * abs(V).max()
        print(f"{name:<14}{power_rb:9.3f}{power_ra:9.3f}{current_lb:9.3f}"
              f"{current_la:9.3f}{peak_voltage:11.1f}")


def plot_envelope(design: dict, deviations: np.ndarray, model: PartModel, tolerances: Tolerances,
                  filename: str = "mc_envelope.png", n_curves: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_S = solve_s_parameters(
        sample_part_values(design, deviations[:n_curves], tolerances), PLOT_FREQS_HZ, model)
    nominal_S = solve_s_parameters(part_values_from_design(design), PLOT_FREQS_HZ, model)

    fig, ax = plt.subplots(figsize=(8, 5))
    traces = [((0, 0), "C0", "S11"), ((1, 1), "C3", "S22"), ((1, 2), "C2", "S23")]
    for (row, col), color, label in traces:
        ax.plot(PLOT_FREQS_HZ / 1e6, 20 * np.log10(abs(random_S[:, :, row, col])).T,
                color, alpha=0.04)
        ax.plot(PLOT_FREQS_HZ / 1e6, 20 * np.log10(abs(nominal_S[0, :, row, col])),
                color, lw=2, label=label)
    ax.axvspan(BAND_FREQS_HZ[0] / 1e6, BAND_FREQS_HZ[-1] / 1e6, color="c", alpha=0.12)
    ax.axhline(SPEC.reflection_db, color="k", ls="--", lw=1)
    ax.set_ylim(-70, 0)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend()
    ax.set_title("Monte Carlo envelope (nominal in bold)")
    fig.savefig(filename, dpi=150, bbox_inches="tight")


# ============================================================================
# 9. PRINTING HELPERS AND MAIN
# ============================================================================
UNIT_OF_KIND = {"inductor": ("nH", 1e9), "capacitor": ("pF", 1e12), "resistor": ("ohm", 1.0)}


def format_value(variable: str, value: float) -> str:
    """Value of a design variable with its unit, for example '75 nH'."""
    unit, scale = UNIT_OF_KIND[DESIGN_VARIABLES[variable].kind]
    return f"{value * scale:.4g} {unit}"


def format_design(design: dict) -> str:
    return ", ".join(f"{var}={format_value(var, design[var])}" for var in VARIABLES)


def print_bom(design: dict, tolerances: Tolerances) -> None:
    print("\nBOM (value, tolerance used in the Monte Carlo):")
    for part in PART_NAMES:
        var = VARIABLE_OF_PART[part]
        tolerance = tolerances.relative(KIND_OF_PART[part], design[var])
        print(f"  {part:<6} {format_value(var, design[var]):<12} +-{100 * tolerance:.1f}%")
    print("\nValues to paste into your scikit-rf netlist (SI units):")
    for var in VARIABLES:
        print(f"  {var} = {design[var]:.4g}")


def print_starting_point_report(start: dict, spec: Spec) -> None:
    print("Cohn start: " + format_design(start))
    for label, model in [("ideal parts", PartModel.ideal()), ("real-part model", PART_MODEL)]:
        m = evaluate_designs(start, model)
        print(f"  {label:<16} worst RL {m.reflection_db[0]:6.1f} dB | "
              f"iso {m.isolation_db[0]:6.1f} dB | IL {m.insertion_loss_db[0]:.2f} dB")


def print_shortlist(candidates: list[Candidate]) -> None:
    print(f"\nShortlist ({MONTE_CARLO.n_runs} runs; spec: reflection <= {SPEC.reflection_db} dB, "
          f"isolation <= {SPEC.isolation_db} dB, loss <= {SPEC.insertion_loss_db} dB, "
          f"imbalance <= {SPEC.imbalance_db} dB):")
    for c in candidates:
        print(f"  yield {100 * c.yield_fraction:5.1f}%  nominal {c.nominal_cost:6.1f} dB  "
              f"p95 RL {np.percentile(c.metrics.reflection_db, 95):6.1f}  "
              f"p95 iso {np.percentile(c.metrics.isolation_db, 95):6.1f}  | "
              + format_design(c.design))


def main() -> None:
    model, spec, tolerances = PART_MODEL, SPEC, TOLERANCES

    # Step 0: the Cohn design converted to lumped parts
    start = cohn_starting_design()
    print_starting_point_report(start, spec)

    # Step 1: continuous optimum (or stay on the Cohn values)
    if SEARCH.optimize_first:
        if SEARCH.optimizer == "minimax":
            centre, worst_db = optimize_design_minimax(start, model, spec)
        else:
            centre, worst_db = optimize_design(start, model, spec)
        print(f"\nContinuous optimum ({SEARCH.optimizer}, cost {worst_db:.1f} dB):")
    else:
        centre = start
        print("\nContinuous optimization skipped; using the Cohn values")
    print("  " + format_design(centre))

    # Steps 2-3: E-series neighbours, exhaustive search on the nominal circuit
    combinations, costs = rank_combinations_by_nominal_cost(
        centre, model, spec, SEARCH.neighbours_per_side)
    print(f"\n{len(combinations)} combinations evaluated; best nominal cost {costs[0]:.1f} dB")

    # Step 4: Monte Carlo on the best combinations, identical random draws for all
    deviations = draw_deviations(MONTE_CARLO, n_parts=len(PART_NAMES))
    candidates = rank_candidates_by_yield(combinations, costs, deviations, model, spec,
                                          tolerances, SEARCH.shortlist_size)
    print_shortlist(candidates)

    # Step 5: reports for the winner
    winner = candidates[0]
    print("\nWinner:", format_design(winner.design))
    print("Sensitivity (Spearman correlation with worst of RL/isolation, top 6):")
    for correlation, part in part_sensitivity(deviations, winner.metrics)[:6]:
        print(f"  {part:<6} {correlation:+.2f}")
    print_power_report(winner.design, model)
    print_bom(winner.design, tolerances)
    plot_envelope(winner.design, deviations, model, tolerances)
    print("\nSaved mc_envelope.png")


if __name__ == "__main__":
    main()
