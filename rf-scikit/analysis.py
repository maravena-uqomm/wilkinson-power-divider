import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from skrf.circuit import Circuit
from circuit_definition import COHN_TABLE_N_2_15, Z0_REFERENCE, FREQ_BAND_DEFINITION, solve_s_parameter, IDEAL_COMPONENT_VALUES, Spec
from scipy.optimize import minimize

N_RUNS = 2000

FREQUENCY_RANGE = rf.Frequency(
        start=FREQ_BAND_DEFINITION['start'],
        stop=FREQ_BAND_DEFINITION['end'],
        npoints=FREQ_BAND_DEFINITION['npoints'],
        unit=FREQ_BAND_DEFINITION['unit']
    )

SPEC = Spec()

def reference_wilkinson_cohn(plot_graph=False):
    # Generates a reference circuit using scikit-rf and returns its scattering matrix
    beta = FREQUENCY_RANGE.w/rf.constants.c

    # Branches - B section
    RB = COHN_TABLE_N_2_15['rB']*Z0_REFERENCE
    rb_line = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=RB)
    rb_resistor = rb_line.resistor(RB, name='RB')

    zb = COHN_TABLE_N_2_15['zB']*Z0_REFERENCE
    b_line = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=zb, gamma=0+beta*1j)
    d_b = b_line.theta_2_d(90, deg=True)
    b_branch_i = b_line.line(d_b, unit='m', name='b_branch_i')
    b_branch_ii = b_line.line(d_b, unit='m', name='b_branch_ii')

    # Branches -A section
    RA = COHN_TABLE_N_2_15['rA']*Z0_REFERENCE
    ra_line = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=RA)
    ra_resistor = ra_line.resistor(RA, name='RA')

    za = COHN_TABLE_N_2_15['zA']*Z0_REFERENCE
    a_line = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=za, gamma=0+beta*1j)
    d_a = a_line.theta_2_d(90, deg=True)
    a_branch_i = a_line.line(d_a, unit='m', name='a_branch_i')
    a_branch_ii = a_line.line(d_a, unit='m', name='a_branch_ii')

    # Ports
    port1 = Circuit.Port(FREQUENCY_RANGE, name='port1', z0=Z0_REFERENCE)
    port2 = Circuit.Port(FREQUENCY_RANGE, name='port2', z0=Z0_REFERENCE)
    port3 = Circuit.Port(FREQUENCY_RANGE, name='port3', z0=Z0_REFERENCE)

    # Connections
    connections = [
        [(port1, 0), (b_branch_i, 0), (b_branch_ii, 0)],
        [(b_branch_i, 1), (rb_resistor, 0), (a_branch_i, 0)],
        [(b_branch_ii, 1), (rb_resistor, 1), (a_branch_ii, 0)],
        [(port2, 0), (ra_resistor, 0), (a_branch_i, 1)],
        [(port3, 0), (ra_resistor, 1), (a_branch_ii, 1)]
    ]

    ideal_circuit = Circuit(connections)
    
    if(plot_graph):
        ideal_circuit.plot_graph(network_labels=True, edge_labels=True, port_labels=True, port_fontize=2)
        plt.show()

    scattering_matrix = ideal_circuit.network.s
    s_db = 20*np.log10(np.abs(scattering_matrix))

    return ideal_circuit, s_db



def reference_lumped_wilkinson_cohn():
    # B-values
    zb = COHN_TABLE_N_2_15['zB']*Z0_REFERENCE
    LB = IDEAL_COMPONENT_VALUES['LB']
    CB = IDEAL_COMPONENT_VALUES['CB']
    RB = COHN_TABLE_N_2_15['rB']*Z0_REFERENCE
    # B-Branches
    b_section = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=zb)
    B_L1 = b_section.inductor(LB, name='LB_1')
    B_C1 = b_section.capacitor(CB, name='CB_1')
    B_L2 = b_section.inductor(LB, name='LB_2')
    B_C2 = b_section.capacitor(CB, name='CB_2')
    B_C0 = b_section.capacitor(2*CB, name='CB_0')

    b_res_media = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=RB)
    B_R = b_res_media.resistor(RB, name='RB')

    # A-values
    za = COHN_TABLE_N_2_15['zA']*Z0_REFERENCE
    LA = IDEAL_COMPONENT_VALUES['LA']
    CA = IDEAL_COMPONENT_VALUES['CA']
    RA = COHN_TABLE_N_2_15['rA']*Z0_REFERENCE
    # A-branches
    a_section = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=za)
    A_C11 = a_section.capacitor(CA, name='CA_11')
    A_L1 = a_section.inductor(LA, name='LA_1')
    A_C12 = a_section.capacitor(CA, name='CA_12')

    A_C21 = a_section.capacitor(CA, name='CA_21')
    A_L2 = a_section.inductor(LA, name='LA_2')
    A_C22 = a_section.capacitor(CA, name='CA_22')

    a_res_media = rf.media.DefinedGammaZ0(frequency=FREQUENCY_RANGE, z0=RA)
    A_R = a_res_media.resistor(RA, name='RA')

    # GND
    gnd = Circuit.Ground(FREQUENCY_RANGE, name='gnd', z0=Z0_REFERENCE)

    # Ports
    port1 = Circuit.Port(FREQUENCY_RANGE, name='port1', z0=50)
    port2 = Circuit.Port(FREQUENCY_RANGE, name='port2', z0=50)
    port3 = Circuit.Port(FREQUENCY_RANGE, name='port3', z0=50)

    connections = [
        [(port1, 0), (B_C0, 0), (B_L1, 0), (B_L2, 0)],
        [(B_R, 0), (B_L1, 1), (B_C1, 0), (A_C11, 0), (A_L1, 0)],
        [(B_R, 1), (B_L2, 1), (B_C2, 0), (A_C21, 0), (A_L2, 0)],
        [(A_L1, 1), (A_C12, 0), (A_R, 0), (port2, 0)],
        [(A_L2, 1), (A_C22, 0), (A_R, 1), (port3, 0)],
        [(gnd, 0), (B_C0, 1), (B_C1, 1), (B_C2, 1), (A_C11, 1), (A_C12, 1), (A_C21,1), (A_C22,1)]
    ]

    C_two_section = Circuit(connections)
    scattering_matrix = C_two_section.network.s
    s_db = 20*np.log10(np.abs(scattering_matrix))

    return C_two_section, s_db



def plot_s_comparison(s_vector, label_vector):
    freq_vector = FREQUENCY_RANGE.f * 1e-6
    fig, axs = plt.subplots(2, 3, figsize=(10,8))
    for i, s_matrix in enumerate(s_vector):
        i_s11 = s_matrix[..., 0, 0]
        axs[0, 0].plot(freq_vector, i_s11, label=label_vector[i])
        axs[0,0].set_title("S11")
        i_s22 = s_matrix[..., 1, 1]
        axs[0, 1].plot(freq_vector, i_s22, label=label_vector[i])
        axs[0, 1].set_title("S22")
        i_s33 = s_matrix[..., 2, 2]
        axs[0, 2].plot(freq_vector, i_s33, label=label_vector[i])
        axs[0, 2].set_title("S33")
        i_s21 = s_matrix[..., 1, 0]
        axs[1, 0].plot(freq_vector, i_s21, label=label_vector[i])
        axs[1, 0].set_title("S21")
        i_s31 = s_matrix[..., 2, 0]
        axs[1, 1].plot(freq_vector, i_s31, label=label_vector[i])
        axs[1, 1].set_title("S31")
        i_s32 = s_matrix[..., 2, 1]
        axs[1, 2].plot(freq_vector, i_s32, label=label_vector[i])
        axs[1, 2].set_title("S32")

    for ax in axs.flat:
        ax.axvspan(145, 186, color='c', alpha=0.2)
        ax.grid()

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)

    return fig, axs


def optimize_mimimax(frequency_range: rf.Frequency, spec: object,  seed: int = 0, attempts: int = 5, max_log_change=1.0):
    base_values = np.array([IDEAL_COMPONENT_VALUES[var] for var in IDEAL_COMPONENT_VALUES])
    rng = np.random.default_rng(seed)
    best = None

    def realize_values(reference, exp_vector):
        realized = {}
        for exp_i, key in enumerate(reference.keys()):
            realized[key] = reference[key]* np.exp(exp_vector[exp_i])
        return realized

    def band_magnitudes_db(x):
        realization_values = realize_values(IDEAL_COMPONENT_VALUES, x)
        S,_ = solve_s_parameter(frequency_range, realization_values)
        return 20*np.log10(np.abs(S))

    def reflection_isolation_db(x):
        db = band_magnitudes_db(x)
        return np.concatenate([db[:, 0, 0], db[:, 1, 1], db[:, 2, 2], db[:, 1, 2]])

    def constraints(z):
        x, t = z[:-1], z[-1]
        db = band_magnitudes_db(x)
        return np.concatenate([
            t - db[:, 0, 0], t - db[:, 1, 1], t - db[:, 2, 2], t - db[:, 1, 2],
            spec.insertion_loss_db + db[:, 1, 0],                # -S21 <= loss limit
            spec.insertion_loss_db + db[:, 2, 0],                # -S31 <= loss limit
        ])

    for attempt in range(attempts):
        x0 = np.zeros(len(base_values)) if attempt == 0 else rng.normal(0, 0.1, len(base_values))
        z0 =np.append(x0, reflection_isolation_db(x0).max())
        result = minimize(lambda z: z[-1], z0, method='SLSQP',
                          constraints=[{"type": "ineq", "fun": constraints}],
                          bounds=[(-max_log_change, max_log_change)]*len(base_values) + [(-80, 0)],
                          options=dict(maxiter=300, ftol=1e-7))
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None: 
        print("WARNING: minimax optimizer did not converge.")
        exit(1)
    print(f"x_opt: {best.x}")
    return realize_values(IDEAL_COMPONENT_VALUES, best.x)


def print_s_metrics(s_vector):
    print(f"S11: {np.max(s_vector[..., 0, 0]):.2f} dB | S21: {np.max(s_vector[..., 1, 0]):.2f} dB | S31: {np.max(s_vector[..., 2, 0]):.2f} dB")
    print(f"S22: {np.max(s_vector[..., 1, 1]):.2f} db | S12: {np.max(s_vector[..., 0, 1]):.2f} dB | S32: {np.max(s_vector[..., 2, 1]):.2f} dB")
    print(f"S33: {np.max(s_vector[..., 2, 2]):.2f} db | S13: {np.max(s_vector[..., 0, 2]):.2f} dB | S23: {np.max(s_vector[..., 1, 2]):.2f} dB")


if __name__ == '__main__':
    print("Starting analysis on Lumped Wilkinson Power Divider/Combiner")
    print("Generating N=2 Wilkinson-Cohn power divider")
    ideal_circuit, s_ideal_db = reference_wilkinson_cohn()
    # ideal_circuit.network.plot_s_db()
    print("####    Ideal S-metrics:")
    print_s_metrics(s_ideal_db)
    print("Generating N=2 Wilkinson-Cohn Lumped Power divider")
    ideal_lumped_circuit, s_lumped_ideal_db = reference_lumped_wilkinson_cohn()
    theorical_lumped_s, theorical_lumped_y = solve_s_parameter(FREQUENCY_RANGE, IDEAL_COMPONENT_VALUES)
    theorical_lumped_s_db = 20*np.log10(np.abs(theorical_lumped_s))
    print_s_metrics(theorical_lumped_s_db)
    print("Minimax optimization")
    optimized_components = optimize_mimimax(FREQUENCY_RANGE, SPEC)
    optimized_s, optimized_y = solve_s_parameter(FREQUENCY_RANGE, optimized_components)
    optimized_s_db = 20*np.log10(np.abs(optimized_s))
    print_s_metrics(optimized_s_db)
    
    plot_s_comparison([s_ideal_db, s_lumped_ideal_db, theorical_lumped_s_db, optimized_s_db], ['ideal', 'lumped-ideal', 'theorical', 'optimized'])
    plt.show()