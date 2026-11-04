from circuit_definition import CircuitComponent, ComponentType, Node, getWilkinsonValues, PORT_NODES, Z0_REF
import numpy as np
import matplotlib.pyplot as plt

TARGET_FREQ = 165e6
BAND_FREQS_HZ = np.linspace(140e6, 190e6, 501)

LOG_GUARD = 1e-12

# Circuit definition
w0 = TARGET_FREQ*2*np.pi
WILK_VALUES = getWilkinsonValues()
# -> Section B values
CB = 1/(w0*WILK_VALUES['zB'])
LB = WILK_VALUES['zB']/w0
RB = WILK_VALUES['rB']
# -> Section A values
CA = 1/(w0*WILK_VALUES['zA'])
LA = WILK_VALUES['zA']/w0
RA = WILK_VALUES['rA']

LumpedCircuit = [
    # Node ALFA
    CircuitComponent("C0", ComponentType.C, Node.ALFA, None, 2*CB),
    CircuitComponent("LIB", ComponentType.L, Node.ALFA, Node.BETA, LB),
    CircuitComponent("LIIB", ComponentType.L, Node.ALFA, Node.THETA, LB),
    # Node BETA
    CircuitComponent("CIB", ComponentType.C, Node.BETA, None, CB),
    CircuitComponent("RB", ComponentType.R, Node.BETA, Node.THETA, RB),
    CircuitComponent("CIA_1", ComponentType.C, Node.BETA, None, CA),
    CircuitComponent("LIA", ComponentType.L, Node.BETA, Node.DELTA, LA),
    # Node DELTA
    CircuitComponent("CIA_2", ComponentType.C, Node.DELTA, None, CA),
    CircuitComponent("RA", ComponentType.R, Node.DELTA, Node.PHI, RA),
    # Node THETA
    CircuitComponent("CIIB", ComponentType.C, Node.THETA, None, CB),
    CircuitComponent("CIIA_1", ComponentType.C, Node.THETA, None, CA),
    CircuitComponent("LIIA", ComponentType.L, Node.THETA, Node.PHI, LA),
    # Node PHI
    CircuitComponent("CIIA_2", ComponentType.C, Node.PHI, None, CA),
]

def componentAdmitance(component: CircuitComponent, omegas):
    if(component.kind == ComponentType.C):
        impedance = 1/(1j*omegas*component.value)
        return (1/impedance)
    if(component.kind == ComponentType.L):
        impedance = 1j*omegas*component.value
        return (1/impedance)
    if(component.kind == ComponentType.R): 
        return 1/component.value


def addComponentAdmitance(matrix, component:CircuitComponent, component_admitance):
    matrix[..., component.node_a, component.node_a] += component_admitance
    if component.node_b is not None:
        matrix[..., component.node_b, component.node_b] += component_admitance
        matrix[..., component.node_a, component.node_b] -= component_admitance
        matrix[..., component.node_b, component.node_a] -= component_admitance


def generateAdmitanceMatrix(freqs):
    omega_array = 2 * np.pi * freqs
    admitance_matrix = np.zeros((omega_array.shape[0], len(Node), len(Node)), dtype=complex)
    for component in LumpedCircuit:
        y_component = componentAdmitance(component, omega_array)
        addComponentAdmitance(admitance_matrix, component, y_component)
    return admitance_matrix

def solveSParameters():
    Y = generateAdmitanceMatrix(BAND_FREQS_HZ)
    # Add port impedance
    for port in PORT_NODES:
        Y[..., port, port] += 1/Z0_REF

    current_sources = np.zeros((len(Node), 3), dtype=complex)
    for port, node in enumerate(PORT_NODES):
        current_sources[node, port] = 1.0 / Z0_REF

    voltages = np.linalg.solve(Y, current_sources)
    return 2.0 * voltages[..., PORT_NODES, :] - np.eye(3)


# if __name__ == '__main__':
#     s = solveSParameters()
#     s_dB = 20*np.log10(np.abs(s) + LOG_GUARD)
#     s11 = s_dB[..., 0,0]
#     s22 = s_dB[..., 1,1]
#     s33 = s_dB[..., 2,2]

#     s21 = s_dB[..., 1,0]
#     s31 = s_dB[..., 2,0]
#     s23 = s_dB[..., 1,2]


#     fig, axs = plt.subplots(2, 1, figsize=(10,8))
#     freq_ax = BAND_FREQS_HZ*1e-6
#     axs[0].plot(freq_ax, s11)
#     axs[0].plot(freq_ax, s22)
#     axs[0].plot(freq_ax, s33, linestyle='--')
#     axs[0].plot(freq_ax, s23)

#     axs[1].plot(freq_ax, s21)
#     axs[1].plot(freq_ax, s31)
    

#     plt.show()


def reference_wilkinson_cohn():
    # Generates a reference circuit using scikit-rf and returns its scattering matrix
    



if __name__ == '__main__':
    print("Starting analysis on Lumped Wilkinson Power Divider/Combiner")
