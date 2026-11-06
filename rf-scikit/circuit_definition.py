from enum import IntEnum
from typing import NamedTuple
import numpy as np
import skrf as rf

Z0_REFERENCE= 50


FREQ_BAND_DEFINITION = {
    'start': 140,
    'end': 190,
    'f0': 165,
    'npoints': 501,
    'multiplier': 1e6,
    'unit': 'MHz'
}

class ComponentType:
    C = "capacitor"
    L = "inductor"
    R = "resistor"


class Node(IntEnum):
    ALFA    = 0 # input A
    BETA    = 1
    DELTA   = 2 # output B
    THETA   = 3
    PHI     = 4 # output C

class CircuitComponent(NamedTuple):
    name: str
    kind: ComponentType
    node_a: int
    node_b: int | None
    value_key: str

COHN_TABLE_N_2_15 = { # Reference table for N = 2 and f2/f1 = 1.5
    'zA': 1.1998,
    'zB': 1.6670,
    'rA': 5.3163,
    'rB': 1.8643
}

PORT_NODES = [
    Node.ALFA,
    Node.DELTA,
    Node.PHI
]

ZA_VALUE = COHN_TABLE_N_2_15['zA']*Z0_REFERENCE
ZB_VALUE = COHN_TABLE_N_2_15['zB']*Z0_REFERENCE
W0 = 2*np.pi*FREQ_BAND_DEFINITION['f0']*1e6

IDEAL_COMPONENT_VALUES = {
    'C0': 2/(W0 * ZB_VALUE),
    'LB': ZB_VALUE/W0,
    'CB': 1/(W0 * ZB_VALUE),
    'RB': Z0_REFERENCE * COHN_TABLE_N_2_15['rB'],
    'CA': 1/(W0* ZA_VALUE),
    'LA': ZA_VALUE / W0,
    'RA': Z0_REFERENCE * COHN_TABLE_N_2_15['rA']
}



LUMPED_CIRCUIT_DEFINITION = [
    # Node ALFA
    CircuitComponent("C0", ComponentType.C, Node.ALFA, None, 'C0'),
    CircuitComponent("LIB", ComponentType.L, Node.ALFA, Node.BETA, 'LB'),
    CircuitComponent("LIIB", ComponentType.L, Node.ALFA, Node.THETA, 'LB'),
    # Node BETA
    CircuitComponent("CIB", ComponentType.C, Node.BETA, None, 'CB'),
    CircuitComponent("RB", ComponentType.R, Node.BETA, Node.THETA, 'RB'),
    CircuitComponent("CIA_1", ComponentType.C, Node.BETA, None, 'CA'),
    CircuitComponent("LIA", ComponentType.L, Node.BETA, Node.DELTA, 'LA'),
    # Node DELTA
    CircuitComponent("CIA_2", ComponentType.C, Node.DELTA, None, 'CA'),
    CircuitComponent("RA", ComponentType.R, Node.DELTA, Node.PHI, 'RA'),
    # Node THETA
    CircuitComponent("CIIB", ComponentType.C, Node.THETA, None, 'CB'),
    CircuitComponent("CIIA_1", ComponentType.C, Node.THETA, None, 'CA'),
    CircuitComponent("LIIA", ComponentType.L, Node.THETA, Node.PHI, 'LA'),
    # Node PHI
    CircuitComponent("CIIA_2", ComponentType.C, Node.PHI, None, 'CA'),
]


def evaluate_component_admitance(component: CircuitComponent, frequency_obj: rf.Frequency, part_list):
    component_value = part_list[component.value_key]
    if(component.kind == ComponentType.C):
        impedance = 1/(1j*frequency_obj.w*component_value)
    elif(component.kind == ComponentType.L):
        impedance = 1j*frequency_obj.w*component_value
    else:
        impedance = component_value
    return 1/impedance

def generate_admitance_matrix(frequency_obj:rf.Frequency, part_list):
    admitance_matrix = np.zeros((frequency_obj.w.shape[0], len(Node), len(Node)), dtype=complex)
    for component in LUMPED_CIRCUIT_DEFINITION:
        y_component = evaluate_component_admitance(component, frequency_obj, part_list)
        # Add the component to the matrix
        admitance_matrix[..., component.node_a, component.node_a] += y_component
        if component.node_b is not None:
            admitance_matrix[..., component.node_b, component.node_b] += y_component
            admitance_matrix[..., component.node_a, component.node_b] -= y_component
            admitance_matrix[..., component.node_b, component.node_a] -= y_component
    
    return admitance_matrix

def solve_s_parameter(frequency_obj: rf.Frequency, part_list):
    Y = generate_admitance_matrix(frequency_obj, part_list)
    for node in PORT_NODES:
        Y[..., node, node] += 1.0 / Z0_REFERENCE
    current_sources = np.zeros((len(Node), len(PORT_NODES)), dtype=complex)
    for port, node in enumerate(PORT_NODES):
        current_sources[node, port] = 1.0 /Z0_REFERENCE
    voltages = np.linalg.solve(Y, current_sources)
    return 2.0 * voltages[..., PORT_NODES, :] - np.eye(3) , Y