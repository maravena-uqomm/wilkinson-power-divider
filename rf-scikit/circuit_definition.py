from enum import IntEnum
from typing import NamedTuple
import numpy as np
import skrf as rf
from dataclasses import dataclass
from abc import ABC, abstractmethod

DEFAULT_INDUCTOR_Q = 60
DEFAULT_INDUCTOR_CP = 0.15e-12
DEFAULT_CAPACITOR_ESR = 0.15
DEFAULT_CAPACITOR_ESL = 0.6e-9


# Standard value series per part kind: (series in decade 1-10, unit)
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9,
       4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
E96 = list(np.round(10 ** (np.arange(96) / 96), 2))       # check against the standard table
STANDARD_SERIES = {"inductor": (E24, 1e-9), "capacitor": (E24, 1e-12), "resistor": (E96, 1.0)}


Z0_REFERENCE= 50


FREQ_BAND_DEFINITION = {
    'start': 140,
    'end': 190,
    'f0': 165,
    'npoints': 51,
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


class NonIdealComponent():
    def __init__(self, component_value):
        self.component_value = component_value

    @abstractmethod
    def get_impedance(self, omega):
        pass

    def get_admitance(self, omega):
        return 1.0/self.get_impedance(omega)

class NonIdealCapacitor(NonIdealComponent):
    def __init__(self, component_value, esR=DEFAULT_CAPACITOR_ESR, esL=DEFAULT_CAPACITOR_ESL):
        self.esR = esR
        self.esL = esL
        super().__init__(component_value)

    def get_impedance(self, omega):
        z = self.esR + 1.0/(1j*omega*self.component_value) + 1j*omega*self.esL
        return z

class NonIdealInductor(NonIdealComponent):
    def __init__(self, component_value, Q=DEFAULT_INDUCTOR_Q, cp=DEFAULT_INDUCTOR_CP):
        self.Q = Q
        self.cp = cp
        super().__init__(component_value)

    def get_series_rl(self, omega):
        return omega*self.component_value*(1j + 1/self.Q)

    def get_impedance(self, omega):
        z_cp = 1.0/(1j*omega*self.cp)
        z_rl = self.get_series_rl(omega)
        z = z_rl * z_cp / (z_rl + z_cp)
        return z



class CircuitComponent(NamedTuple):
    name: str
    kind: ComponentType
    node_a: int
    node_b: int | None
    value_key: str
    # model: NonIdealComponent | None

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
        # impedance = 1/(1j*frequency_obj.w*component_value)
        impedance = NonIdealCapacitor(component_value).get_impedance(frequency_obj.w)
    elif(component.kind == ComponentType.L):
        # impedance = 1j*frequency_obj.w*component_value
        impedance = NonIdealInductor(component_value).get_impedance(frequency_obj.w)
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



@dataclass(frozen=True)
class Spec:
    """Pass/fail limits, applied to the worst case over the band (all values in dB)."""
    reflection_db: float = -20.0       # S11, S22 and S33 must be at or below this
    isolation_db: float = -20.0        # S23 must be at or below this
    insertion_loss_db: float = 3.5     # worst of -S21 and -S31 must be at or below this
    imbalance_db: float = 0.3          # |S21 - S31| must be at or below this


# @dataclass(frozen=True)
# class PartModel:
#     """Non-ideal behaviour of real parts (ASSUMED values, replace with datasheet data)."""
#     inductor_q: float = 60.0                  # quality factor at CENTER_FREQ_HZ
#     inductor_parallel_c: float = 0.15e-12     # F, winding capacitance across the inductor
#     capacitor_esr: float = 0.15               # ohm, series resistance
#     capacitor_esl: float = 0.6e-9             # H, series inductance

#     @classmethod
#     def ideal(cls) -> "PartModel":
#         return cls(inductor_q=float("inf"), inductor_parallel_c=0.0,
#                    capacitor_esr=0.0, capacitor_esl=0.0)



# def inductor_series_impedance(inductance, omega, model: PartModel):
#     """Inductor with a series loss resistance set by Q (specified at the centre frequency)."""
#     series_resistance = 2 * np.pi * FREQ_BAND_DEFINITION['f0']*1e6 * inductance / model.inductor_q
#     return series_resistance + 1j * omega * inductance