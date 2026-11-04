from enum import IntEnum, StrEnum
from typing import NamedTuple


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
    value: int


Z0_REF = 50

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

def getWilkinsonValues(z0_ref = Z0_REF):
    realized_values = {}
    for key in COHN_TABLE_N_2_15.keys():
        realized_values[key] = COHN_TABLE_N_2_15[key]*z0_ref
    return realized_values