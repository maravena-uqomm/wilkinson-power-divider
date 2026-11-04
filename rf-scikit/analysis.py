import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from skrf.circuit import Circuit

from circuit_helper import COHN_TABLE_N_2_15, Z0_REFERENCE, FREQ_BAND_DEFINITION


def reference_wilkinson_cohn():
    # Generates a reference circuit using scikit-rf and returns its scattering matrix
    freq = rf.Frequency(
        start=FREQ_BAND_DEFINITION['start'],
        stop=FREQ_BAND_DEFINITION['end'],
        npoints=FREQ_BAND_DEFINITION['npoints'],
        unit=FREQ_BAND_DEFINITION['unit']
    )
    beta = freq.w/rf.constants.c

    # Branches - B section
    RB = COHN_TABLE_N_2_15['rB']*Z0_REFERENCE
    rb_line = rf.media.DefinedGammaZ0(frequency=freq, z0=RB)
    rb_resistor = rb_line.resistor(RB, name='RB')

    zb = COHN_TABLE_N_2_15['zB']*Z0_REFERENCE
    b_line = rf.media.DefinedGammaZ0(frequency=freq, z0=zb)
    d_b = b_line.theta_2_d(90, deg=True)
    b_branch_i = b_line.line(d_b, unit='m', name='b_branch_i')
    b_branch_ii = b_line.line(d_b, unit='m', name='b_branch_ii')

    # Branches -A section
    RA = COHN_TABLE_N_2_15['rA']*Z0_REFERENCE
    ra_line = rf.media.DefinedGammaZ0(frequency=freq, z0=RA)
    ra_resistor = ra_line.resistor(RA, name='RA')

    za = COHN_TABLE_N_2_15['zA']*Z0_REFERENCE
    a_line = rf.media.DefinedGammaZ0(frequency=freq, z0=za)
    d_a = a_line.theta_2_d(90, deg=True)
    a_branch_i = a_line.line(d_a, unit='m', name='a_branch_i')
    a_branch_ii = a_line.line(d_a, unit='m', name='a_branch_ii')

    # Ports
    port1 = Circuit.Port(freq, name='port1', z0=Z0_REFERENCE)
    port2 = Circuit.Port(freq, name='port2', z0=Z0_REFERENCE)
    port3 = Circuit.Port(freq, name='port3', z0=Z0_REFERENCE)




if __name__ == '__main__':
    print("Starting analysis on Lumped Wilkinson Power Divider/Combiner")
    print("Generating N=2 Wilkinson-Cohn power divider")