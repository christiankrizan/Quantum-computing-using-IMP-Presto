#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_maximum_fidelity_for_x_gate(
        qubit_T1,
        qubit_T2_asterisk,
        gate_time
    ):
    ''' Using Abad et al. 2022 (13), find the maximum possible
        fidelity for an X-gate.
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.150504
    '''
    
    # Get decoherence rates.
    gamma_1 = 1 / qubit_T1
    
    # Get the pure dephasing time, then the dephasing rate.
    qubit_T_phi = 1 / ((1 / qubit_T2_asterisk) - (1 / (2*qubit_T1)))
    gamma_phi = 1 / qubit_T_phi
    
    # Split equation into more easily understandable parts.
    zeroth_expansion = 1
    first_expansion  = gate_time * (gamma_1 + gamma_phi) / 3
    big_expression_in_2 = (11/12)*gamma_1**2 + (5/3)*gamma_1*gamma_phi + gamma_phi**2
    second_expansion = (1/8) * gate_time**2 * ( big_expression_in_2 )
    
    # Calculate fidelity!
    fidelity_pauli_x = zeroth_expansion - first_expansion + second_expansion
    
    # Done!
    return fidelity_pauli_x
