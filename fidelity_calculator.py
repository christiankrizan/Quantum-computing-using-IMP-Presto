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
        gate_time,
        verbose = False
    ):
    ''' Using Abad et al. 2022 (13), find the maximum possible
        fidelity for an X-gate.
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.150504
    '''
    
    # Get decoherence rates.
    gamma_1 = 1 / qubit_T1
    if verbose:
        print("Gamma_1 is " + str(gamma_1))
    
    # Get the pure dephasing time, then the dephasing rate.
    qubit_T_phi = 1 / ((1 / qubit_T2_asterisk) - (1 / (2*qubit_T1)))
    gamma_phi = 1 / qubit_T_phi
    if verbose:
        print("Gamma_phi is " + str(gamma_phi))
    
    # Split equation into more easily understandable parts.
    zeroth_expansion = 1
    first_expansion  = gate_time * (gamma_1 + gamma_phi) / 3
    big_expression_in_2 = (11/12)*gamma_1**2 + (5/3)*gamma_1*gamma_phi + gamma_phi**2
    second_expansion = (1/8) * gate_time**2 * ( big_expression_in_2 )
    
    # Calculate fidelity!
    fidelity_pauli_x = zeroth_expansion - first_expansion + second_expansion
    if verbose:
        print("F_{sigma x} is " + str(fidelity_pauli_x))
    
    # Done!
    return fidelity_pauli_x

def get_maximum_fidelity_for_2q_gate(
        qubit1_T1,
        qubit2_T1,
        qubit1_T2_asterisk,
        qubit2_T2_asterisk,
        two_qubit_gate_time,
        verbose = False
    ):
    ''' Tahereh Abad told me in person that one approach to getting
        the two-qubit gate fidelity when the two qubits are of different
        relaxation and dephasing rates, would be to take the average of
        the relaxation and dephasing rates.
        
        Using Abad et al. 2022 (15), find the maximum possible
        fidelity for a two-qubit gate.
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.150504
    '''
    # Get decoherence rates.
    gamma_1 = 1 / qubit1_T1
    gamma_2 = 1 / qubit2_T1
    gamma_avg = (gamma_1 + gamma_2) / 2
    if verbose:
        print("Gamma_{1c} was averaged to be " + str(gamma_avg))
    
    # Get the pure dephasing time, then the dephasing rate.
    qubit1_T_phi = 1 / ((1 / qubit1_T2_asterisk) - (1 / (2*qubit1_T1)))
    qubit2_T_phi = 1 / ((1 / qubit2_T2_asterisk) - (1 / (2*qubit2_T1)))
    gamma1_phi = 1 / qubit1_T_phi
    gamma2_phi = 1 / qubit2_T_phi
    print(gamma1_phi)
    print(gamma2_phi)
    gamma_phi_avg = (gamma1_phi + gamma2_phi) / 2
    if verbose:
        print("Gamma_{phi c} was averaged to be " + str(gamma_phi_avg))
    
    ## Split equation into more easily understandable parts
    # Expansion part:
    zeroth_expansion = 1
    
    # Multiplicand of time part:
    N = 2     # Number of qubits.
    d = 2**N  # Number of dimensions.
    multiplicand = (N * d) / (2 * (d + 1))
    if verbose:
        print("N is " + str(N) + ", so d becomes " + str(d))
    
    # Time part:
    time_portion = two_qubit_gate_time * (gamma_avg + gamma_phi_avg)
    
    # Calculate fidelity!
    fidelity_2q_gate = zeroth_expansion - (multiplicand * time_portion)
    if verbose:
        print("F_{N}^c is "+str(fidelity_2q_gate))
    
    # Done!
    return fidelity_2q_gate