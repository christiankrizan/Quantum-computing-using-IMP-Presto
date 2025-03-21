#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

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
    if verbose:
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

def plot_maximum_fidelity_for_2q_gate(
        qubit1_T1,
        qubit2_T1,
        qubit1_T2_asterisk,
        qubit2_T2_asterisk,
        time_axis,
        verbose = False,
        plot_for_this_many_seconds = 0.0
    ):
    ''' Plot the maximum 2q-gate fidelity as a function of gate time.
    '''
    
    # Let's prepare Y values.
    y_vector = []
    
    # Fill this Y vector.
    for item in time_axis:
        y_vector.append(
            get_maximum_fidelity_for_2q_gate(
                qubit1_T1 = qubit1_T1,
                qubit2_T1 = qubit2_T1,
                qubit1_T2_asterisk = qubit1_T2_asterisk,
                qubit2_T2_asterisk = qubit2_T2_asterisk,
                two_qubit_gate_time = item,
                verbose = verbose
            )
        )
    
    # Plot!
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, y_vector, marker='o', linestyle='-', color="#034da3")
    plt.title('Two-qubit gate maximum fidelity')
    plt.ylabel('Gate fidelity [-]')
    plt.xlabel('Gate time [s]')
    plt.grid(True)
    plt.show()
    
    # If inserting a positive time for which we want to plot for,
    # then plot for that duration of time. If given a negative
    # time, then instead block the plotted display.
    if plot_for_this_many_seconds > 0.0:
        plt.show(block=False)
        plt.pause(plot_for_this_many_seconds)
        plt.close()
    else:
        plt.show(block=True)

def plot_maximum_fidelity_for_1q_and_2q_gates(
        qubit1_T1,
        qubit2_T1,
        qubit1_T2_asterisk,
        qubit2_T2_asterisk,
        time_axis,
        verbose = False,
        plot_for_this_many_seconds = 0.0
    ):
    ''' Plot the maximum 1q-gate and 2q-gate fidelities,
        as a function of gate time.
    '''
    # Let's prepare Y values.
    y_vector_1q_A = []
    y_vector_1q_B = []
    y_vector_2q = []
    
    # Fill these Y vectors.
    for item in time_axis:
        y_vector_1q_A.append(
            get_maximum_fidelity_for_x_gate(
                qubit_T1 = qubit1_T1,
                qubit_T2_asterisk = qubit1_T2_asterisk,
                gate_time = item,
                verbose = verbose
            )
        )
        y_vector_1q_B.append(
            get_maximum_fidelity_for_x_gate(
                qubit_T1 = qubit2_T1,
                qubit_T2_asterisk = qubit2_T2_asterisk,
                gate_time = item,
                verbose = verbose
            )
        )
        y_vector_2q.append(
            get_maximum_fidelity_for_2q_gate(
                qubit1_T1 = qubit1_T1,
                qubit2_T1 = qubit2_T1,
                qubit1_T2_asterisk = qubit1_T2_asterisk,
                qubit2_T2_asterisk = qubit2_T2_asterisk,
                two_qubit_gate_time = item,
                verbose = verbose
            )
        )
    
    ## Adjust the time axis to the nearest
    ## exponent of 3 (milli, mikro, nano, etc.)
    max_val = max(time_axis)
    if max_val >= 1: # No rescaling.
        scale_factor = 1
        unit = ""
    elif max_val >= 1e-3:   # Values in seconds or milliseconds
        scale_factor = 1e3  # Convert seconds to milliseconds
        unit = "m"
    elif max_val >= 1e-6:   # Values in microseconds
        scale_factor = 1e6  # Convert seconds to microseconds
        unit = "µ"
    elif max_val >= 1e-9:   # Values in nanoseconds
        scale_factor = 1e9  # Convert seconds to nanoseconds
        unit = "n"
    elif max_val >= 1e-12:  # Values in picoseconds
        scale_factor = 1e12 # Convert seconds to picoseconds
        unit = "p"
    elif max_val >= 1e-15:  # Values in femtoseconds
        scale_factor = 1e15 # Convert seconds to femtoseconds
        unit = "f"
    elif max_val >= 1e-18:  # Values in attoseconds
        scale_factor = 1e18 # Convert seconds to attoseconds
        unit = "a"
    elif max_val >= 1e-21:  # Values in zeptoseconds
        scale_factor = 1e21 # Convert seconds to zeptoseconds
        unit = "z"
    elif max_val >= 1e-24:  # Values in yoctoseconds
        scale_factor = 1e24 # Convert seconds to yoctoseconds
        unit = "y"
    elif max_val >= 1e-27:  # Values in rontoseconds
        scale_factor = 1e27 # Convert seconds to rontoseconds
        unit = "r"
    elif max_val >= 1e-30:  # Values in quectoseconds
        scale_factor = 1e30 # Convert seconds to quectoseconds
        unit = "q"
    
    # Scale!
    time_axis = [t * scale_factor for t in time_axis]
    
    # Plot!
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, y_vector_1q_A, marker='o', linestyle='-', color="#000000", label = 'X gate on QB1')
    plt.plot(time_axis, y_vector_1q_B, marker='o', linestyle='-', color="#ef1620", label = 'X gate on QB2')
    plt.plot(time_axis, y_vector_2q, marker='o', linestyle='-', color="#034da3", label = '2-qubit gates')
    plt.title('Gate fidelity versus gate time', fontsize=22)
    plt.ylabel('Gate fidelity [-]', fontsize=22)
    plt.xlabel('Gate time ['+unit+'s]', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    
    # If inserting a positive time for which we want to plot for,
    # then plot for that duration of time. If given a negative
    # time, then instead block the plotted display.
    if plot_for_this_many_seconds > 0.0:
        plt.show(block=False)
        plt.pause(plot_for_this_many_seconds)
        plt.close()
    else:
        plt.show(block=True)
    
def calculate_qubit_fidelity(
    frequency_of_01_transition,
    t1_time_of_qubit
    ):
    ''' Calculate the qubit quality factor.
        Q = ω_q · T₁
    '''
    Q = 2 * np.pi * frequency_of_01_transition * t1_time_of_qubit
    return Q
    