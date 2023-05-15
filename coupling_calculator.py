#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_estimate_of_resonator_qubit_coupling_g01(
    qubit_transition_frequency_f_01,
    resonator_frequency_bare_f_r,
    resonator_frequency_at_ground_state,
    verbose = False
    ):
    ''' Use a coarser estimate for finding the resonator-qubit coupling j.
        g^2 / Delta = X
        
        ... where X has no relation whatsoever to the other chis you know of.
    '''
    # Qubit-resonator detuning.
    delta = resonator_frequency_at_ground_state - \
            qubit_transition_frequency_f_01
    if verbose:
        print("Delta was: "+str(delta))
    
    # Bare resonator shift.
    fat_X = resonator_frequency_at_ground_state - resonator_frequency_bare_f_r
    if verbose:
        print("fat_X was: "+str(fat_X))
    
    # Return coarse estimate of coupling constant.
    estimate_g01 = np.sqrt( delta * fat_X )
    if verbose:
        print("g01 was estimated to be: "+str(estimate_g01))
    return estimate_g01

def get_resonator_qubit_coupling_g01(
    qubit_transition_frequency_f_01,
    resonator_frequency_bare_f_r,
    resonator_frequency_at_ground_state,
    resonator_frequency_at_excited_state,
    ):
    ''' partial_dispersive_shift_chi_01:
        f_res_|0⟩ - f_res_|1⟩
        
        resonator_frequency_bare_f_r:
        The frequency of your resonator at "very high" resonator tone powers.
        
        resonator_frequency_at_ground_state (and ..._at_excited_state):
        The frequency of your resonator when the qubit is in |g⟩ (or |e⟩).
        
        Formula:
        (3.10) in Koch 2007, Physical Review A 76, 042319
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319
    '''
    
    raise NotImplementedError("Halted! These equations have not been implemented correctly.")
    
    ### τ = 2·π
    ##tau = 2*np.pi
    
    # Calculate the partial dispersive shift chi_01:
    chi_01 = resonator_frequency_at_excited_state - \
             resonator_frequency_at_ground_state
    
    # Angular frequencies:
    ##omega_01 = tau * qubit_transition_frequency_f_01
    ##omega_r  = tau * resonator_frequency_bare_f_r
    f_01 = qubit_transition_frequency_f_01
    f_r  = resonator_frequency_bare_f_r
    
    # Calculate qubit-to-resonator coupling:
    g_01 = np.sqrt( chi_01 * (f_01 - f_r) )
    
    # Done!
    return g_01