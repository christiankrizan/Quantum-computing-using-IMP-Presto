#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_zz_amplitude(
    q1_frequency,
    q2_frequency
    ):
    ''' Get the amplitude of the ZZ interaction for two transmons
        interconnected by a SQUID, in units of Hz. Based on Noguchi2020:
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.062408
        
        In this paper, one of the transmons is a SNAIL, and thus
        their Hamiltonian looks funny. But I think the J_ZZ expression
        is still valid.
    '''
    
    # Let's define the energy level of the |00⟩ state, in Hz.
    energy_of_00 = 0
    
    # Let's define the |10⟩, |01⟩, and |11⟩ states, in units of Hz.
    energy_of_01 = q2_frequency
    energy_of_10 = q1_frequency
    energy_of_11 = q1_frequency + q2_frequency
    
    # Get J_ZZ amplitude, in Hz.
    diff_11_10 = np.abs(energy_of_11 - energy_of_10)
    diff_01_00 = np.abs(energy_of_01 - energy_of_00)
    J_ZZ = np.abs(diff_11_10 - diff_01_00)
    return J_ZZ
    