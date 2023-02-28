#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_iswap_frequency(
        qubit_A_frequency,
        qubit_B_frequency,
    ):
    ''' Calculates the expected frequency of an iSWAP gate.
        Which, is the difference frequency between states |10⟩ and |01⟩
    '''
    state_10 = qubit_A_frequency
    state_01 = qubit_B_frequency
    iswap_f = np.abs(state_01 - state_10)
    return iswap_f

def get_cz20_frequency(
        qubit_A_frequency,
        qubit_B_frequency,
        anharmonicity_A
    ):
    ''' Calculates the expected frequency of a CZ₀₂ gate.
        Which, is the difference frequency between states |11⟩ and |02⟩
    '''
    state_20 = qubit_A_frequency + (qubit_A_frequency + anharmonicity_A)
    state_11 = qubit_A_frequency + qubit_B_frequency
    cz20_f = np.abs(state_11 - state_20)
    return cz20_f

def get_cz02_frequency(
        qubit_A_frequency,
        qubit_B_frequency,
        anharmonicity_B
    ):
    ''' Calculates the expected frequency of a CZ₂₀ gate.
        Which, is the difference frequency between states |11⟩ and |20⟩
    '''
    state_02 = qubit_B_frequency + (qubit_B_frequency + anharmonicity_B)
    state_11 = qubit_A_frequency + qubit_B_frequency
    cz02_f = np.abs(state_02 - state_11)
    return cz02_f
    