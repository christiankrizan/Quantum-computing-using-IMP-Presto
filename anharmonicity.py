#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

from random import randint
import time
import numpy as np

## Physical constants
h = 6.62607015e-34       # Planck's constant [J/Hz]
h_bar = h / (2 * np.pi)  # Reduced Planck's constant [J/Hz]

def calculate_anharmonicity_from_E_C_and_E_J_fifth_order(
    E_C_Hz,
    E_J_Hz
    ):
    ''' Function to try and hit. Note that both E_C and E_J are in Hz.
    '''
    
    # Adjustment.
    E_C = E_C_Hz
    E_J = E_J_Hz
    
    # Calculation.
    zeta  = np.sqrt(2 * E_C / E_J)
    term1 = -1 * E_C
    term2 = -9 * (E_C * zeta)/16
    term3 = -81 * (E_C * zeta**2)/128
    term4 = -3645 * (E_C * zeta**3)/4096
    term5 = -46899 * (E_C * zeta**4)/32768
    anharmonicity_Hz = (term1 + term2 + term3 + term4 + term5)
    
    # Return!
    return anharmonicity_Hz

def calculate_E_C(
    anharmonicity_Hz,
    E_J_Hz,
    acceptable_frequency_error = 150,
    verbose = True
    ):
    ''' Use (2.19) from Joseph Rahamim's PhD thesis to calculate the
        charging energy based on anharmonicity and the Josephson
        energy E_J. It's based on a fifth order expansion beyond the
        familiar h_bar · omega_01 = sqrt( E_J · E_C ) - E_C
        
        Link to thesis:
        https://ora.ox.ac.uk/objects/uuid:c3311eef-7f31-4cfe-afd7-f3605b73ab36/files/m374e6c95cf17ab67fc26e61092a53f81
    '''
    
    # Get a value to try and target.
    ## This is the left hand side of the equation we are running.
    known_anharmonicity = anharmonicity_Hz ##  * h_bar
    
    done = False
    E_C_Hz = -1 * anharmonicity_Hz * 1.10 # Initial guess. Note the -1.
    while(not done):
        
        # Try!
        calculated_anharmonicity = calculate_anharmonicity_from_E_C_and_E_J_fifth_order(
            E_C_Hz = E_C_Hz,
            E_J_Hz = E_J_Hz
        )
        
        # Find difference.
        ## Positive difference: the target is above your calculated value.
        ## Negative difference: the target is below your calculated value.
        difference = known_anharmonicity - calculated_anharmonicity
        
        # Finished?
        if (np.abs(difference) < np.abs(acceptable_frequency_error)):
            done = True
        else:
            if difference > 0:
                E_C_Hz *= 0.99
            elif difference < 0:
                E_C_Hz *= 1.01
    
    # Report difference?
    if verbose:
        print("Difference |anharmonicity / E_C|: "+str(np.abs(anharmonicity_Hz / E_C_Hz)))
    
    # Finished!
    return E_C_Hz
    
    