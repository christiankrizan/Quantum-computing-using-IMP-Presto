#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_qubit_temperature_estimate(
        qubit_frequency,
        qubit_thermal_population,
    ):
    ''' The average is done using an average of the Boltzmann distribution.
    '''
    
    planck_reduced = 1.054571817e-34
    boltzmann_const = 1.380649e-23
    omega_f = 2*np.pi * qubit_frequency
    
    # Get T from:
    # P_therm = e ^ ( -E / (kB·T) ) , where E = h_bar · omega_f
    temperature = ((planck_reduced * omega_f) / -np.log( qubit_thermal_population ) ) / boltzmann_const
    
    # Done!
    return temperature
