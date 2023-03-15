#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_safe_dac_nco_freq(
    dac_mode,
    dac_fsample,
    port_number,
    frequency_bands_list = [],
    verbose = False,
    ):
    ''' Calculate the NCO frequency (LO) to use, based on input parameters,
        such as which bands must be held free of any Nyquist-Shannon
        reflections.
        
        Legal arguments for dacmode are:
            Direct  = No digital upconversion.
            Mixed02 = Default IQ mixed mode. Valid up to f_S ≤ 7 GSa/s.
            Mixed04 = Valid for f_out ∈ [0, f_S/4] ∪ [3f_S/4, f_S]
            Mixed42 = Valid for f_out ∈ [f_S/4, 3f_S/4]
        
        Legal arguments for dac_fsample are:
            G2  = 2  GSa/s
            G4  = 4  GSa/s
            G6  = 6  GSa/s
            G8  = 8  GSa/s
            G10 = 10 GSa/s
        
        Example of the syntax for safe_frequency_bands_list:
            safe_frequency_bands_list =
                [ [10e6, 30e6], [300e6, 800e6], [4e9, 4.5e9] ],
        
        ... corresponds to "the bands 10 MHz to 30 MHz, 300 MHz to 800 MHz,
            and 4 GHz to 4.5 GHz, are all off-limits to the NCO frequency,
            as well as any reflections.
    '''
    
    raise NotImplementedError("Halted. This function is not yet completed.")
    
    ## TODO This function is very much not completed.
    ##      The NCO is returned almost rather arbitrarily at the moment.
    ##      However, deterministically.
    
    # Print?
    if verbose:
        print("Setting DAC NCO frequency on port "+str(port_number)+" to "+final_nco_freq+" Hz.")
    
    # Return.
    return final_nco_freq
    
    