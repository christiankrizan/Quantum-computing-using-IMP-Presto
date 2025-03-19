#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import itertools
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

def calculate_nco_groups_from_frequency_lists(
    frequency_list,
    baseband_bandwidth_complex,
    nco_guard_band_complex,
    number_of_frequency_groups = 1,
    ):
    ''' Given a list of frequencies, calculate NCO frequencies for
        n groups, that can still target all qubits.
        
        baseband_bandwidth_complex: The available bandwidth.
        Note: 1000e6 = 1000e6 = from -500e6 to +500e6 Hz.
        
        nco_guard_band_complex: The bandwidth around the NCO signal itself
        that no signal may enter. If set to 20 MHz, for instance, then
        no signal may be sent within ±10 MHz of the NCO itself.
    '''
    
    ## TODO!
    if number_of_frequency_groups != 2:
        raise NotImplementedError("Halted! Only 2 group are supported as of yet. TODO!")
    
    # Function to check if a given NCO fits a set of frequencies
    def is_valid_nco(nco, group):
        return all(nco_guard_band_complex/2 <= abs(f - nco) <= baseband_bandwidth_complex/2 for f in group)

    # Try all ways to split into two groups of four signals each
    for group1 in itertools.combinations(frequency_list, int(np.ceil(len(frequency_list)/number_of_frequency_groups))):
        
        ## TODO! More groups pls.
        group2 = tuple(set(frequency_list) - set(group1))
        
        # Generate potential NCOs as the midpoint of each group range
        for nco1 in np.linspace(min(group1) + 0.01, max(group1) - 0.01, 100):
            if is_valid_nco(nco1, group1):
                for nco2 in np.linspace(min(group2) + 0.01, max(group2) - 0.01, 100):
                    if is_valid_nco(nco2, group2):
                        ## TODO! More groups pls.
                        solution = {
                            "Group 1": group1, "NCO 1": nco1,
                            "Group 2": group2, "NCO 2": nco2
                        }
                        break
                if 'solution' in locals():
                    break
        if 'solution' in locals():
            break
    
    # Return the solution found!
    return solution
    