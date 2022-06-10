#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_legal_phase( value, available_phases_arr ):
    ''' Takes some array of phases deemed legal,
        and returns a value that is present in this array.
        
        All phases are done in radians.
    '''
    # Trim off 2-pi period from the value, and find where the nearest value is.
    diff_arr = np.absolute( available_phases_arr - (value % (2*np.pi)) )
    index = diff_arr.argmin()
    
    # Return result.
    return available_phases_arr[index]
    
def bandsign( if_value, default_to_lsb = False ):
    ''' Return +pi/2 or -pi/2 depending on the sign of the input value.
        But, never return 0.
    '''
    if (not default_to_lsb):
        return (np.sign(if_value)*np.pi/2) if (if_value != 0.0) else (-np.pi/2)
    else:
        return (np.sign(if_value)*np.pi/2) if (if_value != 0.0) else (+np.pi/2)
    
def add_virtual_z(
    at_time,
    current_phase,
    add_this_phase,
    port,
    group,
    available_phases,
    pulse_object,
    ):
    ''' Append a phase to a port, that has had its IF LUT set to 0 Hz.
        Returns the current phase after operation.
    '''
    current_phase = get_legal_phase((current_phase + add_this_phase), available_phases)
    pulse_object.select_frequency(at_time, np.where(available_phases == current_phase)[0][0], port, group = group)
    
    return current_phase


def reset_phase_counter(
    at_time,
    port,
    group,
    available_phases,
    pulse_object
    ):
    ''' Tries to reset the phase counter to 0.
        Returns the nearest legal value in radians.
    '''
    current_phase = get_legal_phase(0.0, available_phases)
    pulse_object.select_frequency(at_time, np.where(available_phases == current_phase)[0][0], port, group = group)
    
    return current_phase
    
    