#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def cap_at_plus_or_minus_two_pi( value_bigger_or_smaller_than_two_pi ):
    ''' Cap the input to a maximum of +2π rad.
        Or, a minimum of -2π rad.
    '''
    # Check if bigger or smaller than 2 pi.
    if value_bigger_or_smaller_than_two_pi > (2*np.pi):
        print("Warning: a phase value was " + \
            str(value_bigger_or_smaller_than_two_pi) + \
            "larger than +6.283185... rad, and thus capped to +6.283185... rad.")
        value_bigger_or_smaller_than_two_pi = (2*np.pi)
    elif value_bigger_or_smaller_than_two_pi < -(2*np.pi):
        print("Warning: a phase value was " + \
            str(value_bigger_or_smaller_than_two_pi) + \
            "smaller than -6.283185... rad, and thus capped to -6.283185... rad.")
        value_bigger_or_smaller_than_two_pi = -(2*np.pi)
    
    # Return result.
    return value_bigger_or_smaller_than_two_pi

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

def bandsign( if_values, default_to_lsb = False ):
    ''' Return +pi/2 or -pi/2 depending on the sign of the input value.
        But, never return 0.
        
        If fed a numpy array of values, return an array filled
        with the appropriate sideband value for all entries.
    '''
    if isinstance(if_values, np.ndarray):
        sign_list = []
        for item in if_values:
            if (not default_to_lsb):
                sign_list.append( (np.sign(item)*np.pi/2) if (item != 0.0) else (-np.pi/2) )
            else:
                sign_list.append( (np.sign(item)*np.pi/2) if (item != 0.0) else (+np.pi/2) )
        return sign_list
    else:
        if (not default_to_lsb):
            return (np.sign(if_values)*np.pi/2) if (if_values != 0.0) else (-np.pi/2)
        else:
            return (np.sign(if_values)*np.pi/2) if (if_values != 0.0) else (+np.pi/2)

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

def track_phase(
    current_time,
    frequency_of_cosinusoidal,
    current_phase
    ):
    ''' Return ***the phase of*** a cosine wave, which at time zero,
        has the same Y-axis value as a cosine wave
        cos( 2*pi * frequency_of_cosinusoidal * current_time + current_phase)
        
        So:
        cos( 2*pi * f * t + phi_orig ) = cos( 2*pi * f * (t=0) + phi_sought)
        = arccos(cos( 2*pi * f * t + phi_orig )) = arccos(cos( 0 + phi_sought )
        <=> phi_sought = 2*pi * f * t + phi_orig [modulo 2 pi something]
        
        As of writing I actually have no idea why there is a pi offset
        from the value that is updated to. All I know is that if I
        change phase [= track_phase(blabla)] into the value pi - phi_sought
        then the phases match up as they should experimentally.
    '''
    #return np.pi - np.arccos(np.cos( 2*np.pi * frequency_of_cosinusoidal * current_time + current_phase ))
    #return (2*np.pi * frequency_of_cosinusoidal * current_time + current_phase) % np.pi
    return (2*np.pi * frequency_of_cosinusoidal * current_time) % (2*np.pi) + current_phase
