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

def legalise_phase_array(
    array_to_legalise,
    available_phases_arr,
    remove_duplicates_and_order_array = True,
    ):
    ''' Takes array of angles in units of radians.
        Then, uses get_legal_phase to get legal values for all entries.
    '''
    # Make all values legal.
    legal_array = np.zeros( len(array_to_legalise) )
    for ii in range(len(array_to_legalise)):
        original_value = array_to_legalise[ii]
        legal_value = get_legal_phase( original_value, available_phases_arr )
        # Is the original value outside of the [0,2π) interval?
        # Then put it back into its original interval, as a legal 2π-multiple.
        ## Edge condition: all negative cases where the original_value
        ## is a multiple of 2π need to be handled separately.
        if (original_value >= (+2*np.pi)) or ((original_value % (2*np.pi)) == 0.0):
            legal_value += int(original_value / (2*np.pi)) * (2*np.pi)
        elif original_value < (0.0):
            legal_value += (int(original_value / (2*np.pi)) - 1) * (2*np.pi)
        # We're done with this legalised value.
        legal_array[ii] = legal_value
    
    # Remove duplicates and order array?
    if remove_duplicates_and_order_array:
        legal_array = np.unique(legal_array)
    
    # Return!
    return legal_array

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
    '''
    #return np.pi - np.arccos(np.cos( 2*np.pi * frequency_of_cosinusoidal * current_time + current_phase ))
    #return (2*np.pi * frequency_of_cosinusoidal * current_time + current_phase) % np.pi
    #return (2*np.pi * frequency_of_cosinusoidal * current_time) % (2*np.pi) + current_phase
    return (2*np.pi * frequency_of_cosinusoidal * current_time + np.pi) % (2*np.pi) + current_phase

def reshape_plot_of_added_phases(
    filepath_to_data_in_plot
    ):
    ''' In plots that show the phase added by some controlled-phase gate,
        like a CZ₂₀ gate, it's very possible that the "added" phase
        between two compared examples, is -0.116 = +6.4 radians.
        Then, the plot would loop around, creating funny discontinuous shapes.
        
        To make a nicer plot, we may scan for discontinuities, and sew together
        the data more nicely into a continuous line.
    '''
    
    # 1. Get file, extract data.
    # 2. Scan for discontinuities.
    # 3. Adjust discontinuities.
    # 4. Export new plot.
    
    ## Get data.
    with h5py.File(os.path.abspath( filepath_to_data_in_plot ), 'r') as h5f:
        extracted_data = h5f["processed_data"][()]
        
        if i_renamed_the_control_phase_arr_to == '':
            try:
                control_phase_arr_values = h5f[ "control_phase_arr" ][()]
            except KeyError:
                if verbose:
                    print("Control phase array not found. But, coupler phase array found.")
                control_phase_arr_values = h5f[ "coupler_phase_arr" ][()]
        else:
            control_phase_arr_values = h5f[i_renamed_the_control_phase_arr_to][()]
    
    
    raise NotImplementedError("Halted! Not finished.")
