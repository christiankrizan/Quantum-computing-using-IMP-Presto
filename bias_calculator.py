#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import presto
import numpy as np

def change_dc_bias(
    pulse_object,
    current_sequencer_time_T,
    dc_bias_value,
    coupler_dc_port_list
    ):
    ''' Set the DC bias of a Presto AWG lock-in amplifier,
        even though the user may have connected together several
        DC bias ports.
    '''
    
    # Set current sequencer time and pulse object.
    T = current_sequencer_time_T
    
    # Get version number, used for determining the DC bias setting method.
    version = presto.__version__
    older_versions_that_do_not_support_list_inputs_for_dc_bias = [ \
        '2.2.1', '2.3.0', '2.3.1', '2.3.2', '2.4.0', '2.4.1', \
        '2.5.0', '2.5.1', '2.6.0', '2.7.0', '2.7.1', '2.8.0'  \
    ]
    
    # Check whether to set the DC bias value as a list or sequentially.
    if version in older_versions_that_do_not_support_list_inputs_for_dc_bias:
        
        # The user is running an older Presto version, set DC sequentially.
        for _port in coupler_dc_port_list:
            pulse_object.output_dc_bias(T, dc_bias_value, _port)
            T += 1e-6
        
    else:
        # The user is running a more modern Presto version!
        # List inputs into coupler_dc_port_list are supported.
        pulse_object.output_dc_bias(T, dc_bias_value, coupler_dc_port_list)
        T += (1e-6 * len(coupler_dc_port_list))
    
    return T
    
def get_dc_dac_range_integer(
    list_or_integer_of_volts_to_output
    ):
    ''' Get the Presto API-specific integer that sets an appropriate
        DC DAC range, depending on what voltages you are going to use.
    '''
    
    # Sanitise user input to list.
    if not isinstance(list_or_integer_of_volts_to_output, list):
        list_or_integer_of_volts_to_output = \
            [list_or_integer_of_volts_to_output]
    
    # The Presto API auto-sets a bipolar value when the range attains at
    # least 90% of "the next range up" that would be needed for that voltage.
    # I have chosen to do the same.
    ## TODO: But I frankly don't know why yet. //2023-02-16
    
    # Check which integer to return.
    found_range = 4  ## Where 4 = ±10.0 V
    biggest_abs = np.max(np.abs(list_or_integer_of_volts_to_output))
    if biggest_abs < 6.0:
        if biggest_abs < 3.0:
            if np.min(list_or_integer_of_volts_to_output) >= 0.0:
                if biggest_abs < 3.0:
                    found_range = 0  ## Where 0 = 0 V → 3.33 V
                elif biggest_abs < 6.0:
                    found_range = 1  ## Where 1 = 0 V → 6.67 V
            else:
                found_range = 2  ## Where 2 = ±3.33 V;  3.0 = 90% of 3.333...
        else:
            found_range = 3  ## Where 3 = ±6.67 V;  6.0 = 90% of 6.666...
    return found_range
    