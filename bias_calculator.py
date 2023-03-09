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


def sanitise_dc_bias_arguments(
    coupler_dc_port,
    coupler_bias_min = None,
    coupler_bias_max = None,
    num_biases       = None,
    coupler_dc_bias  = None,
    ):
    ''' Receives user input values for bias arguments,
        and sanitises them accordingly.
        
        The returned variable "with_or_without_bias_string" is
        formatted according to what was fed into this routine as well.
    '''
    
    # Acquire legal values regarding the coupler port settings.
    if not ((type(coupler_dc_port) == np.ndarray) or (type(coupler_dc_port) == list)):
        raise TypeError( \
            "Halted! The input argument coupler_dc_port must be provided "  + \
            "as a (numpy) list. Typecasting was not done for you, since "   + \
            "some user setups combine several ports together galvanically. "+ \
            "Typecasting something like an int to [int] risks damaging "    + \
            "their setups. All items in the coupler_dc_port list will be "  + \
            "treated as ports to be used for DC-biasing a coupler.")
    
    # Assert that the user is either requesting a DC bias sweep,
    # or setting a static DC bias. Not both.
    user_wants_to_sweep      =  (coupler_bias_min != None) or \
                                (coupler_bias_max != None) or \
                                (num_biases       != None)
    user_wants_a_static_bias =  (coupler_dc_bias  != None)
    assert (user_wants_to_sweep != user_wants_a_static_bias), \
        "Error! Could not determine whether the user is requesting a sweep "+\
        "of DC biases, or a single DC bias point. If the argument for "+\
        "coupler_dc_bias is given, then there may not be any argument given "+\
        "related to sweeping, and vice versa."
    
    # Does the user want to sweep?
    if user_wants_to_sweep:
        # The user wants to perform a sweep.
        
        # Is the coupler array empty? Then, supply parameters for "No DC bias."
        if coupler_dc_port == []:
        
            # Set num_biases to 1 (namely 0 V)
            if num_biases != 1:
                num_biases = 1
                print("Note: num_biases was set to 1, since the coupler_port array was empty.")
            
            # Set sweep limits to 0.0 V
            if (coupler_bias_min != 0.0) or (coupler_bias_max != 0.0):
                print("Note: the bias _min and _max were both set to 0, since the coupler_port array was empty.")
                coupler_bias_min = 0.0
                coupler_bias_max = 0.0
        
        else:
            # The coupler array is not empty.
            # We have to check whether the arguments are legal.
            assert type(num_biases) == int, "Error! A non-understandable number of biases was requested. Received: \""+str(num_biases)+"\" of type "+str(type(num_biases))+"."
            assert (not (num_biases < 0)),  "Error! A number of biases to sweep over cannot be negative. Received: \""+str(num_biases)+"\" number of biases."
            try:
                coupler_bias_min = coupler_bias_min * 1.0
                coupler_bias_max = coupler_bias_max * 1.0
            except TypeError:
                raise TypeError("Error! Could not parse bias sweep limits. coupler_bias_min was \""+str(coupler_bias_min)+"\" of type "+str(type(coupler_bias_min))+", while coupler_bias_max was \""+str(coupler_bias_max)+"\" of type "+str(type(coupler_bias_max))+".")
        
        # Format the with_or_without_bias_string to be returned,
        # which supplies information whether there was a DC sweep or not.
        if num_biases > 1:
            with_or_without_bias_string = "_sweep_bias"
        else:
            # Only one bias point is set.
            if (coupler_bias_min != 0.0) or (coupler_dc_bias != 0.0):
                with_or_without_bias_string = "_with_bias"
            else:
                with_or_without_bias_string = ""
    
    else:
        # The user does not want to perform a sweep.
        
        # Is the coupler array empty? Then, supply parameters for "No DC bias."
        if coupler_dc_port == []:
            
            # Set the fixed bias (if any) to 0.0 V
            if (coupler_dc_bias != 0.0):
                print("Note: the fixed coupler bias was set to 0, since the coupler_port array was empty.")
                coupler_dc_bias = 0.0
        
        else:
            # The coupler array is not empty.
            # We have to check whether the arguments are legal.
            try:
                coupler_dc_bias = coupler_dc_bias * 1.0
            except TypeError:
                raise TypeError("Error! Could not parse the DC bias value. coupler_dc_bias was \""+str(coupler_dc_bias)+"\" of type "+str(type(coupler_dc_bias))+".")
        
        # Format the with_or_without_bias_string to be returned,
        # which supplies information whether there was a DC sweep or not.
        if (coupler_dc_bias != 0.0):
            with_or_without_bias_string = "_with_bias"
        else:
            with_or_without_bias_string = ""
    
    # Sweep or not, time to return.
    return coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, with_or_without_bias_string
