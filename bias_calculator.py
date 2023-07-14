#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import presto
import numpy as np
from time import sleep
from time_calculator import get_timestamp_string

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
        '2.0.0', '2.1.0', '2.2.0', '2.2.1', '2.3.0', \
        '2.3.1', '2.3.2', '2.4.0', '2.4.1', '2.5.0', \
        '2.5.1', '2.6.0', '2.7.0', '2.7.1', '2.8.0'  \
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
    ## Note that numpy arrays will work here as well, since
    ## the np.max and np.abs values below, are compatible
    ## with [ np.array() ] too.
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

def initialise_dc_bias(
    pulse_object,
    static_dc_bias_or_list_to_sweep,
    coupler_dc_port,
    settling_time_of_bias_tee,
    safe_slew_rate = 20e-3, # V / s
    ):
    ''' Initialise the DC bias. In order to avoid sharp transients,
        we'll enforce that the DC biasing is ramped.
    '''
    # Assert that the input is legal.
    assert len(coupler_dc_port) > 0, \
        "Halted! No DC ports were provided, cannot change DC bias."
    
    # Is the intended input a list? In that case, then the value we'd
    # like to ramp to is in fact the first item of the list.
    if  (type(static_dc_bias_or_list_to_sweep) == list) or \
        (type(static_dc_bias_or_list_to_sweep) == np.ndarray):
        # The input seems to be array-like. Grab the first item.
        initial_dc_value_to_set = static_dc_bias_or_list_to_sweep[0]
    elif (type(static_dc_bias_or_list_to_sweep) == float) or \
         (type(static_dc_bias_or_list_to_sweep) == int):
        # The value is compatible as the first DC value to initialise into.
        initial_dc_value_to_set = static_dc_bias_or_list_to_sweep
    else:
        raise TypeError( \
            "Error! Could not understand provided DC value to change " + \
            "into. The provided value was: " + \
            str(static_dc_bias_or_list_to_sweep)+".")
    
    # At this point, we know what is the value that we will ramp to.
    
    # What value are we currently at? And, what is our current range?
    dc_settings = pulse_object.hardware.get_dc_bias( coupler_dc_port[0], True)
    currently_set_dc_bias = dc_settings[0]
    current_range         = dc_settings[1]
    
    # At this point, we can calculate the time required for a
    # ramp's time traversal. This time is used below. TODO: But should it?
    traversal_difference_in_voltage = \
        np.abs(currently_set_dc_bias - initial_dc_value_to_set)
    time_required_for_traversal = \
        traversal_difference_in_voltage / safe_slew_rate
    
    # Should we print a status to the user?
    if time_required_for_traversal > 5.0:
        print(get_timestamp_string(pretty = True) + \
        ": Moving the DC bias from "+str(round(currently_set_dc_bias,3))+\
        " V to "+str(initial_dc_value_to_set)+" V at "+str(safe_slew_rate)+\
        " V/s. The DC bias will be at its intended location in "+\
        str(round(time_required_for_traversal,2))+" seconds.")
    
    '''
    ## To initialise the DAC, we need to set a range that will be compatible
    ## with our upcoming measurement. We end up in a pretty wacky situation.
    ## We might be located at a DC bias point, that is outside of the
    ## target range. We might even have to set a particular range, in order
    ## to be able to sweep to our initial value.
    
    # Algorithm:
    # 1. Does the target_range include the currently held DC value?
    #    If yes:
    #       Then we set the target_range and perform the ramp.
    #    If no:
    #        Go to 2.
    # 2. The target range does not include the currently_held DC value.
    #    But is my current range including the initial target DC value?
    #    If yes:
    #       The initial target and the currently held DC value are in
    #       the same range. We do not have to switch range
    #       just to get to the initial value.
    #       →   Do not switch range, perform the ramp.
    #       →   Then, switch range to the target_range needed to run.
    #    If no:
    #       Go to 3.
    # 3. The currently held DC value is not in the same range as the
    #    final target_range. And, it is in fact not in the range
    #    that the initial value is in, so we can't ramp there.
    #    → Set the range that is needed to go from the current_value
    #      to the initial DC value.
    #    → Ramp to the initial value.
    #    → Switch range to the target_range.
    '''
    
    # Step 0: setting or ramping the DC will depend on the instrument version.
    version = presto.__version__
    older_versions_that_do_not_support_dc_ramping = [ \
        '2.0.0', '2.1.0', '2.2.0', '2.2.1', '2.3.0', \
        '2.3.1', '2.3.2', '2.4.0', '2.4.1', '2.5.0', \
        '2.5.1', '2.6.0', '2.7.0', '2.7.1', '2.8.0'  \
    ]
    
    # Step 1: does the target_range include the currently held DC value?
    #         Append the currently held DC value to the target list,
    #         and see whether that range is different from
    #         the range that you'd get if the currently held DC value
    #         was not in the list.
    if (type(static_dc_bias_or_list_to_sweep) == float) or \
       (type(static_dc_bias_or_list_to_sweep) == int):
        list_of_fictional_sweep = np.array([currently_set_dc_bias, static_dc_bias_or_list_to_sweep])
    else:
        # Perform a "safe" cast to a numpy array.
        # Then, append the initial target DC bias.
        list_of_fictional_sweep = np.array(static_dc_bias_or_list_to_sweep)
        list_of_fictional_sweep = np.append(list_of_fictional_sweep, currently_set_dc_bias) 
    
    # What will be the range that is used in the actual measurement?
    target_range = get_dc_dac_range_integer(static_dc_bias_or_list_to_sweep)
    # Would that have been the same range as a range that covers
    # the actual measurement and the value that you are currently at?
    target_range_plus_ramp = get_dc_dac_range_integer(list_of_fictional_sweep)
    target_range_includes_initial_ramp = (target_range == target_range_plus_ramp)
    if target_range_includes_initial_ramp:
        
        # 1.A  Set the range that will be used in the final measurement.
        # 1.B  Ramp to the target voltage.
        # 1.C  We are done.
        
        ## Is it in fact so that we don't need a range change at all?
        ## Let's see if we can skip that too.
        '''if not (target_range == current_range):'''
        if True: ## TODO Apparently, as of 2023-06-30, you cannot skip this part.
            # The current range is not the target range. Let's switch
            # into the target range.
            ## However, since ramping is not supported for all version,
            ## let's ignore changing the range if the version
            ## requires a normal set_dc_bias command anyway.
            if not (version in older_versions_that_do_not_support_dc_ramping):
                # Ramping is supported, only change range!
                pulse_object.hardware.set_dc_bias( \
                    bias    = currently_set_dc_bias, \
                    port    = coupler_dc_port, \
                    range_i = target_range \
                )
                sleep( len(coupler_dc_port) * 1e-6 )
        
        # At this point, we know that we are in a compatible range,
        # and we may ramp!
        ## In case the Presto version is to old,
        ## then switch range and just set a new DC value.
        ''' TODO: For old versions, instead make a funky for-loop that sets
                  the DC in careful steps, with time delays between steps.'''
        if not (version in older_versions_that_do_not_support_dc_ramping):
            pulse_object.hardware.ramp_dc_bias(
                bias = initial_dc_value_to_set,
                port = coupler_dc_port,
                rate = safe_slew_rate
            )
            sleep( time_required_for_traversal )
        else:
            # Ramping not supported :(
            ''' TODO: See TODO 10 rows above, instead make a loop here. '''
            pulse_object.hardware.set_dc_bias( \
                bias    = initial_dc_value_to_set, \
                port    = coupler_dc_port, \
                range_i = target_range \
            )
            sleep( len(coupler_dc_port) * 1e-6 )
    
    else:
        
        ## At this point, we know that the target range is different
        ## than the range that would have included both the
        ## initial ramping and the target measurement.
        # Step 2: Does the currently set range include the initial bias
        #         ramping? In that case, there is no need to switch
        #         ranges, merely sweep to the initial destination.
        list_of_fictional_sweep = \
            np.array([currently_set_dc_bias, initial_dc_value_to_set])
        range_that_only_includes_the_initial_ramp = \
            get_dc_dac_range_integer(list_of_fictional_sweep)
        currently_set_range_includes_initial_ramp = \
            (range_that_only_includes_the_initial_ramp == current_range)
        if currently_set_range_includes_initial_ramp:
            # The currently set range does indeed include the initial ramp!
            # In this case, go to the initial spot,
            # and then switch to the target range.
            if not (version in older_versions_that_do_not_support_dc_ramping):
                pulse_object.hardware.ramp_dc_bias(
                    bias = initial_dc_value_to_set,
                    port = coupler_dc_port,
                    rate = safe_slew_rate
                )
                sleep( time_required_for_traversal )
            else:
                # Ramping not supported :(
                ''' TODO: See TODO above, instead make a loop-ramp here.
                pulse_object.hardware.set_dc_bias( \
                    bias    = initial_dc_value_to_set, \
                    port    = coupler_dc_port, \
                    range_i = target_range \
                )
                sleep( len(coupler_dc_port) * 1e-6 )'''
                pass
            # Now, once we are at the initial DC value, let's switch into
            # the target range.
            pulse_object.hardware.set_dc_bias( \
                bias    = initial_dc_value_to_set, \
                port    = coupler_dc_port, \
                range_i = target_range \
            )
            sleep( len(coupler_dc_port) * 1e-6 )
        else:
            # At this point, we're toast. We have to switch into a range
            # that supports ramping into the first DC bias point,
            # and then again switch into the target range after ramping.
            if not (version in older_versions_that_do_not_support_dc_ramping):
                # This version supports ramping.
                # 3.A  Set the range required to ramp to the destination.
                # 3.B  Ramp to the target destination.
                # 3.C  Change to the target range.
                # 3.D  We're done!
                pulse_object.hardware.set_dc_bias( \
                    bias    = currently_set_dc_bias, \
                    port    = coupler_dc_port, \
                    range_i = range_that_only_includes_the_initial_ramp \
                )
                sleep( len(coupler_dc_port) * 1e-6 )
                pulse_object.hardware.ramp_dc_bias(
                    bias = initial_dc_value_to_set,
                    port = coupler_dc_port,
                    rate = safe_slew_rate
                )
                sleep( time_required_for_traversal )
            else:
                # Ramping not supported :(
                ''' TODO: See TODO above, instead make a loop-ramp here.
                pulse_object.hardware.set_dc_bias( \
                    bias    = initial_dc_value_to_set, \
                    port    = coupler_dc_port, \
                    range_i = target_range \
                )
                sleep( len(coupler_dc_port) * 1e-6 )
                '''
                pass
            pulse_object.hardware.set_dc_bias( \
                bias    = initial_dc_value_to_set, \
                port    = coupler_dc_port, \
                range_i = target_range \
            )
            sleep( len(coupler_dc_port) * 1e-6 )
    
    # Perform a blanket sleep statement here,
    # to ensure that the bias tee is charged by now.
    sleep( settling_time_of_bias_tee )
    
    # Done initialising the DC bias!
    return

def destroy_dc_bias(
    pulse_object,
    coupler_dc_port,
    settling_time_of_bias_tee,
    safe_slew_rate = 20e-3, # V / s
    static_offset_from_zero = 0.0, # V
    ):
    ''' Reset the currently applied DC bias to zero, using a slew slew rate
        that ramps the DC bias to 0 V.
        
        static_offset_from_zero allows the user to dictate that
        "0 V from the Presto instrument" doesn't really equate to zero flux
        through the SQUID loop. There may be some permanent minor flux.
        Which, this argument allows the user to correct for.
        
        Example: the user runs destroy_dc_bias, which sets the DC bias to 0 V.
        The user detects that there is still a flux left in the SQUID,
        equatable to 14 mV worth of DC bias. The user may now set
        static_offset_from_zero = 14e-3. The "reset to zero" is now
        in fact a reset to -14e-3.
    '''
    
    # To set the DC bias to 0, we may use the initialise DC bias command,
    # but we're "initialising to 0 V" so to speak.
    target_dc_that_is_zero = 0.0 - static_offset_from_zero
    
    initialise_dc_bias(
        pulse_object = pulse_object,
        static_dc_bias_or_list_to_sweep = target_dc_that_is_zero,
        coupler_dc_port = coupler_dc_port,
        settling_time_of_bias_tee = settling_time_of_bias_tee,
        safe_slew_rate = safe_slew_rate,
    )
    
    # Done!
    return