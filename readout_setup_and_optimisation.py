#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

from presto import pulsed
from presto.utils import sin2, get_sourcecode
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

import os
import sys
import time
import h5py # Needed for .h5 data feedback.
import shutil
import numpy as np
from numpy import hanning as von_hann
from datetime import datetime
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_timestamp_string, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save, \
    export_processed_data_to_file
from data_discriminator import \
    calculate_area_mean_perimeter_fidelity, \
    update_discriminator_settings_with_value
from time_remaining_printer import show_user_time_remaining


def optimise_integration_window_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
    integation_window_start_min,
    integation_window_start_max,
    num_integration_window_start_steps,
    
    integation_window_stop_min,
    integation_window_stop_max,
    num_integration_window_stop_steps,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    use_log_browser_database = True,
    suppress_log_browser_export_of_suboptimal_data = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        },
    
    my_weight_given_to_area_spanned_by_qubit_states = 0.0,
    my_weight_given_to_mean_distance_between_all_states = 0.0,
    my_weight_given_to_hamiltonian_path_perimeter = 0.0,
    my_weight_given_to_readout_fidelity = 1.0
    ):
    ''' Perform complex domain readout using swept integration window
        start and stop times. This function will generate one complex-plane
        dataset per resonator frequency step, unless the Log Browser output
        is suppressed.
        The final, stored plot, will have axes:
            Integration time start
            Integration time stop
            Score (on Y)
        
        Score is either the area, perimeter, or mean distance between states
        from the readout using integration start and -stop times for that
        pixel.
    '''
    
    ## Input sanitation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    # Declare arrays for the integration window start and stop times.
    integration_window_start_arr = np.linspace(integation_window_start_min, integation_window_start_max, num_integration_window_start_steps)
    integration_window_stop_arr  = np.linspace(integation_window_stop_min,  integation_window_stop_max,  num_integration_window_stop_steps )
    
    # All weighted output complex data "scores" will be stored later.
    list_of_current_complex_datasets = []
    
    # For this type of measurement, all data will always be saved as
    # complex values inside of the
    # get_complex_data_for_readout_optimisation_g_e_f function. But, we do not
    # want to export complex data.
    save_complex_data = False
    
    # Acquire all complex data.
    # For this, it makes sense to get a time estimate.
    average_duration_per_point = 0.0
    num_tick_tocks = 0
    total_dur = 0.0
    for curr_integration_start in integration_window_start_arr:
        for curr_integration_stop in integration_window_stop_arr:
            tick = time.time() # Get a time estimate.
            
            # Reset the dataset
            current_complex_dataset = ''
            
            # Take care of illegal sweep values
            if curr_integration_stop <= curr_integration_start:
                area_spanned = np.array(0.0)
                mean_state_distance = np.array(0.0)
                hamiltonian_path_perimeter = np.array(0.0)
                readout_fidelity = np.array(0.0)
                
                # Print time remaining?
                print_time_remaining = False
            else:
                # We are on the lookout for the device crashing,
                # see TODO in the except-clause.
                try:
                    current_complex_dataset = get_complex_data_for_readout_optimisation_g_e_f(
                        ip_address = ip_address,
                        ext_clk_present = ext_clk_present,
                        
                        readout_stimulus_port = readout_stimulus_port,
                        readout_sampling_port = readout_sampling_port,
                        readout_freq = readout_freq,
                        readout_amp = readout_amp,
                        readout_duration = readout_duration,
                        
                        sampling_duration = sampling_duration,
                        readout_sampling_delay = readout_sampling_delay,
                        repetition_delay = repetition_delay,
                        integration_window_start = curr_integration_start, # Sweep parameter!
                        integration_window_stop  = curr_integration_stop,  # Sweep parameter!
                        
                        control_port = control_port,
                        control_amp_01 = control_amp_01,
                        control_freq_01 = control_freq_01,
                        control_duration_01 = control_duration_01,
                        control_amp_12 = control_amp_12,
                        control_freq_12 = control_freq_12,
                        control_duration_12 = control_duration_12,
                        
                        coupler_dc_port = coupler_dc_port,
                        coupler_dc_bias = coupler_dc_bias,
                        added_delay_for_bias_tee = added_delay_for_bias_tee,
                        
                        num_averages = num_averages,
                        num_shots_per_state = num_shots_per_state,
                        resonator_transmon_pair_id_number = resonator_transmon_pair_id_number,
                        
                        use_log_browser_database = use_log_browser_database,
                        suppress_log_browser_export = suppress_log_browser_export_of_suboptimal_data,
                        axes = axes
                    )
                    
                    # current_complex_dataset will be a char array. Convert to string.
                    current_complex_dataset = "".join(current_complex_dataset)
                    
                    # Analyse the complex dataset, without ruining the stored
                    # discriminator settings.
                    area_spanned, mean_state_distance, hamiltonian_path_perimeter, readout_fidelity = \
                        calculate_area_mean_perimeter_fidelity( \
                            path_to_data = os.path.abspath(current_complex_dataset)
                        )
                
                    # We no longer need this data file, so we should clean
                    # up the hard drive space.
                    attempt = 0
                    max_attempts = 5
                    success = False
                    while (attempt < max_attempts) and (not success):
                        try:
                            os.remove(os.path.abspath(current_complex_dataset))
                            success = True
                        except FileNotFoundError:
                            attempt += 1
                            time.sleep(0.2)
                    if (not success):
                        raise OSError("Error: could not delete data file "+str(os.path.abspath(current_complex_dataset))+" after "+str(max_attempts)+" attempts. Halting.")

                    # Typecasting to numpy.
                    area_spanned = np.array(area_spanned)
                    mean_state_distance = np.array(mean_state_distance)
                    hamiltonian_path_perimeter = np.array(hamiltonian_path_perimeter)
                    readout_fidelity = np.array(readout_fidelity)
                    
                    # Print time remaining?
                    print_time_remaining = True
                
                except ConnectionRefusedError:
                    
                    # If the Presto drops the connection for whatever reason
                    # that has not been figured out yet, return 0.
                    # (2022-04-17: it's an IMP-confirmed bug)
                    area_spanned = np.array(0.0)
                    mean_state_distance = np.array(0.0)
                    hamiltonian_path_perimeter = np.array(0.0)
                    readout_fidelity = np.array(0.0)
                    
                    # Print time remaining?
                    print_time_remaining = False
                    
                    # TODO Error due to a device bug, still confirmed 2022-04-17.
                    print("DEVICE ERROR: ConnectionRefusedError: the Presto device dropped the Ethernet connection and must be restarted manually.")
            
            tock = time.time() # Get a time estimate.
            num_tick_tocks += 1
            total_dur += (tock - tick)
            average_duration_per_point = total_dur / num_tick_tocks
            calc = ((len(integration_window_start_arr)*len(integration_window_stop_arr))-num_tick_tocks)*average_duration_per_point
            if (calc != 0.0) and print_time_remaining:
                # Print "true" time remaining.
                show_user_time_remaining(calc)
            
            # Append the area spanned, mean state distance gotten,
            # the perimeter spanned by this particular dataset, and the
            # readout fidelity. The weights are applied at this stage.
            weighted_area = area_spanned.astype(np.float64) * my_weight_given_to_area_spanned_by_qubit_states
            weighted_mean_distance = mean_state_distance.astype(np.float64) * my_weight_given_to_mean_distance_between_all_states
            weighted_perimeter = hamiltonian_path_perimeter.astype(np.float64) * my_weight_given_to_hamiltonian_path_perimeter
            weighted_readout_fidelity = readout_fidelity.astype(np.float64) * my_weight_given_to_readout_fidelity
            list_of_current_complex_datasets.append([ \
                weighted_area, \
                weighted_mean_distance, \
                weighted_perimeter, \
                weighted_readout_fidelity \
            ])
    
    # The list of current_complex_datasets was swept in 2 axes.
    # Later, we must reshape this list into curr_integration_start rows and
    # curr_integration_stop columns. Every pixel will then contain
    # the [weighted area, weighted_mean_distance, weighted_perimeter] for
    # that pixel. Meaning we will be able to export 3 plots.
    
    # How many plots should be exported?
    # Prepare the variable num_weights_in_function_to_check, it's used later.
    hdf5_logs = []
    num_weights_in_function_to_check = 4
    if my_weight_given_to_area_spanned_by_qubit_states > 0.0:
        hdf5_logs.append('Spanned areas')
        hdf5_logs.append("(FS)²")
    if my_weight_given_to_mean_distance_between_all_states > 0.0:
        hdf5_logs.append('Mean of every distance between states')
        hdf5_logs.append("FS")
    if my_weight_given_to_hamiltonian_path_perimeter > 0.0:
        hdf5_logs.append('Smallest perimeter spanned by states')
        hdf5_logs.append("FS")
    if my_weight_given_to_readout_fidelity > 0.0:
        hdf5_logs.append('Readout fidelity')
        hdf5_logs.append("")
    processed_data = [[]] * (len(hdf5_logs) // 2)
    assert len(hdf5_logs) != 0, \
        "Error: no non-zero data can be exported using the user-provided " +\
        "readout population score weights: " +\
        "(area, mean dist., perimeter, readout fidelity) = (" +\
        str(my_weight_given_to_area_spanned_by_qubit_states)+", "           +\
        str(my_weight_given_to_mean_distance_between_all_states)+", "       +\
        str(my_weight_given_to_hamiltonian_path_perimeter)+", "             +\
        str(my_weight_given_to_readout_fidelity)+")"
    
    # Manufacture the processed_data matrix
    curr_index_in_processed_data = 0
    for pro in range(num_weights_in_function_to_check):
        if  ((pro == 0) and (my_weight_given_to_area_spanned_by_qubit_states > 0.0)) or \
            ((pro == 1) and (my_weight_given_to_mean_distance_between_all_states > 0.0)) or \
            ((pro == 2) and (my_weight_given_to_hamiltonian_path_perimeter > 0.0)) or \
            ((pro == 3) and (my_weight_given_to_readout_fidelity > 0.0)):
            
            # Make the current parameter grid.
            curr_2d_grid = np.zeros([len(integration_window_start_arr),len(integration_window_stop_arr)])
            where_am_i = 0
            for out_loop in range(len(integration_window_start_arr)):
                for in_loop in range(len(integration_window_stop_arr)):
                    curr_2d_grid[out_loop][in_loop] = (list_of_current_complex_datasets[where_am_i])[pro]
                    where_am_i += 1
            
            # Update processed_data.
            processed_data[curr_index_in_processed_data] = curr_2d_grid
            curr_index_in_processed_data += 1
    
    # Data to be stored.
    hdf5_steps = [
        'integration_window_stop_arr', "s",
        'integration_window_start_arr', "s",
    ]
    hdf5_singles = [
        'readout_stimulus_port', "",
        'readout_sampling_port', "",
        'readout_freq', "Hz",
        'readout_amp', "FS",
        'readout_duration', "s",
        
        'sampling_duration', "s",
        'readout_sampling_delay', "s",
        'repetition_delay', "s",
        
        'integation_window_start_min', "s",
        'integation_window_start_max', "s",
        'num_integration_window_start_steps', "",
        
        'integation_window_stop_min', "s",
        'integation_window_stop_max', "s",
        'num_integration_window_stop_steps', "",
        
        'control_port', "",
        'control_amp_01', "FS",
        'control_freq_01', "Hz",
        'control_duration_01', "s",
        'control_amp_12', "FS",
        'control_freq_12', "Hz",
        'control_duration_12', "s",
        
        #'coupler_dc_port', "",
        'coupler_dc_bias', "FS",
        'added_delay_for_bias_tee', "s",
        
        'num_averages', "",
        'num_shots_per_state', "",
        'resonator_transmon_pair_id_number', "",
        
        'my_weight_given_to_area_spanned_by_qubit_states', "",
        'my_weight_given_to_mean_distance_between_all_states', "",
        'my_weight_given_to_hamiltonian_path_perimeter', "",
        'my_weight_given_to_readout_fidelity', "",
    ]
    
    ## NOTE! The hdf5_logs table has been made in a special way for this file.
    
    # Ensure the keyed elements above are valid.
    assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
        "Error: non-even amount of keys and units provided. " + \
        "Someone likely forgot a comma."
    
    # Stylistically rework underscored characters in the axes dict.
    axes = stylise_axes(axes)
    
    # Create step lists
    ext_keys = []
    for ii in range(0,len(hdf5_steps),2):
        ext_keys.append( get_dict_for_step_list(
            step_entry_name   = hdf5_steps[ii],
            step_entry_object = np.array( eval(hdf5_steps[ii]) ),
            step_entry_unit   = hdf5_steps[ii+1],
            axes = axes,
            axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
        ))
    for jj in range(0,len(hdf5_singles),2):
        ext_keys.append( get_dict_for_step_list(
            step_entry_name   = hdf5_singles[jj],
            step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
            step_entry_unit   = hdf5_singles[jj+1],
        ))
    for qq in range(len(axes['y_scaler'])):
        if (axes['y_scaler'])[qq] != 1.0:
            ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
        if (axes['y_offset'])[qq] != 0.0:
            try:
                ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
            except IndexError:
                # The user is likely stepping a multiplexed readout with seperate plot exports.
                if (axes['y_unit'])[qq] != 'default':
                    print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                else:
                    print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
    
    # Create log lists
    log_dict_list = []
    for kk in range(0,len(hdf5_logs),2):
        if (len(hdf5_logs)/2 > 1):
            if not ( ('Probability for state |') in hdf5_logs[kk] ):
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
        log_dict_list.append( get_dict_for_log_list(
            log_entry_name = hdf5_logs[kk],
            unit           = hdf5_logs[kk+1],
            log_is_complex = save_complex_data,
            axes = axes
        ))
    
    # Export the complex data (in a Log Browser compatible format).
    string_arr_to_return = export_processed_data_to_file(
        filepath_of_calling_script = os.path.realpath(__file__),
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        processed_data = processed_data,
        fetched_data_scale = axes['y_scaler'],
        fetched_data_offset = axes['y_offset'],
        
        #time_vector = time_vector,   # Nothing to export here.
        #fetched_data_arr = [],       # Nothing to export here.
        timestamp = get_timestamp_string(),
        append_to_log_name_before_timestamp = 'readout_integration_window',
        append_to_log_name_after_timestamp = '',
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = False,
    )
    
def perform_readout_optimisation_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    readout_freq_start,
    readout_freq_stop,
    num_readout_freq_steps,
    
    use_log_browser_database = True,
    suppress_log_browser_export_of_suboptimal_data = True,
    suppress_log_browser_export_of_final_optimal_data = False,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        },
    
    my_weight_given_to_area_spanned_by_qubit_states = 1.0,
    my_weight_given_to_mean_distance_between_all_states = 0.0,
    my_weight_given_to_hamiltonian_path_perimeter = 0.0
    ):
    ''' Perform complex domain readout optimisation. This function will
        generate one complex-plane dataset per resonator frequency step,
        unless the Log Browser output is suppressed.
        The final, stored plot, will be the "winning" plot that had the
        optimum readout given the user's settings.
    '''
    
    ## Input sanitation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    # Declare resonator frequency stepping array.
    resonator_freq_arr = np.linspace(readout_freq_start, readout_freq_stop, num_readout_freq_steps)
    
    # All output complex data plots will be stored for reference later on.
    list_of_current_complex_datasets = []
    
    # For this type of measurement, all data will always be saved as
    # complex values.
    save_complex_data = True
    
    # Acquire all complex data.
    for curr_ro_freq in resonator_freq_arr:
        current_complex_dataset = get_complex_data_for_readout_optimisation_g_e_f(
            ip_address = ip_address,
            ext_clk_present = ext_clk_present,
            
            readout_stimulus_port = readout_stimulus_port,
            readout_sampling_port = readout_sampling_port,
            readout_freq = curr_ro_freq,
            readout_amp = readout_amp,
            readout_duration = readout_duration,
            
            sampling_duration = sampling_duration,
            readout_sampling_delay = readout_sampling_delay,
            repetition_delay = repetition_delay,
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            
            control_port = control_port,
            control_amp_01 = control_amp_01,
            control_freq_01 = control_freq_01,
            control_duration_01 = control_duration_01,
            control_amp_12 = control_amp_12,
            control_freq_12 = control_freq_12,
            control_duration_12 = control_duration_12,
            
            coupler_dc_port = coupler_dc_port,
            coupler_dc_bias = coupler_dc_bias,
            added_delay_for_bias_tee = added_delay_for_bias_tee,
            
            num_averages = num_averages,
            num_shots_per_state = num_shots_per_state,
            resonator_transmon_pair_id_number = resonator_transmon_pair_id_number,
            
            use_log_browser_database = use_log_browser_database,
            suppress_log_browser_export = suppress_log_browser_export_of_suboptimal_data,
            axes = axes
        )
        
        # current_complex_dataset will be a char array. Convert to string.
        current_complex_dataset = "".join(current_complex_dataset)
        
        # Analyse the complex dataset.
        assert 1 == 0, 'TODO: added readout_fidelity as a return to be unpacked by this subroutine, but the calling function has not been updated.'
        area_spanned, mean_state_distance, hamiltonian_path_perimeter = \
            calculate_area_mean_perimeter_fidelity( \
                path_to_data = os.path.abspath(current_complex_dataset)
            )
        
        # Append what dataset was built
        list_of_current_complex_datasets.append([current_complex_dataset, area_spanned, mean_state_distance, hamiltonian_path_perimeter])
        
    # We now have a dataset, showing resonator frequencies vs. area spanned
    # by the states, the mean distance between states in the complex plane,
    # and the Hamiltonian path perimeter.
    list_of_current_complex_datasets = np.array(list_of_current_complex_datasets)
    
    # Let's find the winning set for all entries.
    biggest_area_idx                = np.argmax( list_of_current_complex_datasets[:,1] )
    biggest_mean_state_distance_idx = np.argmax( list_of_current_complex_datasets[:,2] )
    biggest_perimeter_idx           = np.argmax( list_of_current_complex_datasets[:,3] )
    
    ##if (biggest_area_idx == biggest_mean_state_distance_idx) and (biggest_area_idx == biggest_perimeter_idx):
    ##    print("\nThe most optimal readout is seen in \""+list_of_current_complex_datasets[biggest_area_idx,0]+"\". This readout wins in every category." )
    ##else:
    ##    print("\n")
    ##    print( "\""+list_of_current_complex_datasets[biggest_area_idx,0]+"\" had the biggest spanned area." )
    ##    print( "\""+list_of_current_complex_datasets[biggest_mean_state_distance_idx,0]+"\" had the biggest mean intra-state distance." )
    ##    print( "\""+list_of_current_complex_datasets[biggest_perimeter_idx,0]+"\" had the biggest perimeter." )
    
    # Now applying weights, to figure out the optimal.
    weighted_area = (list_of_current_complex_datasets[biggest_area_idx,1]).astype(np.float64) * my_weight_given_to_area_spanned_by_qubit_states
    weighted_mean_distance = (list_of_current_complex_datasets[biggest_mean_state_distance_idx,2]).astype(np.float64) * my_weight_given_to_mean_distance_between_all_states
    weighted_perimeter = (list_of_current_complex_datasets[biggest_perimeter_idx,3]).astype(np.float64) * my_weight_given_to_hamiltonian_path_perimeter
    biggest_metric = np.max([weighted_area, weighted_mean_distance, weighted_perimeter])
    
    # Decide on the winner
    if biggest_metric == weighted_area:
        optimal_choice_idx = biggest_area_idx
    elif biggest_metric == weighted_mean_distance:
        optimal_choice_idx = biggest_mean_state_distance_idx
    else:
        optimal_choice_idx = biggest_perimeter_idx
    print("\nAfter weighing the metrics, entry " + list_of_current_complex_datasets[optimal_choice_idx,0]+" is hereby crowned as the optimal readout. (Scores: [Area, Inter-state distance, Perimeter] = "+str([weighted_area, weighted_mean_distance, weighted_perimeter])+")")
    
    # Get the optimal readout frequency for this resonator.
    with h5py.File(os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]), 'r') as h5f:
        optimal_readout_freq = h5f.attrs["readout_freq"]
        ##print("The optimal readout frequency is: "+str(optimal_readout_freq)+" Hz.")
    
    # Load the complex data from the winner, and re-store this in a new file.
    with h5py.File(os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]), 'r') as h5f:
        time_vector = h5f["time_vector"][()]
        processed_data = h5f["processed_data"][()]
        fetched_data_arr = h5f["fetched_data_arr"][()]
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = h5f["prepared_qubit_states"][()]
        shot_arr = h5f["shot_arr"][()]
    
    # At this point, we may also update the discriminator settings JSON.
    update_discriminator_settings_with_value(
        path_to_data = os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0])
    )
    calculate_area_mean_perimeter_fidelity(
        path_to_data = os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]),
        update_discriminator_settings_json = True
    )
    
    # Declare arrays and scalars that will be used for the export.
    analysed_areas = (list_of_current_complex_datasets[:,1]).astype(np.float64)
    analysed_means = (list_of_current_complex_datasets[:,2]).astype(np.float64)
    analysed_perimeters = (list_of_current_complex_datasets[:,3]).astype(np.float64)
    weighed_areas = (list_of_current_complex_datasets[:,1]).astype(np.float64) * my_weight_given_to_area_spanned_by_qubit_states
    weighed_means = (list_of_current_complex_datasets[:,2]).astype(np.float64) * my_weight_given_to_mean_distance_between_all_states
    weighed_perimeters = (list_of_current_complex_datasets[:,3]).astype(np.float64) * my_weight_given_to_hamiltonian_path_perimeter
    
    # Data to be stored.
    hdf5_steps = [
        'shot_arr', "",
        'prepared_qubit_states', "",
    ]
    hdf5_singles = [
        'optimal_readout_freq', "Hz",
        
        'readout_stimulus_port', "",
        'readout_sampling_port', "",
        'readout_amp', "FS",
        'readout_duration', "s",
        
        'sampling_duration', "s",
        'readout_sampling_delay', "s",
        'repetition_delay', "s", 
        'integration_window_start', "s",
        'integration_window_stop', "s",
        
        'control_port', "",
        'control_amp_01', "FS",
        'control_freq_01', "Hz",
        'control_duration_01', "s",
        'control_amp_12', "FS",
        'control_freq_12', "Hz",
        'control_duration_12', "s",
        
        #'coupler_dc_port', "",
        'coupler_dc_bias', "FS",
        'added_delay_for_bias_tee', "s",
        
        'num_averages', "",
        'num_shots_per_state', "",
        'resonator_transmon_pair_id_number', "",
        
        'readout_freq_start', "Hz",
        'readout_freq_stop', "Hz",
        'num_readout_freq_steps', "",
    ]
    hdf5_logs = [
        'fetched_data_arr', "FS",
        #'analysed_areas', "(FS)²",  TODO! Add capability of exporting other logs in the data exporter.
        #'analysed_means', "FS",
        #'analysed_perimeters', "FS",
        #'weighed_areas', "(FS)²",
        #'weighed_means', "FS",
        #'weighed_perimeters', "FS",
    ]
    
    # Ensure the keyed elements above are valid.
    assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
        "Error: non-even amount of keys and units provided. " + \
        "Someone likely forgot a comma."
    
    # Stylistically rework underscored characters in the axes dict.
    axes = stylise_axes(axes)
    
    # Create step lists
    ext_keys = []
    for ii in range(0,len(hdf5_steps),2):
        ext_keys.append( get_dict_for_step_list(
            step_entry_name   = hdf5_steps[ii],
            step_entry_object = np.array( eval(hdf5_steps[ii]) ),
            step_entry_unit   = hdf5_steps[ii+1],
            axes = axes,
            axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
        ))
    for jj in range(0,len(hdf5_singles),2):
        ext_keys.append( get_dict_for_step_list(
            step_entry_name   = hdf5_singles[jj],
            step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
            step_entry_unit   = hdf5_singles[jj+1],
        ))
    for qq in range(len(axes['y_scaler'])):
        if (axes['y_scaler'])[qq] != 1.0:
            ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
        if (axes['y_offset'])[qq] != 0.0:
            try:
                ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
            except IndexError:
                # The user is likely stepping a multiplexed readout with seperate plot exports.
                if (axes['y_unit'])[qq] != 'default':
                    print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                else:
                    print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
    
    # Create log lists
    log_dict_list = []
    for kk in range(0,len(hdf5_logs),2):
        if len(hdf5_logs)/2 > 1:
            hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
        log_dict_list.append( get_dict_for_log_list(
            log_entry_name = hdf5_logs[kk],
            unit           = hdf5_logs[kk+1],
            log_is_complex = save_complex_data,
            axes = axes
        ))
    
    # Export the complex data (in a Log Browser compatible format).
    string_arr_to_return = export_processed_data_to_file(
        filepath_of_calling_script = os.path.realpath(__file__),
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        time_vector = time_vector,
        processed_data = processed_data,
        fetched_data_arr = fetched_data_arr,
        fetched_data_scale = axes['y_scaler'],
        fetched_data_offset = axes['y_offset'],
        
        timestamp = get_timestamp_string(),
        append_to_log_name_before_timestamp = 'optimal_result',
        append_to_log_name_after_timestamp = '',
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export_of_final_optimal_data,
    )
    
    return string_arr_to_return
    
    
def get_complex_data_for_readout_optimisation_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Perform readout for whatever frequency, for gauging where in the
        real-imaginary-plane one finds the |g>, |e> and |f> states.
    '''
    
    # This measurement requires complex data. The user is not given a choice.
    save_complex_data = True
    
    ## Input sanitisation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   True,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run      =   False
    ) as pls:
        print("Connected. Setting up...")
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        # Control ports
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Coupler port(s)
        if coupler_dc_port != []:
            pls.hardware.set_dac_current(coupler_dc_port, 40_500)
            pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        
        if (integration_window_stop - integration_window_start) < plo_clk_T:
            integration_window_stop = integration_window_start + plo_clk_T
            print("Warning: an impossible integration window was defined. The window stop was moved to "+str(integration_window_stop)+" s.")
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixers
        high_res  = max( [control_freq_01, control_freq_12] )
        low_res   = min( [control_freq_01, control_freq_12] )
        control_freq_nco = high_res - (high_res - low_res)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            sync      = (coupler_dc_port == []),
        )
        # Coupler port mixer
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                sync      = True,  # Sync here
            )
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 1,
            scales          = control_amp_12,
        )
        # Coupler port amplitude (the bias)
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
            )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port =   readout_stimulus_port,
            group       =   0,
            duration    =   readout_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup pulse "control_pulse_pi_01" and "control_pulse_pi_12" ###
        
        # Setup control pulse envelopes
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_ns_12 = int(round(control_duration_12 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_12 = sin2(control_ns_12)
        control_pulse_pi_12 = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01 = np.abs(control_freq_nco - control_freq_01)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = control_freq_if_01,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12 = np.abs(control_freq_nco - control_freq_12)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = control_freq_if_12,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        
        
        ### Setup pulse "coupler_bias_tone" ###
        if coupler_dc_port != []:
            # Setup the coupler tone bias.
            coupler_bias_tone = [pls.setup_long_drive(
                output_port = _port,
                group       = 0,
                duration    = added_delay_for_bias_tee,
                amplitude   = 1.0,
                amplitude_q = 1.0,
                rise_time   = 0e-9,
                fall_time   = 0e-9
            ) for _port in coupler_dc_port]
        
            # Setup coupler bias tone "carrier"
            pls.setup_freq_lut(
                output_ports = coupler_dc_port,
                group        = 0,
                frequencies  = 0.0,
                phases       = 0.0,
                phases_q     = 0.0,
            )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0 # s
        
        # Charge the bias tee.
        if coupler_dc_port != []:
            pls.reset_phase(T, coupler_dc_port)
            pls.output_pulse(T, coupler_bias_tone)
            T += added_delay_for_bias_tee
            
        # Redefine the coupler DC pulse duration to keep on playing once
        # the bias tee has charged.
        if coupler_dc_port != []:
            for bias_tone in coupler_bias_tone:
                bias_tone.set_total_duration(
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    control_duration_12 + \
                    readout_duration + \
                    repetition_delay \
                )
            
            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
        
        ''' State |0> '''
        
        # Read out, while the qubit is in |0>
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |1> '''
        
        # Move the qubit to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |2> '''
        
        # Move the qubit to |2>:
        # First, move to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Second, move to |2>
        pls.output_pulse(T, control_pulse_pi_12)
        T += control_duration_12
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        # Repeat for some number of shots per state.
        pls.run(
            period       = T,
            repeat_count = num_shots_per_state,
            num_averages = num_averages,
            print_time   = True,
        )
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = [0, 1, 2]
        shot_arr = np.linspace(  \
            1,                   \
            num_shots_per_state, \
            num_shots_per_state  \
        )
        
        # Data to be stored.
        hdf5_steps = [
            'shot_arr', "",
            'prepared_qubit_states', "",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12', "FS",
            'control_freq_12', "Hz",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'num_averages', "",
            'num_shots_per_state', "",
            'resonator_transmon_pair_id_number', "",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Ensure the keyed elements above are valid.
        assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
            "Error: non-even amount of keys and units provided. " + \
            "Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        axes = stylise_axes(axes)
        
        # Create step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_steps[ii],
                step_entry_object = np.array( eval(hdf5_steps[ii]) ),
                step_entry_unit   = hdf5_steps[ii+1],
                axes = axes,
                axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
            ))
        for jj in range(0,len(hdf5_singles),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_singles[jj],
                step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
                step_entry_unit   = hdf5_singles[jj+1],
            ))
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                try:
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
                except IndexError:
                    # The user is likely stepping a multiplexed readout with seperate plot exports.
                    if (axes['y_unit'])[qq] != 'default':
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                    else:
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
        
        # Create log lists
        log_dict_list = []
        for kk in range(0,len(hdf5_logs),2):
            if len(hdf5_logs)/2 > 1:
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append( get_dict_for_log_list(
                log_entry_name = hdf5_logs[kk],
                unit           = hdf5_logs[kk+1],
                log_is_complex = save_complex_data,
                axes = axes
            ))
        
        # Save data!
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(prepared_qubit_states),
            outer_loop_size = num_shots_per_state,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'g_e_f',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            force_matrix_reshape_flip_row_and_column = True,
            suppress_log_browser_export = suppress_log_browser_export,
        )
    
    return string_arr_to_return
    

def get_time_traces_for_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    num_averages,
    
    save_complex_data = True,
    use_log_browser_database = True,
    axes =  {
        "x_name":   'Time trace',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'Demodulated amplitude',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Plot the time trace of |g>, |e> and |f>.
    '''
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   True,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run      =   False
    ) as pls:
        print("Connected. Setting up...")
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        # Control ports
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixer
        high_res  = max( [control_freq_01, control_freq_12] )
        low_res   = min( [control_freq_01, control_freq_12] )
        control_freq_nco = high_res - (high_res - low_res)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            sync      = True,
        )
        
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 1,
            scales          = control_amp_12,
        )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port =   readout_stimulus_port,
            group       =   0,
            duration    =   readout_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        ### Setup pulse "control_pulse_pi_01" and "control_pulse_pi_12" ###
        
        # Setup control pulse envelopes
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_ns_12 = int(round(control_duration_12 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_12 = sin2(control_ns_12)
        control_pulse_pi_12 = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01 = np.abs(control_freq_nco - control_freq_01)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = control_freq_if_01,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12 = np.abs(control_freq_nco - control_freq_12)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = control_freq_if_12,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################

        # Start of sequence
        T = 0.0  # s
        
        # For all three states to get time traces of:
        for ii in range(3):
            if ii > 0:
                # Prepare state |e>
                pls.reset_phase(T, control_port)
                pls.output_pulse(T, control_pulse_pi_01)
                T += control_duration_01
            
            if ii > 1:
                # Prepare state |f>
                pls.output_pulse(T, control_pulse_pi_12)
                T += control_duration_12
        
            # Commence readout at some frequency.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Wait for decay
            T += repetition_delay   
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################

        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   1, # Keep at 1, but set outer loop size = 3.
            num_averages    =   num_averages,
            print_time      =   True,
        )
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Establish whether to include biasing in the exported file name.
        try:
            if num_biases > 1:
                with_or_without_bias_string = "_sweep_bias"
            else:
                with_or_without_bias_string = ""
        except NameError:
            try:
                if coupler_dc_bias > 0.0:
                    with_or_without_bias_string = "_with_bias"
                else:
                    with_or_without_bias_string = ""
            except NameError:
                pass
        
        # Data to be stored.
        hdf5_steps = [
            'time_vector', "s",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s", 
            
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12', "FS",
            'control_freq_12', "Hz",
            'control_duration_12', "s",
            
            'num_averages', "",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Ensure the keyed elements above are valid.
        assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
            "Error: non-even amount of keys and units provided. " + \
            "Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        axes = stylise_axes(axes)
        
        # Create step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_steps[ii],
                step_entry_object = np.array( eval(hdf5_steps[ii]) ),
                step_entry_unit   = hdf5_steps[ii+1],
                axes = axes,
                axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
            ))
        for jj in range(0,len(hdf5_singles),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_singles[jj],
                step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
                step_entry_unit   = hdf5_singles[jj+1],
            ))
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                try:
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
                except IndexError:
                    # The user is likely stepping a multiplexed readout with seperate plot exports.
                    if (axes['y_unit'])[qq] != 'default':
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                    else:
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
        
        # Create log lists
        log_dict_list = []
        for kk in range(0,len(hdf5_logs),2):
            if len(hdf5_logs)/2 > 1:
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append( get_dict_for_log_list(
                log_entry_name = hdf5_logs[kk],
                unit           = hdf5_logs[kk+1],
                log_is_complex = save_complex_data,
                axes = axes
            ))
        
        # Save data!
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = 0.0,
            integration_window_stop = 0.0,
            inner_loop_size = len(time_vector),
            outer_loop_size = 3, # Note! 3 b/c |g>, |e>, and |f>
            
            save_complex_data = save_complex_data,
            append_to_log_name_before_timestamp = 'time_traces_g_e_f',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            data_to_store_consists_of_time_traces_only = True,
        )
    
    return string_arr_to_return
    

def get_wire_to_readout_delay(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    repetition_delay,
    
    num_averages,
    
    save_complex_data = True,
    use_log_browser_database = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Find the delay from sending a pulse to seeing it appearing
        in the scoped readout data.
    '''
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   True,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run      =   False
    ) as pls:
        print("Connected. Setting up...")
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = True,
        )
        
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port =   readout_stimulus_port,
            group       =   0,
            duration    =   readout_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################

        # Start of sequence
        T = 0.0  # s
        
        # Commence readout at some frequency.
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T) # Sampling window. Note that there is no delay here.
        T += readout_duration
        
        # Await a new repetition, after which a new coupler DC bias tone
        # will be added - and a new frequency set for the readout tone.
        T += repetition_delay
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################

        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   1,
            num_averages    =   num_averages,
            print_time      =   True,
        )
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Establish whether to include biasing in the exported file name.
        try:
            if num_biases > 1:
                with_or_without_bias_string = "_sweep_bias"
            else:
                with_or_without_bias_string = ""
        except NameError:
            try:
                if coupler_dc_bias > 0.0:
                    with_or_without_bias_string = "_with_bias"
                else:
                    with_or_without_bias_string = ""
            except NameError:
                pass
        
        # Data to be stored.
        hdf5_steps = [
            'time_vector', "s",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            
            'readout_freq', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'repetition_delay', "s", 
            
            'num_averages', "",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Ensure the keyed elements above are valid.
        assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
            "Error: non-even amount of keys and units provided. " + \
            "Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        axes = stylise_axes(axes)
        
        # Create step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_steps[ii],
                step_entry_object = np.array( eval(hdf5_steps[ii]) ),
                step_entry_unit   = hdf5_steps[ii+1],
                axes = axes,
                axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
            ))
        for jj in range(0,len(hdf5_singles),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_singles[jj],
                step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
                step_entry_unit   = hdf5_singles[jj+1],
            ))
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                try:
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
                except IndexError:
                    # The user is likely stepping a multiplexed readout with seperate plot exports.
                    if (axes['y_unit'])[qq] != 'default':
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                    else:
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
        
        # Create log lists
        log_dict_list = []
        for kk in range(0,len(hdf5_logs),2):
            if len(hdf5_logs)/2 > 1:
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append( get_dict_for_log_list(
                log_entry_name = hdf5_logs[kk],
                unit           = hdf5_logs[kk+1],
                log_is_complex = save_complex_data,
                axes = axes
            ))
        
        # Save data!
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = 0.0,
            integration_window_stop = 0.0,
            inner_loop_size = len(time_vector),
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            append_to_log_name_before_timestamp = 'get_wire_delay',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            data_to_store_consists_of_time_traces_only = True,
        )
    
    return string_arr_to_return
    



# ##################################################### #
'''  The following three defs have been uncommented,  '''
''' since they have been superceded by the first def. '''
# ##################################################### #

"""def get_complex_data_for_readout_optimisation_g_e_f_ro0(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_g_state,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    use_log_browser_database = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Perform readout in |0> for gauging where in the real-imaginary-plane
        one finds the |g>, |e> and |f> states.
    '''
    
    # This measurement requires complex data. The user is not given a choice.
    save_complex_data = True
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   True,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run      =   False
    ) as pls:
        print("Connected. Setting up...")
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        # Control ports
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Coupler port(s)
        if coupler_dc_port != []:
            pls.hardware.set_dac_current(coupler_dc_port, 40_500)
            pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        
        if (integration_window_stop - integration_window_start) < plo_clk_T:
            integration_window_stop = integration_window_start + plo_clk_T
            print("Warning: an impossible integration window was defined. The window stop was moved to "+str(integration_window_stop)+" s.")
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_g_state,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixers
        high_res  = max( [control_freq_01, control_freq_12] )
        low_res   = min( [control_freq_01, control_freq_12] )
        control_freq_nco = high_res - (high_res - low_res)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            sync      = (coupler_dc_port == []),
        )
        # Coupler port mixer
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                sync      = True,  # Sync here
            )
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 1,
            scales          = control_amp_12,
        )
        # Coupler port amplitude (the bias)
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
            )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port =   readout_stimulus_port,
            group       =   0,
            duration    =   readout_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup pulse "control_pulse_pi_01" and "control_pulse_pi_12" ###
        
        # Setup control pulse envelopes
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_ns_12 = int(round(control_duration_12 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_12 = sin2(control_ns_12)
        control_pulse_pi_12 = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01 = np.abs(control_freq_nco - control_freq_01)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = control_freq_if_01,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12 = np.abs(control_freq_nco - control_freq_12)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = control_freq_if_12,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        
        
        ### Setup pulse "coupler_bias_tone" ###
        if coupler_dc_port != []:
            # Setup the coupler tone bias.
            coupler_bias_tone = [pls.setup_long_drive(
                output_port = _port,
                group       = 0,
                duration    = added_delay_for_bias_tee,
                amplitude   = 1.0,
                amplitude_q = 1.0,
                rise_time   = 0e-9,
                fall_time   = 0e-9
            ) for _port in coupler_dc_port]
        
            # Setup coupler bias tone "carrier"
            pls.setup_freq_lut(
                output_ports = coupler_dc_port,
                group        = 0,
                frequencies  = 0.0,
                phases       = 0.0,
                phases_q     = 0.0,
            )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0 # s
        
        # Charge the bias tee.
        if coupler_dc_port != []:
            pls.reset_phase(T, coupler_dc_port)
            pls.output_pulse(T, coupler_bias_tone)
            T += added_delay_for_bias_tee
            
        # Redefine the coupler DC pulse duration to keep on playing once
        # the bias tee has charged.
        if coupler_dc_port != []:
            for bias_tone in coupler_bias_tone:
                bias_tone.set_total_duration(
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    control_duration_12 + \
                    readout_duration + \
                    repetition_delay \
                )
            
            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
        
        ''' State |0>, reading out in |0> '''
        
        # Read out the ground state
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |1>, reading out in |0> '''
        
        # Read out the resonator in |0>, but move the qubit to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |2>, reading out in |0> '''
        
        # Read out the resonator in |0>, but move the qubit to |2>
        # First, move to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Second, move to |2>
        pls.output_pulse(T, control_pulse_pi_12)
        T += control_duration_12
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        # Repeat for some number of shots per state.
        pls.run(
            period       = T,
            repeat_count = num_shots_per_state,
            num_averages = num_averages,
            print_time   = True,
        )
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = [0, 1, 2]
        shot_arr = np.linspace(  \
            1,                   \
            num_shots_per_state, \
            num_shots_per_state  \
        )
        
        # Data to be stored.
        hdf5_steps = [
            'shot_arr', "",
            'prepared_qubit_states', "",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_g_state', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12', "FS",
            'control_freq_12', "Hz",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'num_averages', "",
            'num_shots_per_state', "",
            'resonator_transmon_pair_id_number', "",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Ensure the keyed elements above are valid.
        assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
            "Error: non-even amount of keys and units provided. " + \
            "Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        axes = stylise_axes(axes)
        
        # Create step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_steps[ii],
                step_entry_object = np.array( eval(hdf5_steps[ii]) ),
                step_entry_unit   = hdf5_steps[ii+1],
                axes = axes,
                axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
            ))
        for jj in range(0,len(hdf5_singles),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_singles[jj],
                step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
                step_entry_unit   = hdf5_singles[jj+1],
            ))
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                try:
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
                except IndexError:
                    # The user is likely stepping a multiplexed readout with seperate plot exports.
                    if (axes['y_unit'])[qq] != 'default':
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                    else:
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
        
        # Create log lists
        log_dict_list = []
        for kk in range(0,len(hdf5_logs),2):
            if len(hdf5_logs)/2 > 1:
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append( get_dict_for_log_list(
                log_entry_name = hdf5_logs[kk],
                unit           = hdf5_logs[kk+1],
                log_is_complex = save_complex_data,
                axes = axes
            ))
        
        # Save data!
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(prepared_qubit_states),
            outer_loop_size = num_shots_per_state,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'ro0_g_e_f',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            force_matrix_reshape_flip_row_and_column = True,
        )
    
    return string_arr_to_return
    
"""

"""def get_complex_data_for_readout_optimisation_g_e_f_ro1(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_e_state,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    use_log_browser_database = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Perform readout in |1> for gauging where in the real-imaginary-plane
        one finds the |g>, |e> and |f> states.
    '''
    
    # This measurement requires complex data. The user is not given a choice.
    save_complex_data = True
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   True,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run      =   False
    ) as pls:
        print("Connected. Setting up...")
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        # Control ports
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Coupler port(s)
        if coupler_dc_port != []:
            pls.hardware.set_dac_current(coupler_dc_port, 40_500)
            pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        
        if (integration_window_stop - integration_window_start) < plo_clk_T:
            integration_window_stop = integration_window_start + plo_clk_T
            print("Warning: an impossible integration window was defined. The window stop was moved to "+str(integration_window_stop)+" s.")
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_e_state,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixers
        high_res  = max( [control_freq_01, control_freq_12] )
        low_res   = min( [control_freq_01, control_freq_12] )
        control_freq_nco = high_res - (high_res - low_res)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            sync      = (coupler_dc_port == []),
        )
        # Coupler port mixer
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                sync      = True,  # Sync here
            )
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 1,
            scales          = control_amp_12,
        )
        # Coupler port amplitude (the bias)
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
            )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port =   readout_stimulus_port,
            group       =   0,
            duration    =   readout_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup pulse "control_pulse_pi_01" and "control_pulse_pi_12" ###
        
        # Setup control pulse envelopes
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_ns_12 = int(round(control_duration_12 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_12 = sin2(control_ns_12)
        control_pulse_pi_12 = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01 = np.abs(control_freq_nco - control_freq_01)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = control_freq_if_01,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12 = np.abs(control_freq_nco - control_freq_12)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = control_freq_if_12,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        
        
        ### Setup pulse "coupler_bias_tone" ###
        if coupler_dc_port != []:
            # Setup the coupler tone bias.
            coupler_bias_tone = [pls.setup_long_drive(
                output_port = _port,
                group       = 0,
                duration    = added_delay_for_bias_tee,
                amplitude   = 1.0,
                amplitude_q = 1.0,
                rise_time   = 0e-9,
                fall_time   = 0e-9
            ) for _port in coupler_dc_port]
        
            # Setup coupler bias tone "carrier"
            pls.setup_freq_lut(
                output_ports = coupler_dc_port,
                group        = 0,
                frequencies  = 0.0,
                phases       = 0.0,
                phases_q     = 0.0,
            )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0 # s
        
        # Charge the bias tee.
        if coupler_dc_port != []:
            pls.reset_phase(T, coupler_dc_port)
            pls.output_pulse(T, coupler_bias_tone)
            T += added_delay_for_bias_tee
            
        # Redefine the coupler DC pulse duration to keep on playing once
        # the bias tee has charged.
        if coupler_dc_port != []:
            for bias_tone in coupler_bias_tone:
                bias_tone.set_total_duration(
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    control_duration_12 + \
                    readout_duration + \
                    repetition_delay \
                )
            
            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
        
        ''' State |0>, reading out in |0> '''
        
        # Read out the ground state
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |1>, reading out in |0> '''
        
        # Read out the resonator in |0>, but move the qubit to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |2>, reading out in |0> '''
        
        # Read out the resonator in |0>, but move the qubit to |2>
        # First, move to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Second, move to |2>
        pls.output_pulse(T, control_pulse_pi_12)
        T += control_duration_12
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        # Repeat for some number of shots per state.
        pls.run(
            period       = T,
            repeat_count = num_shots_per_state,
            num_averages = num_averages,
            print_time   = True,
        )
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = [0, 1, 2]
        shot_arr = np.linspace(  \
            1,                   \
            num_shots_per_state, \
            num_shots_per_state  \
        )
        
        # Data to be stored.
        hdf5_steps = [
            'shot_arr', "",
            'prepared_qubit_states', "",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_e_state', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12', "FS",
            'control_freq_12', "Hz",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'num_averages', "",
            'num_shots_per_state', "",
            'resonator_transmon_pair_id_number', "",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Ensure the keyed elements above are valid.
        assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
            "Error: non-even amount of keys and units provided. " + \
            "Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        axes = stylise_axes(axes)
        
        # Create step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_steps[ii],
                step_entry_object = np.array( eval(hdf5_steps[ii]) ),
                step_entry_unit   = hdf5_steps[ii+1],
                axes = axes,
                axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
            ))
        for jj in range(0,len(hdf5_singles),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_singles[jj],
                step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
                step_entry_unit   = hdf5_singles[jj+1],
            ))
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                try:
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
                except IndexError:
                    # The user is likely stepping a multiplexed readout with seperate plot exports.
                    if (axes['y_unit'])[qq] != 'default':
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                    else:
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
        
        # Create log lists
        log_dict_list = []
        for kk in range(0,len(hdf5_logs),2):
            if len(hdf5_logs)/2 > 1:
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append( get_dict_for_log_list(
                log_entry_name = hdf5_logs[kk],
                unit           = hdf5_logs[kk+1],
                log_is_complex = save_complex_data,
                axes = axes
            ))
        
        # Save data!
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(prepared_qubit_states),
            outer_loop_size = num_shots_per_state,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'ro1_g_e_f',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            force_matrix_reshape_flip_row_and_column = True,
        )
    
    return string_arr_to_return
    
"""

"""def get_complex_data_for_readout_optimisation_g_e_f_ro2(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_f_state,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    use_log_browser_database = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Perform readout in |2> for gauging where in the real-imaginary-plane
        one finds the |g>, |e> and |f> states.
    '''
    
    # This measurement requires complex data. The user is not given a choice.
    save_complex_data = True
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   True,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run      =   False
    ) as pls:
        print("Connected. Setting up...")
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        # Control ports
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Coupler port(s)
        if coupler_dc_port != []:
            pls.hardware.set_dac_current(coupler_dc_port, 40_500)
            pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        
        if (integration_window_stop - integration_window_start) < plo_clk_T:
            integration_window_stop = integration_window_start + plo_clk_T
            print("Warning: an impossible integration window was defined. The window stop was moved to "+str(integration_window_stop)+" s.")
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_f_state,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixers
        high_res  = max( [control_freq_01, control_freq_12] )
        low_res   = min( [control_freq_01, control_freq_12] )
        control_freq_nco = high_res - (high_res - low_res)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            sync      = (coupler_dc_port == []),
        )
        # Coupler port mixer
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                sync      = True,  # Sync here
            )
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 1,
            scales          = control_amp_12,
        )
        # Coupler port amplitude (the bias)
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
            )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port =   readout_stimulus_port,
            group       =   0,
            duration    =   readout_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup pulse "control_pulse_pi_01" and "control_pulse_pi_12" ###
        
        # Setup control pulse envelopes
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_ns_12 = int(round(control_duration_12 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_12 = sin2(control_ns_12)
        control_pulse_pi_12 = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01 = np.abs(control_freq_nco - control_freq_01)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = control_freq_if_01,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12 = np.abs(control_freq_nco - control_freq_12)
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = control_freq_if_12,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        
        
        ### Setup pulse "coupler_bias_tone" ###
        if coupler_dc_port != []:
            # Setup the coupler tone bias.
            coupler_bias_tone = [pls.setup_long_drive(
                output_port = _port,
                group       = 0,
                duration    = added_delay_for_bias_tee,
                amplitude   = 1.0,
                amplitude_q = 1.0,
                rise_time   = 0e-9,
                fall_time   = 0e-9
            ) for _port in coupler_dc_port]
        
            # Setup coupler bias tone "carrier"
            pls.setup_freq_lut(
                output_ports = coupler_dc_port,
                group        = 0,
                frequencies  = 0.0,
                phases       = 0.0,
                phases_q     = 0.0,
            )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0 # s
        
        # Charge the bias tee.
        if coupler_dc_port != []:
            pls.reset_phase(T, coupler_dc_port)
            pls.output_pulse(T, coupler_bias_tone)
            T += added_delay_for_bias_tee
            
        # Redefine the coupler DC pulse duration to keep on playing once
        # the bias tee has charged.
        if coupler_dc_port != []:
            for bias_tone in coupler_bias_tone:
                bias_tone.set_total_duration(
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    readout_duration + \
                    repetition_delay + \
                    
                    control_duration_01 + \
                    control_duration_12 + \
                    readout_duration + \
                    repetition_delay \
                )
            
            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
        
        ''' State |0>, reading out in |0> '''
        
        # Read out the ground state
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |1>, reading out in |0> '''
        
        # Read out the resonator in |0>, but move the qubit to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        ''' State |2>, reading out in |0> '''
        
        # Read out the resonator in |0>, but move the qubit to |2>
        # First, move to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Second, move to |2>
        pls.output_pulse(T, control_pulse_pi_12)
        T += control_duration_12
        
        # Commence readout
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Wait for decay
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        # Repeat for some number of shots per state.
        pls.run(
            period       = T,
            repeat_count = num_shots_per_state,
            num_averages = num_averages,
            print_time   = True,
        )
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = [0, 1, 2]
        shot_arr = np.linspace(  \
            1,                   \
            num_shots_per_state, \
            num_shots_per_state  \
        )
        
        # Data to be stored.
        hdf5_steps = [
            'shot_arr', "",
            'prepared_qubit_states', "",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_f_state', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12', "FS",
            'control_freq_12', "Hz",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'num_averages', "",
            'num_shots_per_state', "",
            'resonator_transmon_pair_id_number', "",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Ensure the keyed elements above are valid.
        assert ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs), \
            "Error: non-even amount of keys and units provided. " + \
            "Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        axes = stylise_axes(axes)
        
        # Create step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_steps[ii],
                step_entry_object = np.array( eval(hdf5_steps[ii]) ),
                step_entry_unit   = hdf5_steps[ii+1],
                axes = axes,
                axis_parameter = ('x' if (ii == 0) else 'z' if (ii == 2) else ''),
            ))
        for jj in range(0,len(hdf5_singles),2):
            ext_keys.append( get_dict_for_step_list(
                step_entry_name   = hdf5_singles[jj],
                step_entry_object = np.array( [eval(hdf5_singles[jj])] ),
                step_entry_unit   = hdf5_singles[jj+1],
            ))
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                try:
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
                except IndexError:
                    # The user is likely stepping a multiplexed readout with seperate plot exports.
                    if (axes['y_unit'])[qq] != 'default':
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+str(hdf5_logs[1])+").")
                    else:
                        print("Warning: an IndexError occured when setting the ext_key unit for Y"+str(qq+1)+". Falling back to the first log_list entry's unit ("+(axes['y_unit'])[qq]+").")
                    ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[1], values=(axes['y_offset'])[qq]))
        
        # Create log lists
        log_dict_list = []
        for kk in range(0,len(hdf5_logs),2):
            if len(hdf5_logs)/2 > 1:
                hdf5_logs[kk] += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append( get_dict_for_log_list(
                log_entry_name = hdf5_logs[kk],
                unit           = hdf5_logs[kk+1],
                log_is_complex = save_complex_data,
                axes = axes
            ))
        
        # Save data!
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(prepared_qubit_states),
            outer_loop_size = num_shots_per_state,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'ro2_g_e_f',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            force_matrix_reshape_flip_row_and_column = True,
        )
    
    return string_arr_to_return
"""