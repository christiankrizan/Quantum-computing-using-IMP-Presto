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
from phase_calculator import bandsign
from bias_calculator import \
    sanitise_dc_bias_arguments, \
    get_dc_dac_range_integer, \
    initialise_dc_bias, \
    destroy_dc_bias, \
    change_dc_bias
from repetition_rate_calculator import get_repetition_rate_T
from time_calculator import \
    check_if_integration_window_is_legal, \
    show_user_time_remaining, \
    get_timestamp_string
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save, \
    export_processed_data_to_file
from data_discriminator import \
    calculate_area_mean_perimeter_fidelity, \
    update_discriminator_settings_with_value
from connection_fault_handler import force_system_restart_over_ssh


def get_complex_data_for_readout_optimisation_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_of_static_resonator,
    readout_amp_of_static_resonator,
    readout_freq_of_swept_resonator,
    readout_amp_of_swept_resonator,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_rate,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_freq_nco,
    control_freq_01,
    control_amp_01,
    control_duration_01,
    control_freq_12,
    control_amp_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    reset_dc_to_zero_when_finished = True,
    
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    default_exported_log_file_name = 'default',
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    # This measurement requires complex data.
    save_complex_data = True
    
    ## Input sanitisation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   False,
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
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            initialise_dc_bias(
                pulse_object = pls,
                static_dc_bias_or_list_to_sweep = coupler_dc_bias,
                coupler_dc_port = coupler_dc_port,
                settling_time_of_bias_tee = settling_time_of_bias_tee,
                safe_slew_rate = 20e-3, # V / s
            )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port, multiplexed, calculate an optimal NCO frequency.
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            tune      = True,
            sync      = True,
        )
        
        
        ''' Setup scale LUTs '''
        
        # Readout port amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp_of_static_resonator,
        )
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 1,
            scales          = readout_amp_of_swept_resonator,
        )
        # Control port amplitudes
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
        
        # Setup readout pulse envelopes
        readout_pulse_A = pls.setup_long_drive(
            output_port = readout_stimulus_port,
            group       = 0,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
        readout_pulse_B = pls.setup_long_drive(
            output_port = readout_stimulus_port,
            group       = 1,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
        # Setup readout carriers, considering the multiplexed readout NCO.
        readout_freq_if_A = readout_freq_nco - readout_freq_of_static_resonator
        readout_freq_if_B = readout_freq_nco - readout_freq_of_swept_resonator
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = np.abs(readout_freq_if_A),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if_A),
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = np.abs(readout_freq_if_B),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if_B),
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
        control_freq_if_01 = control_freq_nco - control_freq_01
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01),
        )
        control_freq_if_12 = control_freq_nco - control_freq_12
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = np.abs(control_freq_if_12),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_12),
        )
        
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0 # s
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # Do we have to perform an initial set sequence of the DC bias?
        if coupler_dc_port != []:
            T_begin = T # Get a time reference.
            T = change_dc_bias(pls, T, coupler_dc_bias, coupler_dc_port)
            T += settling_time_of_bias_tee
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
        ''' State |0> '''
        
        # Get a time reference, used for gauging the iteration length.
        T_begin = T
        
        # Read out multiplexed, while the qubit is in |0>
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Get T that aligns with the repetition rate.
        T, repetition_counter = get_repetition_rate_T(
            T_begin, T, repetition_rate, repetition_counter,
        )
        
        ''' State |1> '''
        
        # Re-take a new time reference.
        T_begin = T
        
        # Move the qubit to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Make multiplexed readout, while the qubit is in |1>
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Get T that aligns with the repetition rate.
        T, repetition_counter = get_repetition_rate_T(
            T_begin, T, repetition_rate, repetition_counter,
        )
        
        ''' State |2> '''
        
        # Re-take a new time reference.
        T_begin = T
        
        # First, move to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Second, move to |2>
        pls.output_pulse(T, control_pulse_pi_12)
        T += control_duration_12
        
        # Read out multiplexed, while the qubit is in |2>
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Get T that aligns with the repetition rate.
        T, repetition_counter = get_repetition_rate_T(
            T_begin, T, repetition_rate, repetition_counter,
        )
        
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
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            destroy_dc_bias(
                pulse_object = pls,
                coupler_dc_port = coupler_dc_port,
                settling_time_of_bias_tee = settling_time_of_bias_tee,
                safe_slew_rate = 20e-3, # V / s
                static_offset_from_zero = 0.0, # V
            )
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        print("Raw data downloaded to PC.")
        
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
            
            'readout_freq_nco', "Hz",
            'readout_freq_of_static_resonator', "Hz",
            'readout_amp_of_static_resonator', "FS",
            'readout_freq_of_swept_resonator', "Hz",
            'readout_amp_of_swept_resonator', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_freq_nco', "Hz",
            'control_freq_01', "Hz",
            'control_amp_01', "FS",
            'control_duration_01', "s",
            'control_freq_12', "Hz",
            'control_amp_12', "FS",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'num_averages', "",
            'num_shots_per_state', "",
            'resonator_transmon_pair_id_number', "",
        ]
        hdf5_logs = []
        try:
            if len(states_to_discriminate_between) > 0:
                for statep in states_to_discriminate_between:
                    hdf5_logs.append('Probability for state |'+statep+'⟩')
                    hdf5_logs.append("")
            save_complex_data = False
        except NameError:
            pass # Fine, no state discrimnation.
        if len(hdf5_logs) == 0:
            hdf5_logs = [
                'fetched_data_arr', "FS", # Note that there is a multiplexed readout pulse, TODO: but only one signal is analysed per iteration.
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
        string_arr_to_return.append(save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_B ], # TODO: This script should be made multiplexed, check whether the state discrimination (and data_exporter.py) can handle that.
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(prepared_qubit_states),
            outer_loop_size = num_shots_per_state,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'g_e_f' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            force_matrix_reshape_flip_row_and_column = True,
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            
            save_raw_time_data = True
        ))
    
    return string_arr_to_return

def get_complex_data_for_readout_optimisation_g_e(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_of_static_resonator,
    readout_amp_of_static_resonator,
    readout_freq_of_swept_resonator,
    readout_amp_of_swept_resonator,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_rate,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_freq_nco,
    control_freq_01,
    control_amp_01,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    reset_dc_to_zero_when_finished = True,
    
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    default_exported_log_file_name = 'default',
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
        real-imaginary-plane one finds the |g>, |e> states.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    # This measurement requires complex data.
    save_complex_data = True
    
    ## Input sanitisation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   False,
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
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            initialise_dc_bias(
                pulse_object = pls,
                static_dc_bias_or_list_to_sweep = coupler_dc_bias,
                coupler_dc_port = coupler_dc_port,
                settling_time_of_bias_tee = settling_time_of_bias_tee,
                safe_slew_rate = 20e-3, # V / s
            )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port, multiplexed, calculate an optimal NCO frequency.
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            tune      = True,
            sync      = True,
        )
        
        
        ''' Setup scale LUTs '''
        
        # Readout port amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp_of_static_resonator,
        )
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 1,
            scales          = readout_amp_of_swept_resonator,
        )
        # Control port amplitudes
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelopes
        readout_pulse_A = pls.setup_long_drive(
            output_port = readout_stimulus_port,
            group       = 0,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
        readout_pulse_B = pls.setup_long_drive(
            output_port = readout_stimulus_port,
            group       = 1,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
        # Setup readout carriers, considering the multiplexed readout NCO.
        readout_freq_if_A = readout_freq_nco - readout_freq_of_static_resonator
        readout_freq_if_B = readout_freq_nco - readout_freq_of_swept_resonator
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = np.abs(readout_freq_if_A),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if_A),
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = np.abs(readout_freq_if_B),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if_B),
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
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01 = control_freq_nco - control_freq_01
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01),
        )
        
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0 # s
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # Do we have to perform an initial set sequence of the DC bias?
        if coupler_dc_port != []:
            T_begin = T # Get a time reference.
            T = change_dc_bias(pls, T, coupler_dc_bias, coupler_dc_port)
            T += settling_time_of_bias_tee
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
        ''' State |0> '''
        
        # Get a time reference, used for gauging the iteration length.
        T_begin = T
        
        # Read out multiplexed, while the qubit is in |0>
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Get T that aligns with the repetition rate.
        T, repetition_counter = get_repetition_rate_T(
            T_begin, T, repetition_rate, repetition_counter,
        )
        
        ''' State |1> '''
        
        # Re-take a new time reference.
        T_begin = T
        
        # Move the qubit to |1>
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Make multiplexed readout, while the qubit is in |1>
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
        pls.store(T + readout_sampling_delay) # Sampling window
        T += readout_duration
        
        # Get T that aligns with the repetition rate.
        T, repetition_counter = get_repetition_rate_T(
            T_begin, T, repetition_rate, repetition_counter,
        )
        
        # Get T that aligns with the repetition rate.
        T, repetition_counter = get_repetition_rate_T(
            T_begin, T, repetition_rate, repetition_counter,
        )
        
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
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            destroy_dc_bias(
                pulse_object = pls,
                coupler_dc_port = coupler_dc_port,
                settling_time_of_bias_tee = settling_time_of_bias_tee,
                safe_slew_rate = 20e-3, # V / s
                static_offset_from_zero = 0.0, # V
            )
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        print("Raw data downloaded to PC.")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = [0, 1]
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
            
            'readout_freq_nco', "Hz",
            'readout_freq_of_static_resonator', "Hz",
            'readout_amp_of_static_resonator', "FS",
            'readout_freq_of_swept_resonator', "Hz",
            'readout_amp_of_swept_resonator', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_freq_nco', "Hz",
            'control_freq_01', "Hz",
            'control_amp_01', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'num_averages', "",
            'num_shots_per_state', "",
            'resonator_transmon_pair_id_number', "",
        ]
        hdf5_logs = []
        try:
            if len(states_to_discriminate_between) > 0:
                for statep in states_to_discriminate_between:
                    hdf5_logs.append('Probability for state |'+statep+'⟩')
                    hdf5_logs.append("")
            save_complex_data = False
        except NameError:
            pass # Fine, no state discrimnation.
        if len(hdf5_logs) == 0:
            hdf5_logs = [
                'fetched_data_arr', "FS", # Note that there is a multiplexed readout pulse, TODO: but only one signal is analysed per iteration.
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
        string_arr_to_return.append(save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_B ], # TODO: This script should be made multiplexed, check whether the state discrimination (and data_exporter.py) can handle that.
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(prepared_qubit_states),
            outer_loop_size = num_shots_per_state,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'g_e' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            force_matrix_reshape_flip_row_and_column = True,
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            
            save_raw_time_data = True
        ))
    
    return string_arr_to_return


def optimise_readout_frequency_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_of_static_resonator,
    readout_amp_of_static_resonator,
    readout_freq_of_swept_resonator_start,
    readout_freq_of_swept_resonator_stop,
    readout_amp_of_swept_resonator,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_rate,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_freq_nco,
    control_freq_01,
    control_amp_01,
    control_duration_01,
    control_freq_12,
    control_amp_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    num_readout_freq_steps,
    
    reset_dc_to_zero_when_finished = True,
    force_device_reboot_on_connection_error = False,
    
    my_weight_given_to_area_spanned_by_qubit_states = 0.0,
    my_weight_given_to_mean_distance_between_all_states = 0.0,
    my_weight_given_to_hamiltonian_path_perimeter = 0.0,
    my_weight_given_to_readout_fidelity = 1.0,
    
    use_log_browser_database = True,
    suppress_log_browser_export_of_suboptimal_data = True,
    suppress_log_browser_export_of_final_optimal_data = False,
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
    ''' Perform complex domain readout optimisation. This function will
        generate one complex-plane dataset per resonator frequency step,
        unless the Log Browser output is suppressed.
        
        The final, stored plot, will be the "winning" plot that had the
        optimum readout given the user's settings.
        
        Since the amount of stored data will be enormous, only four files
        will be kept as the measurement runs, corresponding to
        whichever metric was the highest.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    ## Input sanitation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    ## Check that at least some kind of data will be saved.
    if  (my_weight_given_to_area_spanned_by_qubit_states == 0) and \
        (my_weight_given_to_mean_distance_between_all_states == 0) and \
        (my_weight_given_to_hamiltonian_path_perimeter == 0) and \
        (my_weight_given_to_readout_fidelity == 0):
        raise AttributeError("Error: All user-argument weights are set to 0, there is nothing to optimise since everything will be 0. No data will be saved. Halting.")
    
    # Declare resonator frequency stepping array.
    resonator_freq_arr = np.linspace(readout_freq_of_swept_resonator_start, readout_freq_of_swept_resonator_stop, num_readout_freq_steps)
    
    # All output complex data plots will be stored for reference later on.
    list_of_current_complex_datasets = []
    
    # For this type of measurement, all data will always be saved as
    # complex values.
    save_complex_data = True
    
    # Prepare variables for keeping track of what time to print out.
    num_tick_tocks = 0
    total_dur = 0
    
    # And, declare variables to keep track of what files' data to keep.
    currently_best_weighted_area = 0.0
    currently_best_weighted_mean_distance = 0.0
    currently_best_weighted_perimeter = 0.0
    currently_best_weighted_fidelity = 0.0
    name_of_file_with_best_area = ''
    name_of_file_with_best_mean_distance = ''
    name_of_file_with_best_perimeter = ''
    name_of_file_with_best_fidelity = ''
    
    # Acquire all complex data.
    for curr_ro_freq in resonator_freq_arr:
        
        # Get a time estimate for printing "time remaining" to the user.
        tick = time.time()
        
        # Grab data.
        ## Perform this step within a connection handler loop,
        ## to catch crashes.
        success = False
        tries = 0
        while ((not success) and (tries <= 5)):
            tries += 1
            try:
                current_complex_dataset = get_complex_data_for_readout_optimisation_g_e_f(
                    ip_address = ip_address,
                    ext_clk_present = ext_clk_present,
                    
                    readout_stimulus_port = readout_stimulus_port,
                    readout_sampling_port = readout_sampling_port,
                    readout_freq_nco = readout_freq_nco,
                    readout_freq_of_static_resonator = readout_freq_of_static_resonator,
                    readout_amp_of_static_resonator = readout_amp_of_static_resonator,
                    readout_freq_of_swept_resonator = curr_ro_freq, # Note: swept here!
                    readout_amp_of_swept_resonator = readout_amp_of_swept_resonator,
                    readout_duration = readout_duration,
                    
                    sampling_duration = sampling_duration,
                    readout_sampling_delay = readout_sampling_delay,
                    repetition_rate = repetition_rate,
                    integration_window_start = integration_window_start,
                    integration_window_stop = integration_window_stop,
                    
                    control_port = control_port,
                    control_freq_nco = control_freq_nco,
                    control_freq_01 = control_freq_01,
                    control_amp_01 = control_amp_01,
                    control_duration_01 = control_duration_01,
                    control_freq_12 = control_freq_12,
                    control_amp_12 = control_amp_12,
                    control_duration_12 = control_duration_12,
                    
                    coupler_dc_port = coupler_dc_port,
                    coupler_dc_bias = coupler_dc_bias,
                    settling_time_of_bias_tee = settling_time_of_bias_tee,
                    
                    num_averages = num_averages,
                    num_shots_per_state = num_shots_per_state,
                    resonator_transmon_pair_id_number = resonator_transmon_pair_id_number,
                    
                    reset_dc_to_zero_when_finished = reset_dc_to_zero_when_finished,
                    
                    use_log_browser_database = use_log_browser_database,
                    suppress_log_browser_export = suppress_log_browser_export_of_suboptimal_data,
                    axes = axes
                )
                
                success = True # Done
            except ConnectionRefusedError:
                if force_device_reboot_on_connection_error:
                    force_system_restart_over_ssh("129.16.115.184")
        assert success, "Halted! Unrecoverable crash detected."
        
        # current_complex_dataset will be a char array. Convert to string.
        current_complex_dataset = "".join(current_complex_dataset)
        
        # Analyse the complex dataset.
        area_spanned, \
        mean_state_distance, \
        hamiltonian_path_perimeter, \
        readout_fidelity, \
        confusion_matrix = \
            calculate_area_mean_perimeter_fidelity( \
                path_to_data = os.path.abspath(current_complex_dataset)
            )
        
        # Append what dataset was built
        list_of_current_complex_datasets.append([current_complex_dataset, area_spanned, mean_state_distance, hamiltonian_path_perimeter, readout_fidelity])
        
        # Figure out whether the data is save-worthy.
        # If the data fails all checks, destroy the current data.
        current_data_is_save_worthy = False
        if (area_spanned*my_weight_given_to_area_spanned_by_qubit_states > currently_best_weighted_area) and (my_weight_given_to_area_spanned_by_qubit_states > 0):
            # This file is better. Save it, delete the old one.
            current_data_is_save_worthy = True
            
            # Delete!
            if name_of_file_with_best_area != '':
                attempt = 0
                max_attempts = 5
                success = False
                while (attempt < max_attempts) and (not success):
                    try:
                        os.remove(os.path.abspath(name_of_file_with_best_area))
                        success = True
                    except FileNotFoundError:
                        attempt += 1
                        time.sleep(0.2)
                if (not success):
                    raise OSError("Error: could not delete data file "+str(os.path.abspath(name_of_file_with_best_area))+" after "+str(max_attempts)+" attempts. Halting.")
            
            # Update status quo
            currently_best_weighted_area = area_spanned*my_weight_given_to_area_spanned_by_qubit_states
            name_of_file_with_best_area = current_complex_dataset
            
        if (mean_state_distance*my_weight_given_to_mean_distance_between_all_states > currently_best_weighted_mean_distance) and (my_weight_given_to_mean_distance_between_all_states > 0):
            # This file is better. Save it, delete the old one.
            current_data_is_save_worthy = True
            
            # Delete!
            if name_of_file_with_best_mean_distance != '':
                attempt = 0
                max_attempts = 5
                success = False
                while (attempt < max_attempts) and (not success):
                    try:
                        os.remove(os.path.abspath(name_of_file_with_best_mean_distance))
                        success = True
                    except FileNotFoundError:
                        attempt += 1
                        time.sleep(0.2)
                if (not success):
                    raise OSError("Error: could not delete data file "+str(os.path.abspath(name_of_file_with_best_mean_distance))+" after "+str(max_attempts)+" attempts. Halting.")
            
            # Update status quo
            currently_best_weighted_mean_distance = mean_state_distance*my_weight_given_to_mean_distance_between_all_states
            name_of_file_with_best_mean_distance = current_complex_dataset
            
        if (hamiltonian_path_perimeter*my_weight_given_to_hamiltonian_path_perimeter > currently_best_weighted_perimeter) and (my_weight_given_to_hamiltonian_path_perimeter > 0):
            # This file is better. Save it, delete the old one.
            current_data_is_save_worthy = True
            
            # Delete!
            if name_of_file_with_best_perimeter != '':
                attempt = 0
                max_attempts = 5
                success = False
                while (attempt < max_attempts) and (not success):
                    try:
                        os.remove(os.path.abspath(name_of_file_with_best_perimeter))
                        success = True
                    except FileNotFoundError:
                        attempt += 1
                        time.sleep(0.2)
                if (not success):
                    raise OSError("Error: could not delete data file "+str(os.path.abspath(name_of_file_with_best_perimeter))+" after "+str(max_attempts)+" attempts. Halting.")
            
            # Update status quo
            currently_best_weighted_perimeter = hamiltonian_path_perimeter*my_weight_given_to_hamiltonian_path_perimeter
            name_of_file_with_best_perimeter = current_complex_dataset
            
        if (readout_fidelity*my_weight_given_to_readout_fidelity > currently_best_weighted_fidelity) and (my_weight_given_to_readout_fidelity > 0):
            # This file is better. Save it, delete the old one.
            current_data_is_save_worthy = True
            
            # Delete!
            if name_of_file_with_best_fidelity != '':
                attempt = 0
                max_attempts = 5
                success = False
                while (attempt < max_attempts) and (not success):
                    try:
                        os.remove(os.path.abspath(name_of_file_with_best_fidelity))
                        success = True
                    except FileNotFoundError:
                        attempt += 1
                        time.sleep(0.2)
                if (not success):
                    raise OSError("Error: could not delete data file "+str(os.path.abspath(name_of_file_with_best_fidelity))+" after "+str(max_attempts)+" attempts. Halting.")
            
            # Update status quo
            currently_best_weighted_fidelity = readout_fidelity*my_weight_given_to_readout_fidelity
            name_of_file_with_best_fidelity = current_complex_dataset
            
        # The acquired data was not better at all. Discard it.
        if (not current_data_is_save_worthy):
            # The recent data is irrelevant, kill it.
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
        
        tock = time.time() # Get a time estimate.
        num_tick_tocks += 1
        total_dur += (tock - tick)
        average_duration_per_point = total_dur / num_tick_tocks
        calc = (len(resonator_freq_arr)-num_tick_tocks)*average_duration_per_point
        if (calc != 0.0):
            # Print "true" time remaining.
            show_user_time_remaining(calc)
    
    # We now have a dataset, showing resonator frequencies vs. area spanned
    # by the states, the mean distance between states in the complex plane,
    # and the Hamiltonian path perimeter.
    list_of_current_complex_datasets = np.array(list_of_current_complex_datasets)
    
    # Let's find the winning set for all entries.
    biggest_area_idx                = np.argmax( list_of_current_complex_datasets[:,1] )
    biggest_mean_state_distance_idx = np.argmax( list_of_current_complex_datasets[:,2] )
    biggest_perimeter_idx           = np.argmax( list_of_current_complex_datasets[:,3] )
    biggest_fidelity_idx            = np.argmax( list_of_current_complex_datasets[:,4] )
    
    ##if (biggest_area_idx == biggest_mean_state_distance_idx) and (biggest_area_idx == biggest_perimeter_idx):
    ##    print("\nThe most optimal readout is seen in \""+list_of_current_complex_datasets[biggest_area_idx,0]+"\". This readout wins in every category." )
    ##else:
    ##    print("\n")
    ##    print( "\""+list_of_current_complex_datasets[biggest_area_idx,0]+"\" had the biggest spanned area." )
    ##    print( "\""+list_of_current_complex_datasets[biggest_mean_state_distance_idx,0]+"\" had the biggest mean intra-state distance." )
    ##    print( "\""+list_of_current_complex_datasets[biggest_perimeter_idx,0]+"\" had the biggest perimeter." )
    ##    print( "\""+list_of_current_complex_datasets[biggest_fidelity_idx,0]+"\" had the biggest readout fidelity." )
    
    # Now applying weights, to figure out the optimal.
    weighted_area = (list_of_current_complex_datasets[biggest_area_idx,1]).astype(np.float64) * my_weight_given_to_area_spanned_by_qubit_states
    weighted_mean_distance = (list_of_current_complex_datasets[biggest_mean_state_distance_idx,2]).astype(np.float64) * my_weight_given_to_mean_distance_between_all_states
    weighted_perimeter = (list_of_current_complex_datasets[biggest_perimeter_idx,3]).astype(np.float64) * my_weight_given_to_hamiltonian_path_perimeter
    weighted_readout_fidelity = (list_of_current_complex_datasets[biggest_fidelity_idx,4]).astype(np.float64) * my_weight_given_to_readout_fidelity
    biggest_metric = np.max([weighted_area, weighted_mean_distance, weighted_perimeter, weighted_readout_fidelity])
    
    # Decide on the winner
    if biggest_metric == weighted_area:
        optimal_choice_idx = biggest_area_idx
    elif biggest_metric == weighted_mean_distance:
        optimal_choice_idx = biggest_mean_state_distance_idx
    elif biggest_metric == weighted_perimeter:
        optimal_choice_idx = biggest_perimeter_idx
    else:
        optimal_choice_idx = biggest_fidelity_idx
    print("\nAfter weighing the metrics, entry " + list_of_current_complex_datasets[optimal_choice_idx,0]+" is hereby crowned as the optimal readout. (Scores: [Area, Inter-state distance, Perimeter, Readout assignment fidelity] = "+str([weighted_area, weighted_mean_distance, weighted_perimeter, weighted_readout_fidelity])+")")
    
    # We know the winner, get its confusion matrix.
    ## Perhaps a TODO: the confusion matrix was acquired several times before.
    ## An optimisation would be to somehow figure out a way where
    ## This additional call to calculate_area_mean_perimeter_fidelity
    ## is removed.
    final_confusion_matrix = ( \
        calculate_area_mean_perimeter_fidelity( \
            list_of_current_complex_datasets[optimal_choice_idx,0]
        ))[4]
    
    # Get the optimal readout frequency for this resonator.
    with h5py.File(os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]), 'r') as h5f:
        optimal_readout_freq = h5f.attrs["readout_freq_of_swept_resonator"] ## TODO! Note that the get_complex_data_for_readout_optimisation_g_e_f is not generating a file with the full multiplexed readout. Only readout_freq_of_swept_resonator is swept in the measurement. This "semi-multiplexed" fact is a TODO.
        '''optimal_readout_freq = h5f.attrs["readout_freq"]''' # Deprecated, mixing needed.
        ##print("The optimal readout frequency is: "+str(optimal_readout_freq)+" Hz.")
    
    # Load the complex data from the winner, and re-store this in a new file.
    with h5py.File(os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]), 'r') as h5f:
        time_vector = h5f["time_vector"][()]
        processed_data = h5f["processed_data"][()]
        fetched_data_arr = h5f["fetched_data_arr"][()]
        
        ## Create a hacky-like array structure for storage's sake.
        prepared_qubit_states = h5f["prepared_qubit_states"][()]
        try:
            shot_arr = h5f["shot_arr"][()]
        except KeyError:
            # The shot arr may either just be a single entry long, in which
            # the Data exporter likely stored the value as an attribute.
            # Or, the user could have renamed the vector via argument.
            # In either case, we may generate the vector here.
            num_shots_per_state = (h5f.attrs["num_shots_per_state"])[0]
            shot_arr = np.linspace(  \
                1,                   \
                num_shots_per_state, \
                num_shots_per_state  \
            )
    
    # At this point, we may also update the discriminator settings JSON.
    update_discriminator_settings_with_value(
        path_to_data = os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0])
    )
    area_spanned, mean_state_distance, hamiltonian_path_perimeter, readout_fidelity, confusion_matrix = calculate_area_mean_perimeter_fidelity(
        path_to_data = os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]),
        update_discriminator_settings_json = True
    )
    
    # Declare arrays and scalars that will be used for the export.
    analysed_areas = (list_of_current_complex_datasets[:,1]).astype(np.float64)
    analysed_means = (list_of_current_complex_datasets[:,2]).astype(np.float64)
    analysed_perimeters = (list_of_current_complex_datasets[:,3]).astype(np.float64)
    analysed_fidelities = (list_of_current_complex_datasets[:,4]).astype(np.float64)
    weighed_areas = (list_of_current_complex_datasets[:,1]).astype(np.float64) * my_weight_given_to_area_spanned_by_qubit_states
    weighed_means = (list_of_current_complex_datasets[:,2]).astype(np.float64) * my_weight_given_to_mean_distance_between_all_states
    weighed_perimeters = (list_of_current_complex_datasets[:,3]).astype(np.float64) * my_weight_given_to_hamiltonian_path_perimeter
    weighed_fidelities = (list_of_current_complex_datasets[:,4]).astype(np.float64) * my_weight_given_to_readout_fidelity
    
    # We are done with the winning file, we may now delete it.
    # The acquired data was not better at all. Discard it.
    attempt = 0
    max_attempts = 5
    success = False
    while (attempt < max_attempts) and (not success):
        try:
            os.remove(os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]))
            success = True
        except FileNotFoundError:
            attempt += 1
            time.sleep(0.2)
    if (not success):
        raise OSError("Error: could not delete data file "+str(os.path.abspath(list_of_current_complex_datasets[optimal_choice_idx,0]))+" after "+str(max_attempts)+" attempts. Halting.")
    
    # Confusion matrix entries. Keep in mind that we know that states
    # |g>, |e> and |f> are present due to the specific measurement that ran.
    prob_meas0_prep0 = final_confusion_matrix[0][0]
    prob_meas1_prep0 = final_confusion_matrix[0][1]
    prob_meas2_prep0 = final_confusion_matrix[0][2]
    prob_meas0_prep1 = final_confusion_matrix[1][0]
    prob_meas1_prep1 = final_confusion_matrix[1][1]
    prob_meas2_prep1 = final_confusion_matrix[1][2]
    prob_meas0_prep2 = final_confusion_matrix[2][0]
    prob_meas1_prep2 = final_confusion_matrix[2][1]
    prob_meas2_prep2 = final_confusion_matrix[2][2]
    
    # Data to be stored.
    hdf5_steps = [
        'shot_arr', "",
        'prepared_qubit_states', "",
    ]
    hdf5_singles = [
        'optimal_readout_freq', "Hz",
        'area_spanned', "(FS)²",
        'mean_state_distance', "FS",
        'hamiltonian_path_perimeter', "FS",
        'readout_fidelity', "",
        
        'readout_stimulus_port', "",
        'readout_sampling_port', "",
        'readout_freq_nco', "Hz",
        'readout_freq_of_static_resonator', "Hz",
        'readout_amp_of_static_resonator', "FS",
        'readout_freq_of_swept_resonator_start', "Hz",
        'readout_freq_of_swept_resonator_stop', "Hz",
        'readout_amp_of_swept_resonator', "FS",
        'readout_duration', "s",
        
        'sampling_duration', "s",
        'readout_sampling_delay', "s",
        'repetition_rate', "s", 
        'integration_window_start', "s",
        'integration_window_stop', "s",
        
        'control_port', "",
        'control_freq_nco', "Hz",
        'control_amp_01', "FS",
        'control_freq_01', "Hz",
        'control_duration_01', "s",
        'control_amp_12', "FS",
        'control_freq_12', "Hz",
        'control_duration_12', "s",
        
        #'coupler_dc_port', "",
        'coupler_dc_bias', "V",
        'settling_time_of_bias_tee', "s",
        
        'num_averages', "",
        'num_shots_per_state', "",
        'resonator_transmon_pair_id_number', "",
        
        'num_readout_freq_steps', "",
        
        'my_weight_given_to_area_spanned_by_qubit_states', "",
        'my_weight_given_to_mean_distance_between_all_states', "",
        'my_weight_given_to_hamiltonian_path_perimeter', "",
        'my_weight_given_to_readout_fidelity', "",
        
        'prob_meas0_prep0', "",
        'prob_meas1_prep0', "",
        'prob_meas2_prep0', "",
        'prob_meas0_prep1', "",
        'prob_meas1_prep1', "",
        'prob_meas2_prep1', "",
        'prob_meas0_prep2', "",
        'prob_meas1_prep2', "",
        'prob_meas2_prep2', "",
    ]
    hdf5_logs = [
        'fetched_data_arr', "FS",
        #'analysed_areas', "(FS)²",  TODO! Add capability of exporting other logs in the data exporter.
        #'analysed_means', "FS",
        #'analysed_perimeters', "FS",
        #'analysed_fidelities', "",
        #'weighed_areas', "(FS)²",
        #'weighed_means', "FS",
        #'weighed_perimeters', "FS",
        #'weighed_readout_fidelities', "",
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

def optimise_integration_window_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_A,
    readout_amp_A,
    readout_freq_B,
    readout_amp_B,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_rate,
    
    integration_window_start_min,
    integration_window_start_max,
    num_integration_window_start_steps,
    integration_window_stop_min,
    integration_window_stop_max,
    num_integration_window_stop_steps,
    
    control_port,
    control_freq_nco,
    control_freq_01,
    control_amp_01,
    control_duration_01,
    control_freq_12,
    control_amp_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    num_averages,
    num_shots_per_state,
    resonator_transmon_pair_id_number,
    
    reset_dc_to_zero_when_finished = True,
    
    my_weight_given_to_area_spanned_by_qubit_states = 0.0,
    my_weight_given_to_mean_distance_between_all_states = 0.0,
    my_weight_given_to_hamiltonian_path_perimeter = 0.0,
    my_weight_given_to_readout_fidelity = 1.0,
    
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    suppress_log_browser_export_of_suboptimal_data = True,
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    ## Input sanitation
    assert type(resonator_transmon_pair_id_number) == int, "Error: the argument resonator_transmon_pair_id_number expects an int, but a "+str(type(resonator_transmon_pair_id_number))+" was provided."
    
    ## Check that at least some kind of data will be saved.
    if  (my_weight_given_to_area_spanned_by_qubit_states == 0) and \
        (my_weight_given_to_mean_distance_between_all_states == 0) and \
        (my_weight_given_to_hamiltonian_path_perimeter == 0) and \
        (my_weight_given_to_readout_fidelity == 0):
        raise AttributeError("Error: All user-argument weights are set to 0, there is nothing to optimise since everything will be 0. No data will be saved. Halting.")
    
    # Check whether the integration window is legal.
    integration_window_stop = check_if_integration_window_is_legal(
        sample_rate = 1e9,
        sampling_duration = sampling_duration,
        integration_window_start = np.min([integration_window_start_min, integration_window_start_max]),
        integration_window_stop  = np.min([integration_window_stop_min,  integration_window_stop_max ]),
    )
    
    # Declare arrays for the integration window start and stop times.
    integration_window_start_arr = np.linspace(integration_window_start_min, integration_window_start_max, num_integration_window_start_steps)
    integration_window_stop_arr  = np.linspace(integration_window_stop_min,  integration_window_stop_max,  num_integration_window_stop_steps )
    
    # All output complex data plots will be stored for reference later on.
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
                ## Perform this step within a connection handler loop,
                ## to catch crashes.
                success = False
                tries = 0
                while ((not success) and (tries <= 5)):
                    tries += 1
                    try:
                        current_complex_dataset = get_complex_data_for_readout_optimisation_g_e_f(
                            ip_address = ip_address,
                            ext_clk_present = ext_clk_present,
                            
                            readout_stimulus_port = readout_stimulus_port,
                            readout_sampling_port = readout_sampling_port,
                            readout_freq_nco = readout_freq_nco,
                            readout_freq_of_static_resonator = readout_freq_A,
                            readout_amp_of_static_resonator = readout_amp_A,
                            readout_freq_of_swept_resonator = readout_freq_B,
                            readout_amp_of_swept_resonator = readout_amp_B,
                            readout_duration = readout_duration,
                            
                            sampling_duration = sampling_duration,
                            readout_sampling_delay = readout_sampling_delay,
                            repetition_rate = repetition_rate,
                            integration_window_start = curr_integration_start, # A swept parameter!
                            integration_window_stop  = curr_integration_stop,  # A swept parameter!
                            
                            control_port = control_port,
                            control_freq_nco = control_freq_nco,
                            control_amp_01 = control_amp_01,
                            control_freq_01 = control_freq_01,
                            control_duration_01 = control_duration_01,
                            control_amp_12 = control_amp_12,
                            control_freq_12 = control_freq_12,
                            control_duration_12 = control_duration_12,
                            
                            coupler_dc_port = coupler_dc_port,
                            coupler_dc_bias = coupler_dc_bias,
                            settling_time_of_bias_tee = settling_time_of_bias_tee,
                            
                            num_averages = num_averages,
                            num_shots_per_state = num_shots_per_state,
                            resonator_transmon_pair_id_number = resonator_transmon_pair_id_number,
                            
                            reset_dc_to_zero_when_finished = reset_dc_to_zero_when_finished,
                            
                            use_log_browser_database = use_log_browser_database,
                            suppress_log_browser_export = suppress_log_browser_export_of_suboptimal_data,
                            axes = axes
                        )
                        
                        success = True # Done
                    except ConnectionRefusedError:
                        if force_device_reboot_on_connection_error:
                            force_system_restart_over_ssh("129.16.115.184")
                assert success, "Halted! Unrecoverable crash detected."
                
                # current_complex_dataset will be a char array. Convert to string.
                current_complex_dataset = "".join(current_complex_dataset)
                
                ## Analyse the complex dataset. Note here that the
                ## discriminator's coordinate dataset *must* be changed
                ## as the integration window is altered. Because,
                ## changing the integration window, moves the complex states'
                ## population blob centre coordinates.
                
                # At this point, we must update the discriminator settings JSON.
                update_discriminator_settings_with_value(
                    path_to_data = os.path.abspath( current_complex_dataset )
                )
                ## TODO is it really needed to have both
                ## update_discriminator_settings_with_value, and
                ## calculate_area_mean_perimeter_fidelity, since
                ## the latter already calculates the population centres?
                ## Perhaps it is the the latter that should be changed?
                ## So that it doesn't re-calculate population centres for
                ## a provided set of data? Think about this choice.
                area_spanned, mean_state_distance, hamiltonian_path_perimeter, readout_fidelity, confusion_matrix = calculate_area_mean_perimeter_fidelity(
                    path_to_data = os.path.abspath( current_complex_dataset ),
                    update_discriminator_settings_json = True
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
    
    hdf5_steps = [
        'integration_window_stop_arr', "s",
        'integration_window_start_arr', "s",
    ]
    hdf5_singles = [
        'readout_stimulus_port', "",
        'readout_sampling_port', "",
        'readout_freq_nco', "Hz",
        'readout_freq_A', "Hz",
        'readout_amp_A', "FS",
        'readout_freq_B', "Hz",
        'readout_amp_B', "FS",
        'readout_duration', "s",
        
        'sampling_duration', "s",
        'readout_sampling_delay', "s",
        'repetition_rate', "s",
        
        'integration_window_start_min', "s",
        'integration_window_start_max', "s",
        'num_integration_window_start_steps', "",
        
        'integration_window_stop_min', "s",
        'integration_window_stop_max', "s",
        'num_integration_window_stop_steps', "",
        
        'control_port', "",
        'control_freq_nco', "Hz",
        'control_freq_01', "Hz",
        'control_amp_01', "FS",
        'control_duration_01', "s",
        'control_freq_12', "Hz",
        'control_amp_12', "FS",
        'control_duration_12', "s",
        
        #'coupler_dc_port', "",
        'coupler_dc_bias', "V",
        'settling_time_of_bias_tee,', "s",
        
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
        append_to_log_name_before_timestamp = 'integration_window',
        append_to_log_name_after_timestamp = '',
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export,
    )
    
    # Return.
    return string_arr_to_return


def get_time_traces_for_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_rate,
    
    control_port,
    control_freq_nco,
    
    control_amp_01,
    control_freq_01,
    control_duration_01,
    
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    num_averages,
    
    reset_dc_to_zero_when_finished = True,
    
    save_complex_data = True,
    use_log_browser_database = True,
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    ## Initial array declaration
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   False,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G10, DacFSample.G6, DacFSample.G6],
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
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            initialise_dc_bias(
                pulse_object = pls,
                static_dc_bias_or_list_to_sweep = coupler_dc_bias,
                coupler_dc_port = coupler_dc_port,
                settling_time_of_bias_tee = settling_time_of_bias_tee,
                safe_slew_rate = 20e-3, # V / s
            )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixer
        pls.hardware.configure_mixer(
            freq      = control_freq_nco,
            out_ports = control_port,
            tune      = True,
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
        readout_freq_if = readout_freq_nco - readout_freq
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = np.abs(readout_freq_if),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if),
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
        control_freq_if_01 = control_freq_nco - control_freq_01
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01),
        )
        control_freq_if_12 = control_freq_nco - control_freq_12
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = np.abs(control_freq_if_12),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_12),
        )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################

        # Start of sequence
        T = 0.0  # s
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # Do we have to perform an initial set sequence of the DC bias?
        if coupler_dc_port != []:
            T_begin = T # Get a time reference.
            T = change_dc_bias(pls, T, coupler_dc_bias, coupler_dc_port)
            T += settling_time_of_bias_tee
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
        # For all three states to get time traces of:
        for ii in range(3):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
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
            
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
        
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
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            destroy_dc_bias(
                pulse_object = pls,
                coupler_dc_port = coupler_dc_port,
                settling_time_of_bias_tee = settling_time_of_bias_tee,
                safe_slew_rate = 20e-3, # V / s
                static_offset_from_zero = 0.0, # V
            )
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        print("Raw data downloaded to PC.")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
            'time_vector', "s",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_nco', "Hz",
            'readout_freq', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s", 
            
            'control_port', "",
            'control_freq_nco', "Hz",
            
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12', "FS",
            'control_freq_12', "Hz",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'num_averages', "",
        ]
        hdf5_logs = []
        try:
            if len(states_to_discriminate_between) > 0:
                for statep in states_to_discriminate_between:
                    hdf5_logs.append('Probability for state |'+statep+'⟩')
                    hdf5_logs.append("")
            save_complex_data = False
        except NameError:
            pass # Fine, no state discrimnation.
        if len(hdf5_logs) == 0:
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
        string_arr_to_return.append(save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [ readout_freq_if ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = 0.0,
            integration_window_stop = 0.0,
            inner_loop_size = len(time_vector),
            outer_loop_size = 3, # Note! 3 b/c |g>, |e>, and |f>
            
            save_complex_data = save_complex_data,
            append_to_log_name_before_timestamp = 'time_traces_g_e_f' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            
            data_to_store_consists_of_time_traces_only = True,
        ))
    
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
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
        force_reload =   False,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G10, DacFSample.G6, DacFSample.G6],
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
            tune      = True,
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
        
        raise NotImplementedError("Halted! Surely, this measurement script cannot be complete? There is no IF argument provided in the .save() call below.")
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
        print("Raw data downloaded to PC.")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
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
        string_arr_to_return.append(save(
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
            
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            
            data_to_store_consists_of_time_traces_only = True,
        ))
    
    return string_arr_to_return