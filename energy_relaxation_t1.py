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
import shutil
import numpy as np
from numpy import hanning as von_hann
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
    get_timestamp_string
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save

def t1_01_sweep_coupler(
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
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_freq_nco,
    control_freq_01,
    control_amp_01,
    control_duration_01,
    
    coupler_dc_port,
    settling_time_of_bias_tee,
    
    num_averages,
    delay_arr,
    
    num_biases,
    coupler_bias_min = -0.0,
    coupler_bias_max = +0.0,
    
    reset_dc_to_zero_when_finished = True,
    
    save_complex_data = True,
    save_raw_time_data = False,
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
    ''' Characterises the energy relaxation time T1 as a function of applied
        coupler bias.
        
        delay_arr is the entire array of all delays to be applied in the
        experiment. Example: for a logarithmic sweep from 2 ns to 150 ns,
        done in 313 points, you would (as an argument) write:
            delay_arr = np.logspace( np.log10(2e-09), np.log10(150e-09), 313)
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    ## Input sanitisation
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = coupler_bias_min,
        coupler_bias_max = coupler_bias_max,
        num_biases       = num_biases,
        coupler_dc_bias  = None
    )
    
    ## Initial array declaration
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
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
                static_dc_bias_or_list_to_sweep = coupler_amp_arr,
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
        
        # Make all values in the delay_arr representatble.
        for jj in range(len(delay_arr)):
            delay_arr[jj] = int(round(delay_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the array.
        delay_arr = np.unique( np.array(delay_arr) )
        
        
        ''' Setup mixers '''
        
        # Readout port mixer
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
        # Control port amplitude for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        
        ### Setup readout pulse ###
        
        # Setup readout pulse envelope
        readout_pulse = pls.setup_long_drive(
            output_port = readout_stimulus_port,
            group       = 0,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
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
        
        
        ### Setup pulse "control_pulse_pi_01" ###

        # Setup control_pulse_pi_01 pulse envelope.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        # Setup control_pulse_pi_01 carrier tone, considering that there is a digital mixer.
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
            T = change_dc_bias(pls, T, coupler_amp_arr[0], coupler_dc_port)
            T += settling_time_of_bias_tee
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
        # Since we have two arrays to step through, and since the
        # Presto API as of 2023-02-02 does not support having
        # "variable LUTs" in the sequencer, we have zero choice
        # but to make a nested for loop, and setting the repeat argument to 1.
        for ii in range(len(coupler_amp_arr)):
            
            # For every delay to step through:
            for jj in range(len(delay_arr)):
                
                # Get a time reference, used for gauging the iteration length.
                T_begin = T
                
                # Output the pi_01-pulse
                pls.reset_phase(T, control_port)
                pls.output_pulse(T, control_pulse_pi_01)
                T += control_duration_01
                
                # Wait for the qubit to decohere some.
                T += delay_arr[jj]
                
                # Commence readout after said delay.
                pls.reset_phase(T, readout_stimulus_port)
                pls.output_pulse(T, readout_pulse)
                pls.store(T + readout_sampling_delay) # Sampling window
                T += readout_duration
                
                # Is this the last inner loop iteration?
                if jj == len(delay_arr)-1:
                    # It is, step the coupler bias to the next bias point?
                    if (coupler_dc_port != []) and (ii != len(coupler_amp_arr)-1):
                        # For all points that is not the last iteration,
                        # step the DC bias to the next point.
                        T = change_dc_bias(pls, T, coupler_amp_arr[ii+1], coupler_dc_port)
                        T += settling_time_of_bias_tee
                
                # Get T that aligns with the repetition rate.
                T, repetition_counter = get_repetition_rate_T(
                    T_begin, T, repetition_rate, repetition_counter,
                )
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        pls.run(
            period       = T,
            repeat_count = 1, # The repeat argument cannot be used for this measurement.
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
        
        hdf5_steps = [
            'delay_arr', "s",
            'coupler_amp_arr', "V",
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
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port', "",
            'control_freq_nco', "Hz",
            'control_freq_01', "Hz",
            'control_amp_01', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'settling_time_of_bias_tee', "s",
            
            'num_averages', "",
            
            'num_biases', "",
            'coupler_bias_min', "V",
            'coupler_bias_max', "V",
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
            if (len(hdf5_logs)/2 > 1):
                if not ( ('Probability for state |') in hdf5_logs[kk] ):
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
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(delay_arr),
            outer_loop_size = num_biases,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '_01' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def t1_01_sweep_coupler_multiplexed_ro0(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_A,
    readout_amp_A,
    readout_freq_B,
    readout_amp_B,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_averages,
    
    num_delays,
    dt_per_time_step,
    
    num_biases,
    coupler_bias_min = -1.0,
    coupler_bias_max = +1.0,
    
    reset_dc_to_zero_when_finished = True,
    
    save_complex_data = True,
    save_raw_time_data = False,
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
        "y_scaler": [1.0, 1.0],
        "y_offset": [0.0, 0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Characterises the energy relaxation time T1 as a function of applied
        coupler bias. Readout is multiplexed onto two resonators.
        
        ro0 designates that "the readout is done in state |0⟩."
    '''
    
    raise NotImplementedError("Halted! This function requires a larger update in terms of its DC biasing.")
    
    ## Input sanitisation
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = coupler_bias_min,
        coupler_bias_max = coupler_bias_max,
        num_biases       = num_biases,
        coupler_dc_bias  = None
    )
    
    ## Initial array declaration
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
    
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
        pls.hardware.set_dac_current(control_port_A, 40_500)
        pls.hardware.set_inv_sinc(control_port_A, 0)
        pls.hardware.set_dac_current(control_port_B, 40_500)
        pls.hardware.set_inv_sinc(control_port_B, 0)
        
        # Coupler port
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
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        dt_per_time_step = int(round(dt_per_time_step / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port, multiplexed, calculate an optimal NCO frequency.
        high_res  = max( [readout_freq_A, readout_freq_B] )
        low_res   = min( [readout_freq_A, readout_freq_B] )
        readout_freq_nco = high_res - (high_res - low_res)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_01_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_01_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = (coupler_dc_port == []),
        )
        # Coupler port mixer
        if coupler_dc_port != []:
            for curr_coupler_dc_port in range(len(coupler_dc_port)):
                pls.hardware.configure_mixer(
                    freq      = 0.0,
                    out_ports = coupler_dc_port,
                    tune      = True,
                    sync      = curr_coupler_dc_port == (len(coupler_dc_port)-1),
                )
        
        
        ''' Setup scale LUTs '''
        
        # Readout port amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp_A,
        )
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 1,
            scales          = readout_amp_B,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port_A,
            group           = 0,
            scales          = control_amp_01_A,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 0,
            scales          = control_amp_01_B,
        )
        # Coupler port amplitude (the bias to be swept)
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_amp_arr,
            )


        ### Setup readout pulses ###
        
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
        assert 1 == 0, "Halted! The readout in this function needs to be updated to automatic sideband selection."
        # Setup readout carriers, considering the multiplexed readout NCO.
        readout_freq_if_A = np.abs(readout_freq_nco - readout_freq_A)
        readout_freq_if_B = np.abs(readout_freq_nco - readout_freq_B)
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_A,
            phases       = np.full_like(readout_freq_if_A, 0.0),
            phases_q     = np.full_like(readout_freq_if_A, -np.pi/2), # USB ! ##+np.pi/2, # LSB
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = readout_freq_if_B,
            phases       = np.full_like(readout_freq_if_B, 0.0),
            phases_q     = np.full_like(readout_freq_if_B, -np.pi/2), # USB!  ##+np.pi/2, # LSB
        )
        
        
        ### Setup pulses "control_pulse_pi_01_A" and "control_pulse_pi_01_B ###
        
        # Setup control_pulse_pi_01_A and _B pulse envelopes.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_pulse_pi_01_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        # Setup control_pulse_pi_01 _A and _B carrier tones, considering that there is a digital mixer.
        assert 1 == 0, "Halted! The control pulse generation in this function needs to be updated to automatic sideband selection."
        pls.setup_freq_lut(
            output_ports    = control_port_A,
            group           = 0,
            frequencies     = 0.0,
            phases          = 0.0,
            phases_q        = 0.0,
        )
        pls.setup_freq_lut(
            output_ports    = control_port_B,
            group           = 0,
            frequencies     = 0.0,
            phases          = 0.0,
            phases_q        = 0.0,
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
        
        assert 1 == 0, "Halted! The data saving routine in this function has to be updated."
        
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
        
        for ii in range(num_delays):
            
            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
            if coupler_dc_port != []:
                for bias_tone in coupler_bias_tone:
                    coupler_bias_tone.set_total_duration(
                        control_duration_01 + \
                        ii * dt_per_time_step + \
                        readout_duration + \
                        repetition_delay \
                    )
            
                # Re-apply the coupler bias tone.
                pls.output_pulse(T, coupler_bias_tone)
            
            # Output the pi01-pulses
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
            T += control_duration_01
            
            # Wait for the qubit to decohere some.
            T += ii * dt_per_time_step
            
            # Commence readout after said delay.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next dt step in the T1 measurement.
            T += repetition_delay
        
        # Increment the coupler port's DC bias.
        if coupler_dc_port != []:
            pls.next_scale(T, coupler_dc_port)
        
        # Move to next iteration.
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        assert 1 == 0, "Halted! This function (or its DC biasing) does not have a proper reset clause."
        
        pls.run(
            period       = T,
            repeat_count = num_biases,
            num_averages = num_averages,
            print_time   = True,
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
        ## Create a linear vector corresponding to the time sweep.
        delay_arr = np.linspace(0, num_delays * dt_per_time_step, num_delays)
        
        hdf5_steps = [
            'delay_arr', "s",
            'coupler_amp_arr', "FS",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            
            'control_port_A', "",
            'control_freq_01_A', "Hz", 
            'control_amp_01_A', "FS",
            'control_port_B', "",
            'control_freq_01_B', "Hz", 
            'control_amp_01_B', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'added_delay_for_bias_tee', "s",
            
            'num_averages', "",
            'num_delays', "",
            'dt_per_time_step', "s",
            
            'num_biases', "",
            'coupler_bias_min', "FS",
            'coupler_bias_max', "FS",
        ]
        hdf5_logs = [
            'fetched_data_arr_1', "FS",
            'fetched_data_arr_2', "FS",
        ]
        
        # Assert that the received keys bear (an even number of) entries,
        # implying whether a unit is missing.
        number_of_keyed_elements_is_even = \
            ((len(hdf5_steps) % 2) == 0) and \
            ((len(hdf5_singles) % 2) == 0) and \
            ((len(hdf5_logs) % 2) == 0)
        assert number_of_keyed_elements_is_even, "Error: non-even amount "  + \
            "of keys and units provided. Someone likely forgot a comma."
        
        # Stylistically rework underscored characters in the axes dict.
        for axis in ['x_name','x_unit','y_name','y_unit','z_name','z_unit']:
            axes[axis] = axes[axis].replace('/2','/₂')
            axes[axis] = axes[axis].replace('/3','/₃')
            axes[axis] = axes[axis].replace('_01','₀₁')
            axes[axis] = axes[axis].replace('_02','₀₂')
            axes[axis] = axes[axis].replace('_03','₀₃')
            axes[axis] = axes[axis].replace('_12','₁₂')
            axes[axis] = axes[axis].replace('_13','₁₃')
            axes[axis] = axes[axis].replace('_20','₂₀')
            axes[axis] = axes[axis].replace('_23','₂₃')
            axes[axis] = axes[axis].replace('_0','₀')
            axes[axis] = axes[axis].replace('_1','₁')
            axes[axis] = axes[axis].replace('_2','₂')
            axes[axis] = axes[axis].replace('_3','₃')
            axes[axis] = axes[axis].replace('lambda','λ')
            axes[axis] = axes[axis].replace('Lambda','Λ')
        
        # Build step lists, re-scale and re-unit where necessary.
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            if (hdf5_steps[ii] != 'fetched_data_arr') and (hdf5_steps[ii] != 'time_vector'):
                temp_name   = hdf5_steps[ii]
                temp_object = np.array( eval(hdf5_steps[ii]) )
                temp_unit   = hdf5_steps[ii+1]
                if ii == 0:
                    if (axes['x_name']).lower() != 'default':
                        # Replace the x-axis name
                        temp_name = axes['x_name']
                    if axes['x_scaler'] != 1.0:
                        # Re-scale the x-axis
                        temp_object *= axes['x_scaler']
                    if (axes['x_unit']).lower() != 'default':
                        # Change the unit on the x-axis
                        temp_unit = axes['x_unit']
                elif ii == 2:
                    if (axes['z_name']).lower() != 'default':
                        # Replace the z-axis name
                        temp_name = axes['z_name']
                    if axes['z_scaler'] != 1.0:
                        # Re-scale the z-axis
                        temp_object *= axes['z_scaler']
                    if (axes['z_unit']).lower() != 'default':
                        # Change the unit on the z-axis
                        temp_unit = axes['z_unit']
                ext_keys.append(dict(name=temp_name, unit=temp_unit, values=temp_object))
        for jj in range(0,len(hdf5_singles),2):
            if (hdf5_singles[jj] != 'fetched_data_arr') and (hdf5_singles[jj] != 'time_vector'):
                temp_object = np.array( [eval(hdf5_singles[jj])] )
                ext_keys.append(dict(name=hdf5_singles[jj], unit=hdf5_singles[jj+1], values=temp_object))
        
        log_dict_list = []
        for qq in range(len(axes['y_scaler'])):
            if (axes['y_scaler'])[qq] != 1.0:
                ext_keys.append(dict(name='Y-axis scaler for Y'+str(qq+1), unit='', values=(axes['y_scaler'])[qq]))
            if (axes['y_offset'])[qq] != 0.0:
                ext_keys.append(dict(name='Y-axis offset for Y'+str(qq+1), unit=hdf5_logs[2*qq+1], values=(axes['y_offset'])[qq]))
        for kk in range(0,len(hdf5_logs),2):
            log_entry_name = hdf5_logs[kk]
            # Set unit on the y-axis
            if (axes['y_unit']).lower() != 'default':
                temp_log_unit = axes['y_unit']
            else:
                temp_log_unit = hdf5_logs[kk+1]
            if (axes['y_name']).lower() != 'default':
                # Replace the y-axis name
                log_entry_name = axes['y_name']
                if len(hdf5_logs)/2 > 1:
                    log_entry_name += (' ('+str((kk+2)//2)+' of '+str(len(hdf5_logs)//2)+')')
            log_dict_list.append(dict(name=log_entry_name, unit=temp_log_unit, vector=False, complex=save_complex_data))
        
        # Save data!
        string_arr_to_return.append(save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_delays,
            outer_loop_size = num_biases,
            
            save_complex_data = save_complex_data,
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '_01_multiplexed_ro' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def t1_12_sweep_coupler_ro1(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_excited,
    readout_amp,
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
    settling_time_of_bias_tee,
    
    num_averages,
    delay_arr,
    
    num_biases,
    coupler_bias_min = -0.0,
    coupler_bias_max = +0.0,
    
    reset_dc_to_zero_when_finished = True,
    
    save_complex_data = True,
    save_raw_time_data = False,
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
    ''' Characterises the energy relaxation time T1 as a function of applied
        coupler bias.
        
        delay_arr is the entire array of all delays to be applied in the
        experiment. Example: for a logarithmic sweep from 2 ns to 150 ns,
        done in 313 points, you would (as an argument) write:
            delay_arr = np.logspace( np.log10(2e-09), np.log10(150e-09), 313)
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    assert 1 == 0, "Halted! This function is finished, but it has not yet been tested."
    ## Input sanitisation
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = coupler_bias_min,
        coupler_bias_max = coupler_bias_max,
        num_biases       = num_biases,
        coupler_dc_bias  = None
    )
    
    ## Initial array declaration
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
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
                static_dc_bias_or_list_to_sweep = coupler_amp_arr,
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
        
        # Make all values in the delay_arr representatble.
        for jj in range(len(delay_arr)):
            delay_arr[jj] = int(round(delay_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the array.
        delay_arr = np.unique( np.array(delay_arr) )
        
        
        ''' Setup mixers '''
        
        # Readout port mixer
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
        # Control port amplitude for pi_01 and pi_12
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
            output_port = readout_stimulus_port,
            group       = 0,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        readout_freq_if = readout_freq_nco - readout_freq_excited
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
        T = 0.0 # s
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # Do we have to perform an initial set sequence of the DC bias?
        if coupler_dc_port != []:
            T_begin = T # Get a time reference.
            T = change_dc_bias(pls, T, coupler_amp_arr[0], coupler_dc_port)
            T += settling_time_of_bias_tee
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
        # Since we have two arrays to step through, and since the
        # Presto API as of 2023-02-02 does not support having
        # "variable LUTs" in the sequencer, we have zero choice
        # but to make a nested for loop, and setting the repeat argument to 1.
        for ii in range(len(coupler_amp_arr)):
            
            # For every delay to step through:
            for jj in range(len(delay_arr)):
                
                # Get a time reference, used for gauging the iteration length.
                T_begin = T
                
                # Output the pi_01-pulse
                pls.reset_phase(T, control_port)
                pls.output_pulse(T, control_pulse_pi_01)
                T += control_duration_01
                
                # Output the pi_12-pulse
                pls.reset_phase(T, control_port)
                pls.output_pulse(T, control_pulse_pi_12)
                T += control_duration_12
                
                # Wait for the qubit to decohere some.
                T += delay_arr[jj]
                
                # Commence readout after said delay.
                pls.reset_phase(T, readout_stimulus_port)
                pls.output_pulse(T, readout_pulse)
                pls.store(T + readout_sampling_delay) # Sampling window
                T += readout_duration
                
                # Is this the last inner loop iteration?
                if jj == len(delay_arr)-1:
                    # It is, step the coupler bias to the next bias point?
                    if (coupler_dc_port != []) and (ii != len(coupler_amp_arr)-1):
                        # For all points that is not the last iteration,
                        # step the DC bias to the next point.
                        T = change_dc_bias(pls, T, coupler_amp_arr[ii+1], coupler_dc_port)
                        T += settling_time_of_bias_tee
                
                # Get T that aligns with the repetition rate.
                T, repetition_counter = get_repetition_rate_T(
                    T_begin, T, repetition_rate, repetition_counter,
                )
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        pls.run(
            period       = T,
            repeat_count = 1, # The repeat argument cannot be used for this measurement.
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
        
        hdf5_steps = [
            'delay_arr', "s",
            'coupler_amp_arr', "V",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_nco', "Hz",
            'readout_freq_excited', "Hz",
            'readout_amp', "FS",
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
            'settling_time_of_bias_tee', "s",
            
            'num_averages', "",
            
            'num_biases', "",
            'coupler_bias_min', "V",
            'coupler_bias_max', "V",
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
            if (len(hdf5_logs)/2 > 1):
                if not ( ('Probability for state |') in hdf5_logs[kk] ):
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
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = len(delay_arr),
            outer_loop_size = num_biases,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '_12_ro1' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return
