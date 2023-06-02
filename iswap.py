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
from datetime import datetime
from phase_calculator import \
    legalise_phase_array, \
    reset_phase_counter, \
    add_virtual_z, \
    track_phase, \
    bandsign
from bias_calculator import \
    sanitise_dc_bias_arguments, \
    get_dc_dac_range_integer, \
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

def iswap_sweep_duration_and_detuning(
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
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_amp_iswap,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap_centre,
    coupler_ac_freq_iswap_span,
    
    num_freqs,
    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    
    prepare_input_state = '10',
    
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
    ''' Tune an iSWAP-interaction between two qubits using
        a tuneable coupler, by fixing the gate amplitude and gate bias.
        Thus, the gate duration and detuning is swept.
        
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
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    # Sanitisation for whether the user has a
    # span engaged but only a single frequency.
    if ((num_freqs == 1) and (coupler_ac_freq_iswap_span != 0.0)):
        print("Note: single coupler frequency point requested, ignoring span parameter.")
        coupler_ac_freq_iswap_span = 0.0
    
    ## Assure that the requested input state to prepare, is valid.
    assert ( (prepare_input_state == '10') or (prepare_input_state == '01') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'10\' and \'01\'"
    
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_min = int(round(coupler_ac_plateau_duration_iswap_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_max = int(round(coupler_ac_plateau_duration_iswap_max / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Make the user-set time variables representable '''
        
        # For all elements, round to the programmable logic clock period.
        # Then, remove duplicates and update the num_time_steps parameter.
        iswap_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_iswap_min + 2 * coupler_ac_single_edge_time_iswap, \
            coupler_ac_plateau_duration_iswap_max + 2 * coupler_ac_single_edge_time_iswap, \
            num_time_steps
        )
        for jj in range(len(iswap_total_pulse_duration_arr)):
            iswap_total_pulse_duration_arr[jj] = int(round(iswap_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the array.
        iswap_total_pulse_duration_arr = np.unique( np.array(iswap_total_pulse_duration_arr) )
        num_time_steps = len(iswap_total_pulse_duration_arr)
        
        
        ''' Setup mixers '''
        
        # Readout port mixer
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = True,
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_iswap,
        )
        
        
        ### Setup readout pulses ###
        
        # Setup readout pulse envelope
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_A),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_A),
        )
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_B),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_B),
        )
        
        
        ### Setup the iSWAP gate pulse
        
        # The initially set duration is temporary, and will be swept by the
        # sequencer program.
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_plateau_duration_iswap_min + \
                          2 * coupler_ac_single_edge_time_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        # Setup the iSWAP pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_iswap_centre_if = coupler_ac_freq_nco - coupler_ac_freq_iswap_centre  
        f_start = coupler_ac_freq_iswap_centre_if - coupler_ac_freq_iswap_span / 2
        f_stop  = coupler_ac_freq_iswap_centre_if + coupler_ac_freq_iswap_span / 2
        coupler_ac_freq_iswap_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_iswap_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_iswap_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_iswap_if_arr),
            phases          = np.full_like(coupler_ac_freq_iswap_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_iswap_if_arr, bandsign(coupler_ac_freq_iswap_centre_if)),
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
        
        # For every pulse duration to sweep over:
        for ii in range(len(iswap_total_pulse_duration_arr)):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Redefine the iSWAP pulse's total duration,
            coupler_ac_duration_iswap = iswap_total_pulse_duration_arr[ii]
            coupler_ac_pulse_iswap.set_total_duration(coupler_ac_duration_iswap)
            
            # Put the system into state |01> or |10> with pi01 pulse(s)
            pls.reset_phase(T, [control_port_A, control_port_B])
            if prepare_input_state == '10':
                pls.output_pulse(T, control_pulse_pi_01_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_B)
            T += control_duration_01
            
            # Apply the iSWAP gate, with parameters being swept.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Is this the last iteration?
            if ii == len(iswap_total_pulse_duration_arr)-1:
                # Move to the next scanned frequency
                pls.next_frequency(T, coupler_ac_port, group = 0)
                T += 20e-9 # Add some time for changing the frequency.
            
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
            repeat_count    =   num_freqs,
            num_averages    =   num_averages,
            print_time      =   True,
        )
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
    
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
            'iswap_total_pulse_duration_arr', "s",
            'coupler_ac_pulse_iswap_freq_arr', "Hz",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port_A', "",
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz",
            'control_port_B,', "",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_iswap_centre_if', "Hz",
            'coupler_ac_freq_iswap_span', "Hz",
            
            'num_freqs', "",
            'num_averages', "",
            'num_time_steps', "",
            
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap_min', "s",
            'coupler_ac_plateau_duration_iswap_max', "s",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_time_steps,
            outer_loop_size = num_freqs,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'sweep_duration_and_detuning',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def iswap_sweep_duration_and_detuning_state_probability(
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
    control_port_B,
    
    control_amp_01_A,
    control_freq_01_A,
    control_amp_01_B,
    control_freq_01_B,
    control_duration_01,
    
    control_freq_12_A,
    control_freq_12_B,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_amp_iswap,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap_centre,
    coupler_ac_freq_iswap_span,
    
    num_freqs,
    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['01', '10'],
    
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
    ''' Tune an iSWAP-interaction between two qubits using
        a tuneable coupler, by fixing the gate amplitude and gate bias.
        Thus, the gate duration and detuning is swept.
        
        The end result is state discriminated.
    '''
    
    ## Input sanitisation
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    # Sanitisation for whether the user has a
    # span engaged but only a single frequency.
    if ((num_freqs == 1) and (coupler_ac_freq_iswap_span != 0.0)):
        print("Note: single coupler frequency point requested, ignoring span parameter.")
        coupler_ac_freq_iswap_span = 0.0
    
    
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
        
        # Coupler port(s)
        if coupler_dc_port != []:
            pls.hardware.set_dac_current(coupler_dc_port, 40_500)
            pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_min = int(round(coupler_ac_plateau_duration_iswap_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_max = int(round(coupler_ac_plateau_duration_iswap_max / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Make the user-set time variables representable '''
        
        # For all elements, round to the programmable logic clock period.
        # Then, remove duplicates and update the num_time_steps parameter.
        iswap_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_iswap_min + 2 * coupler_ac_single_edge_time_iswap, \
            coupler_ac_plateau_duration_iswap_max + 2 * coupler_ac_single_edge_time_iswap, \
            num_time_steps
        )
        for jj in range(len(iswap_total_pulse_duration_arr)):
            iswap_total_pulse_duration_arr[jj] = int(round(iswap_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        new_list = []
        for kk in range(len(iswap_total_pulse_duration_arr)):
            if not (iswap_total_pulse_duration_arr[kk] in new_list):
                new_list.append(iswap_total_pulse_duration_arr[kk])
        iswap_total_pulse_duration_arr = new_list
        num_time_steps = len(iswap_total_pulse_duration_arr)
        
        
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
        high_res_A  = max( [control_freq_01_A, control_freq_12_A] )
        low_res_A   = min( [control_freq_01_A, control_freq_12_A] )
        control_freq_nco_A = high_res_A - (high_res_A - low_res_A)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        high_res_B  = max( [control_freq_01_B, control_freq_12_B] )
        low_res_B   = min( [control_freq_01_B, control_freq_12_B] )
        control_freq_nco_B = high_res_B - (high_res_B - low_res_B)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = (coupler_dc_port == []),
        )
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_iswap,
        )
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
            )
        
        
        ### Setup readout pulses ###
        
        # Setup readout pulse envelope
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_A),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_A),
        )
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_B),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_B),
        )
        
        
        ### Setup the iSWAP gate pulse
        
        # The initially set duration is temporary, and will be swept by the
        # sequencer program.
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_plateau_duration_iswap_min + \
                          2 * coupler_ac_single_edge_time_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        # Setup the iSWAP pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_iswap_centre_if = coupler_ac_freq_nco - coupler_ac_freq_iswap_centre  
        f_start = coupler_ac_freq_iswap_centre_if - coupler_ac_freq_iswap_span / 2
        f_stop  = coupler_ac_freq_iswap_centre_if + coupler_ac_freq_iswap_span / 2
        coupler_ac_freq_iswap_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_iswap_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_iswap_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_iswap_if_arr),
            phases          = np.full_like(coupler_ac_freq_iswap_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_iswap_if_arr, bandsign(coupler_ac_freq_iswap_centre_if)),
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
        T = 0.0  # s

        # Charge the bias tee.
        if coupler_dc_port != []:
            pls.reset_phase(T, coupler_dc_port)
            pls.output_pulse(T, coupler_bias_tone)
            T += added_delay_for_bias_tee
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # For every AC coupler frequency to sweep over:
        for jj in range(num_freqs):
        
            # For every pulse duration to sweep over:
            for ii in iswap_total_pulse_duration_arr:

                # Redefine the iSWAP pulse's total duration,
                coupler_ac_duration_iswap = ii
                coupler_ac_pulse_iswap.set_total_duration(coupler_ac_duration_iswap)
                
                # Redefine the coupler DC pulse duration to keep on playing once
                # the bias tee has charged.
                if coupler_dc_port != []:
                    coupler_bias_tone.set_total_duration(
                        control_duration_01 + \
                        coupler_ac_duration_iswap + \
                        readout_duration + \
                        repetition_delay \
                    )
                    
                    # Re-apply the coupler bias tone.
                    pls.output_pulse(T, coupler_bias_tone)
                
                # Put the system into state |01> or |10> with pi01 pulse(s)
                pls.reset_phase(T, [control_port_A, control_port_B])
                pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
                T += control_duration_01
                
                # Apply the iSWAP gate, with parameters being swept.
                pls.reset_phase(T, coupler_ac_port)
                pls.output_pulse(T, coupler_ac_pulse_iswap)
                T += coupler_ac_duration_iswap
                
                # Commence multiplexed readout
                pls.reset_phase(T, readout_stimulus_port)
                pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                pls.store(T + readout_sampling_delay) # Sampling window
                T += readout_duration
                
                # Await a new repetition, after which a new coupler DC bias tone
                # will be added - and a new frequency set for the readout tone.
                T = repetition_delay * repetition_counter
                repetition_counter += 1
            
            # Move to the next scanned frequency
            pls.next_frequency(T, coupler_ac_port, group = 0)
            
            # Move to next iteration.
            T = repetition_delay * repetition_counter
            repetition_counter += 1
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        assert 1 == 0, "Halted! This function's DC biasing has not been modernised."
        
        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   num_single_shots,
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
            'iswap_total_pulse_duration_arr', "s",
            'coupler_ac_pulse_iswap_freq_arr', "Hz",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            
            'control_port_A', "",
            'control_port_B,', "",
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz",
            'control_duration_01', "s",
            
            'control_freq_12_A', "Hz",
            'control_freq_12_B', "Hz",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_iswap_centre', "Hz",
            'coupler_ac_freq_iswap_span', "Hz",
            
            'num_freqs', "",
            'num_averages', "",
            'num_time_steps', "",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap_min', "s",
            'coupler_ac_plateau_duration_iswap_max', "s",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_time_steps,
            outer_loop_size = num_freqs,
            
            single_shot_repeats_to_discretise = num_single_shots,
            ordered_resonator_ids_in_readout_data = resonator_ids,
            get_probabilities_on_these_states = states_to_discriminate_between,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'sweep_duration_and_detuning_state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def iswap_sweep_duration_and_amplitude(
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
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_averages,
    num_amplitudes,
    num_time_steps,
    
    prepare_input_state = '10',
    
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
    ''' Tune an iSWAP-interaction between two qubits, where it is known at
        what gate frequency the iSWAP interaction takes place (and at
        which coupler bias), but not the iSWAP gate amplitude nor the gate
        duration.
        
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
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    ## Assure that the requested input state to prepare, is valid.
    assert ( (prepare_input_state == '10') or (prepare_input_state == '01') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'10\' and \'01\'"
    
    ## Initial array declaration
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_min = int(round(coupler_ac_plateau_duration_iswap_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_max = int(round(coupler_ac_plateau_duration_iswap_max / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Make the user-set time variables representable '''
        
        # For all elements, round to the programmable logic clock period.
        # Then, remove duplicates and update the num_time_steps parameter.
        iswap_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_iswap_min + 2 * coupler_ac_single_edge_time_iswap, \
            coupler_ac_plateau_duration_iswap_max + 2 * coupler_ac_single_edge_time_iswap, \
            num_time_steps
        )
        for jj in range(len(iswap_total_pulse_duration_arr)):
            iswap_total_pulse_duration_arr[jj] = int(round(iswap_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the array.
        iswap_total_pulse_duration_arr = np.unique( np.array(iswap_total_pulse_duration_arr) )
        num_time_steps = len(iswap_total_pulse_duration_arr)
        
        
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
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = True,
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_arr, # This value will be swept!
        )
        
        
        ### Setup readout pulses ###
        
        # Setup readout pulse envelope
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        
        # Setup control pulse carriers, considering that there is a digital mixer
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_A),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_A),
        )
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_B),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_B),
        )
        
        
        ### Setup the iSWAP gate pulse
        
        # The initially set duration is temporary, and will be swept by the
        # sequencer program.
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_plateau_duration_iswap_min + \
                          2 * coupler_ac_single_edge_time_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        # Setup the iSWAP pulse carrier.
        coupler_ac_freq_if_iswap = coupler_ac_freq_nco - coupler_ac_freq_iswap
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_if_iswap),
            phases          = 0.0,
            phases_q        = bandsign(coupler_ac_freq_if_iswap),
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
        
        # For every pulse duration to sweep over:
        for ii in range(len(iswap_total_pulse_duration_arr)):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Redefine the iSWAP pulse's total duration,
            coupler_ac_duration_iswap = iswap_total_pulse_duration_arr[ii]
            coupler_ac_pulse_iswap.set_total_duration(coupler_ac_duration_iswap)
            
            # Put the system into state |01> or |10> with pi01 pulse(s)
            pls.reset_phase(T, [control_port_A, control_port_B])
            if prepare_input_state == '10':
                pls.output_pulse(T, control_pulse_pi_01_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_B)
            T += control_duration_01
            
            # Apply the iSWAP gate, with parameters being swept.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Is this the last iteration?
            if ii == len(iswap_total_pulse_duration_arr)-1:
                # Increment the swept amplitude.
                pls.next_scale(T, coupler_ac_port, group = 0)
                T += 20e-9 # Add some time for changing the amplitude.
            
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
            repeat_count    =   num_amplitudes,
            num_averages    =   num_averages,
            print_time      =   True,
        )
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        print("Raw data downloaded to PC.")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        # Note that typically, the variable that matches "the inner loop"
        # would be listed first. This specific subroutine is making an
        # exception to this b/c order-of-operations restrictions in the
        # Labber Log browser. All in all, this order reversal here
        # also means that the store_data shape is also altered.
        hdf5_steps = [
            'iswap_total_pulse_duration_arr', "s",
            'coupler_ac_amp_arr', "FS",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco', "Hz",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s",
            
            'control_port_A', "",
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz",
            'control_port_B', "",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_iswap', "Hz",
            
            'num_amplitudes', "",
            'coupler_ac_amp_min', "FS",
            'coupler_ac_amp_max', "FS",

            'num_averages', "",
            'num_time_steps', "",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap_min', "s",
            'coupler_ac_plateau_duration_iswap_max', "s",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_time_steps,
            outer_loop_size = num_amplitudes,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'sweep_duration_and_amplitude',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def iswap_sweep_amplitude_and_detuning(
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
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap_centre,
    coupler_ac_freq_iswap_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_amplitudes,
    
    num_averages,
    
    prepare_input_state = '10',
    
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
    ''' Tune an iSWAP-interaction between two qubits using
        a tuneable coupler.
        
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
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    # Sanitisation for whether the user has a
    # span engaged but only a single frequency.
    if ((num_freqs == 1) and (coupler_ac_freq_iswap_span != 0.0)):
        print("Note: single coupler frequency point requested, ignoring span parameter.")
        coupler_ac_freq_iswap_span = 0.0
    
    ## Assure that the requested input state to prepare, is valid.
    assert ( (prepare_input_state == '10') or (prepare_input_state == '01') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'10\' and \'01\'"
    
    ## Initial array declaration
    
    # Declare amplitude array for the AC coupler tone to be swept.
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
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
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = True,
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_arr,
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_A),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_A),
        )
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_B),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_B),
        )
        
        
        ### Setup the iSWAP gate pulse
        
        # The initially set duration will not be swept by the sequencer
        # program.
        coupler_ac_duration_iswap = \
            2 * coupler_ac_single_edge_time_iswap + \
            coupler_ac_plateau_duration_iswap
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        # Setup the iSWAP pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_iswap_centre_if = coupler_ac_freq_nco - coupler_ac_freq_iswap_centre  
        f_start = coupler_ac_freq_iswap_centre_if - coupler_ac_freq_iswap_span / 2
        f_stop  = coupler_ac_freq_iswap_centre_if + coupler_ac_freq_iswap_span / 2
        coupler_ac_freq_iswap_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_iswap_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_iswap_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_iswap_if_arr),
            phases          = np.full_like(coupler_ac_freq_iswap_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_iswap_if_arr, bandsign(coupler_ac_freq_iswap_centre_if)),
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
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Put the system into state |01> or |10> with pi01 pulse(s)
            pls.reset_phase(T, [control_port_A, control_port_B])
            if prepare_input_state == '10':
                pls.output_pulse(T, control_pulse_pi_01_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_B)
            T += control_duration_01
            
            # Apply the iSWAP gate, with parameters being swept.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next scanned frequency
            pls.next_frequency(T, coupler_ac_port, group = 0)
            T += 20e-9 # Add some time for changing the frequency.
            
            # Is this the last iteration?
            if ii == num_freqs-1:
                # Increment the swept amplitude.
                pls.next_scale(T, coupler_ac_port, group = 0)
                T += 20e-9 # Add some time for changing the amplitude.
            
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
            repeat_count    =   num_amplitudes,
            num_averages    =   num_averages,
            print_time      =   True,
        )
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
    
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
            'coupler_ac_pulse_iswap_freq_arr', "Hz",
            'coupler_ac_amp_arr', "FS",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco', "Hz",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s",
            
            'control_port_A', "",
            'control_port_B', "",
            
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_duration_iswap', "s",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_iswap_centre', "Hz",
            'coupler_ac_freq_iswap_span', "Hz",
            
            'coupler_ac_amp_min', "FS",
            'coupler_ac_amp_max', "FS",
            
            'num_freqs', "",
            'num_averages', "",
            'num_amplitudes', "",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_freqs,
            outer_loop_size = num_amplitudes,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'sweep_amplitude_and_detuning',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def iswap_tune_local_accumulated_phase(
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
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap,
    coupler_ac_amp_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_iswap_A = 0.0,
    phase_adjustment_after_iswap_B = 0.0,
    phase_adjustment_after_iswap_C = 0.0,
    
    prepare_input_state = '+0',
    analyse_iswap_phase_using_this_qubit = 'A',
    
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
    ''' Tune the phases of the individual qubits partaking in an iSWAP gate.
        Based on the methods from:
        
        M. Ganzhorn et al. PR Research 2 033447 (2020)
        https://doi.org/10.48550/arXiv.2005.05696
        
        ... with theoretical assistance found in:
        
        Deanna M. Abrams et al. Nature electronics, 3, December 2020 pp.744–750
        https://doi.org/10.48550/arXiv.1912.04424
        
        The method is similar to a Ramsey sequence.
        The input state is prepared to |+0⟩ (or |0+⟩).
        Then, run an iSWAP, followed by an iSWAP†.
        Finally, put |+0⟩ (or |0+⟩) back to |10⟩ (or |01⟩), readout in Z.
        
        analyse_iswap_phase_using_this_qubit dictates which qubit will have
        its final π/2-pulse's phase swept in the last quantum circuit moment.
    '''
    
    ## Input sanitisation
    
    # Already here at the beginning, we may calculate a rough placeholder
    # for the necessary post-iSWAP frame correction on the coupler drive.
    # If, the local accumulated phase is known for both qubit A and qubit B.
    if ((phase_adjustment_after_iswap_C == 0.0) and \
        (phase_adjustment_after_iswap_A != 0.0) and \
        (phase_adjustment_after_iswap_B != 0.0)):
        # The phase adjustment on the coupler drive is unknown,
        # but the local accumulated phase of qubit A and qubit B are known.
        # Thus, calculate an estimate of the coupler drive phase adjustment.
        phase_adjustment_after_iswap_C = phase_adjustment_after_iswap_B - phase_adjustment_after_iswap_A
        print("Note: no post-iSWAP coupler drive phase adjustment was provided, but both qubits' local accumulated phases were provided. An estimate of the coupler drive phase adjustment was set: "+str(phase_adjustment_after_iswap_C)+" rad.")
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    ## Assure that the requested input state to prepare, is valid.
    assert ( \
        (prepare_input_state == '0+') or \
        (prepare_input_state == '+0') or \
        (prepare_input_state == '++') or \
        (prepare_input_state == '1+') or \
        (prepare_input_state == '+1') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'0+\', \'+0\', \'++\', \'1+\' and \'+1\'"
    assert ( \
        (analyse_iswap_phase_using_this_qubit == 'A') or \
        (analyse_iswap_phase_using_this_qubit == 'B') ),\
        "Error! Invalid qubit selected for analysis. " + \
        "Legal values are \'A\' and \'B\'"
    
    ## Initial array declaration
    
    # Declare what phases are available
    phases_declared = np.linspace(0, 2*np.pi, 512)
    
    # Declare phase array to sweep, and make it legal.
    control_phase_arr = np.linspace(phase_sweep_rad_min, phase_sweep_rad_max, num_phases)
    control_phase_arr = legalise_phase_array( control_phase_arr, phases_declared )
    num_phases = len(control_phase_arr)
    
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port mixer
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = True,
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_iswap,
        )
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            scales          = coupler_ac_amp_iswap,
        )
        
        ### Setup readout pulses ###
        
        # Setup readout pulse envelope
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        control_pulse_pi_01_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_pulse_pi_01_half_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tones, considering that there are
        # digital mixers.
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_A)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_A))
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_B)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_B))
        )
        
        
        ### Setup the iSWAP gate pulse
        
        # In the actual experiment, there will be a "normal" iSWAP pulse
        # followed by an iSWAP†.
        coupler_ac_duration_iswap = \
            coupler_ac_plateau_duration_iswap + \
            2 * coupler_ac_single_edge_time_iswap
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        coupler_ac_pulse_iswap_inverted = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 1,
            duration    = coupler_ac_duration_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        
        ## Setup the iSWAP pulse carrier.
        
        # Setup LUT
        coupler_ac_freq_iswap_if = coupler_ac_freq_nco - coupler_ac_freq_iswap
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.full_like(phases_declared, np.abs(coupler_ac_freq_iswap_if)),
            phases          = phases_declared,
            phases_q        = phases_declared + np.full_like(phases_declared, bandsign(coupler_ac_freq_iswap_if)),
        )
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = np.full_like(phases_declared, np.abs(coupler_ac_freq_iswap_if)),
            phases          = phases_declared + np.full_like(phases_declared, np.pi), # 180 degree inversion
            phases_q        = phases_declared + np.full_like(phases_declared, np.pi + bandsign(coupler_ac_freq_iswap_if)), # 180 degree inversion
        )
        '''
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_iswap_if),
            phases          = 0.0,
            phases_q        = bandsign(coupler_ac_freq_iswap_if),
        )
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = np.abs(coupler_ac_freq_iswap_if),
            phases          = np.pi, # 180 degree inversion
            phases_q        = np.pi + bandsign(coupler_ac_freq_iswap_if), # 180 degree inversion
        )
        '''
        
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
        
        # For every phase value of the final pi-half gate:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases.
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            phase_C = reset_phase_counter(T, coupler_ac_port, 0, phases_declared, pls)
            
            # Put the system in its sought-for input state.
            if prepare_input_state[0] == '+':
                pls.output_pulse(T, control_pulse_pi_01_half_A)
            elif prepare_input_state[0] == '1':
                pls.output_pulse(T, control_pulse_pi_01_A)
            if prepare_input_state[1] == '+':
                pls.output_pulse(T, control_pulse_pi_01_half_B)
            elif prepare_input_state[1] == '1':
                pls.output_pulse(T, control_pulse_pi_01_B)
            T += control_duration_01
            
            # Track phases!
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Apply an iSWAP gate. Adjust the phase of the coupler drive too.
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            phase_C = add_virtual_z(T, phase_C, -phase_adjustment_after_iswap_C, coupler_ac_port, None, phases_declared, pls)
            
            # Track phases!
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Apply an iSWAP† gate.
            pls.output_pulse(T, coupler_ac_pulse_iswap_inverted)
            T += coupler_ac_duration_iswap
            ##phase_C = add_virtual_z(T, phase_C, -phase_adjustment_after_iswap_C, coupler_ac_port, None, phases_declared, pls)
            
            # Track phases!
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Sweep the phase of some qubit, in order
            # to get a Ramsey-like sequence.
            if analyse_iswap_phase_using_this_qubit == 'A':
                # Set the phase for the final π/2 gate at this point.
                phase_A = add_virtual_z(T, phase_A, control_phase_arr[ii] -phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
                pls.output_pulse(T, control_pulse_pi_01_half_A)
            else:
                # Set the phase for the final π/2 gate at this point.
                phase_B = add_virtual_z(T, phase_B, control_phase_arr[ii] -phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
                pls.output_pulse(T, control_pulse_pi_01_half_B)
            T += control_duration_01
            
            ## Track phases!
            ##phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            ##phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            ##phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            ## Moving the phase of the to-be-investigated π/2 pulse
            ## was done higher up, just before applying the final Vz-swept
            ## X gate.
            
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
            repeat_count    =   1,
            num_averages    =   num_averages,
            print_time      =   True,
        )
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
    
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
            'control_phase_arr', "rad",
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
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port_A', "",
            'control_freq_nco_A', "Hz",
            'control_freq_01_A', "Hz",
            'control_amp_01_A', "FS",
            'control_port_B', "",
            'control_freq_nco_B', "Hz",
            'control_freq_01_B', "Hz",
            'control_amp_01_B', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_iswap', "Hz",
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap', "s",
            
            'num_averages', "",
            'num_phases', "",
            
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            'phase_adjustment_after_iswap_A', "rad",
            'phase_adjustment_after_iswap_B', "rad",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_phases,
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'tune_local_accumulated_phase',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def iswap_tune_coupler_drive_phase(
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
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap,
    coupler_ac_amp_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_iswap_A  = 0.0,
    phase_adjustment_after_iswap_B  = 0.0,
    phase_adjustment_before_iswap_C = 0.0,
    phase_adjustment_after_iswap_C  = 0.0,
    
    analyse_iswap_phase_using_this_qubit = 'A',
    
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
    ''' Tune the phase of the iSWAP coupler drive ("AC flux driveline")
        that applies an iSWAP gate between two pairwise-coupled qubits.
        Based on the methods from:
        
        M. Ganzhorn et al. PR Research 2 033447 (2020)
        https://doi.org/10.48550/arXiv.2005.05696
        
        ... with theoretical assistance found in:
        
        Deanna M. Abrams et al. Nature electronics, 3, December 2020 pp.744–750
        https://doi.org/10.48550/arXiv.1912.04424
        
        Method: perform a cross-Ramsey experiment between two qubits,
        using an iSWAP gate to transfer population from qubit A to B.
        
        > The input state is |++⟩.
        > Then, an iSWAP is applied onto the connecting tunable coupler.
          For every new datapoint, the phase of the coupler's drive is swept
          to a different number (in radians).
        > Finally, one qubit gets a normal π/2-pulse whereas the other
          qubit gets a phase-shifted π/2-pulse.
        
        Ideally, the π/2 pulses are in fact X gates. The referred-to
        phase-shifted π/2-pulse, is in fact a Y pulse.
        
        analyse_iswap_phase_using_this_qubit defines which qubit will
        be moved to the Bloch sphere equator at the end of the experiment.
        The expected outcome is that this qubit, will have a small oscillation
        about the magnitude level defining 50 % population.
    '''
    
    ## Input sanitisation
    
    # Already here at the beginning, we may calculate a rough placeholder
    # for the necessary post-iSWAP frame correction on the coupler drive.
    # If, the local accumulated phase is known for both qubit A and qubit B.
    if ((phase_adjustment_after_iswap_C == 0.0) and \
        (phase_adjustment_after_iswap_A != 0.0) and \
        (phase_adjustment_after_iswap_B != 0.0)):
        # The phase adjustment on the coupler drive is unknown,
        # but the local accumulated phase of qubit A and qubit B are known.
        # Thus, calculate an estimate of the coupler drive phase adjustment.
        phase_adjustment_after_iswap_C = phase_adjustment_after_iswap_B - phase_adjustment_after_iswap_A
        print("Note: no post-iSWAP coupler drive phase adjustment was provided, but both qubits' local accumulated phases were provided. An estimate of the coupler drive phase adjustment was set: "+str(phase_adjustment_after_iswap_C)+" rad.")
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    ## The input state is already known to be |++⟩, there is no need
    ## to monitor the user-requested input state at this time.
    
    ## Initial array declaration
    
    # Declare what phases are available
    phases_declared = np.linspace(0, 2*np.pi, 512)
    
    # Declare array for the coupler phase that is swept, and make it legal.
    coupler_phase_arr = np.linspace(phase_sweep_rad_min, phase_sweep_rad_max, num_phases)
    coupler_phase_arr = legalise_phase_array( coupler_phase_arr, phases_declared )
    num_phases = len(coupler_phase_arr)
    
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port mixer
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = True,
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_iswap,
        )
        
        ### Setup readout pulses ###
        
        # Setup readout pulse envelope
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        
        ### Setup pulses "control_pulse_pi_01_A" and "control_pulse_pi_01_B" ###
        
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
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        control_pulse_pi_01_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_pulse_pi_01_half_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tones, considering that there are
        # digital mixers.
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_A)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_A))
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_B)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_B))
        )
        
        
        ### Setup the iSWAP gate pulses
        
        # There will be a "normal" iSWAP pulse, followed by an iSWAP†
        coupler_ac_duration_iswap = \
            coupler_ac_plateau_duration_iswap + \
            2 * coupler_ac_single_edge_time_iswap
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        ## Setup the iSWAP pulse carrier.
        
        # Setup LUT
        coupler_ac_freq_iswap_if = coupler_ac_freq_nco - coupler_ac_freq_iswap
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.full_like(phases_declared, np.abs(coupler_ac_freq_iswap_if)),
            phases          = phases_declared,
            phases_q        = phases_declared + np.full_like(phases_declared, bandsign(coupler_ac_freq_iswap_if))
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
        
        # For every phase value of the coupler drive:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            phase_C = reset_phase_counter(T, coupler_ac_port, 0, phases_declared, pls)
            
            # Put the system into state |++> using pi01_half pulses.
            pls.output_pulse(T, [control_pulse_pi_01_half_A, control_pulse_pi_01_half_B])
            ##pls.output_pulse(T, [control_pulse_pi_01_B])
            T += control_duration_01
            
            # Track phases!
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Apply virtual-Z to the coupler drive, a priori.
            phase_C = add_virtual_z(T, phase_C, coupler_phase_arr[ii] -phase_adjustment_before_iswap_C, coupler_ac_port, 0, phases_declared, pls)
            
            # Apply coupler drive phase-swept iSWAP gate. Also, track phases!
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Adjust the local frames of the qubits,
            # as well as the coupler drive.
            ## Here, we must also add the virtual-Z for the final
            ## "Y" gate, that will take one qubit to a different place
            ## than the other one.
            if analyse_iswap_phase_using_this_qubit == 'A':
                # Add +π/2 rad to qubit A.
                phase_A = add_virtual_z(T, phase_A, +np.pi/2 -phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
                phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
            else:
                # Add +π/2 rad to qubit B.
                phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
                phase_B = add_virtual_z(T, phase_B, +np.pi/2 -phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
            print("TODO: Phase correction after iSWAP disabled for now.")
            '''phase_C = add_virtual_z(T, phase_C, -phase_adjustment_after_iswap_C, coupler_ac_port, 0, phases_declared, pls)'''
            
            # Apply final round of π/2 gates.
            # One is +π/2 rad out-of-phase with the other one.
            pls.output_pulse(T, control_pulse_pi_01_half_A)
            pls.output_pulse(T, control_pulse_pi_01_half_B)
            T += control_duration_01
            
            ## # Track phases!
            ## phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            ## phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            ## phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Commence multiplexed readout
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
        
        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   1,
            num_averages    =   num_averages,
            print_time      =   True,
        )
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
    
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
            'coupler_phase_arr', "rad",
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
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port_A', "",
            'control_freq_nco_A', "Hz",
            'control_freq_01_A', "Hz",
            'control_amp_01_A', "FS",
            'control_port_B', "",
            'control_freq_nco_B', "Hz",
            'control_freq_01_B', "Hz",
            'control_amp_01_B', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_freq_iswap', "Hz",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap', "s",
            
            'num_averages', "",
            
            'num_phases', "",
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            
            'phase_adjustment_after_iswap_A', "rad",
            'phase_adjustment_after_iswap_B', "rad",
            'phase_adjustment_before_iswap_C', "rad",
            'phase_adjustment_after_iswap_C', "rad",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_phases,
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'unconditional_cross_Ramsey',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def iswap_conditional_cross_ramsey(
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
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_freq_nco,
    coupler_ac_freq_iswap,
    coupler_ac_amp_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_iswap_A  = 0.0,
    phase_adjustment_after_iswap_B  = 0.0,
    phase_adjustment_before_iswap_C = 0.0,
    phase_adjustment_after_iswap_C  = 0.0,
    
    prepare_input_state = '0+',
    
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
    ''' Perform a cross-Ramsey experiment between two qubits,
        using an iSWAP gate to transfer population between qubit A and B.
        The experiment shows to what extent your iSWAP is phase coherent.
        
        > The input state is |0+⟩ or |1+⟩.
            → The roles of qubits A and B can also be exchanged.
        > Apply the iSWAP to be characterised.
        > Finally, apply a final π/2 pulse onto the qubit that originally
          was *not* prepared in |+⟩.
        
        The expected outcome would be two sine curves that are π rad
        out-of-phase with eachother.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    ## Input sanitisation
    
    # Already here at the beginning, we may calculate a rough placeholder
    # for the necessary post-iSWAP frame correction on the coupler drive.
    # If, the local accumulated phase is known for both qubit A and qubit B.
    if ((phase_adjustment_after_iswap_C == 0.0) and \
        (phase_adjustment_after_iswap_A != 0.0) and \
        (phase_adjustment_after_iswap_B != 0.0)):
        # The phase adjustment on the coupler drive is unknown,
        # but the local accumulated phase of qubit A and qubit B are known.
        # Thus, calculate an estimate of the coupler drive phase adjustment.
        phase_adjustment_after_iswap_C = phase_adjustment_after_iswap_B - phase_adjustment_after_iswap_A
        print("Note: no post-iSWAP coupler drive phase adjustment was provided, but both qubits' local accumulated phases were provided. An estimate of the coupler drive phase adjustment was set: "+str(phase_adjustment_after_iswap_C)+" rad.")
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    ## Assure that the requested input state to prepare, is valid.
    assert ( \
        (prepare_input_state == '0+') or \
        (prepare_input_state == '+0') or \
        (prepare_input_state == '1+') or \
        (prepare_input_state == '+1') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'0+\', \'+0\', \'1+\' and \'+1\'"
    
    ## Initial array declaration
    
    # Declare what phases are available
    phases_declared = np.linspace(0, 2*np.pi, 512)
    
    # Declare array for the coupler phase that is swept, and make it legal.
    control_phase_arr = np.linspace(phase_sweep_rad_min, phase_sweep_rad_max, num_phases)
    control_phase_arr = legalise_phase_array( control_phase_arr, phases_declared )
    num_phases = len(control_phase_arr)
    
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port mixer
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            tune      = True,
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = True,
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
        # Control port amplitudes
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
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp_iswap,
        )
        
        ### Setup readout pulses ###
        
        # Setup readout pulse envelope
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
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
        
        ### Setup pulses "control_pulse_pi_01_A" and "control_pulse_pi_01_B" ###
        
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
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        control_pulse_pi_01_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_pulse_pi_01_half_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tones, considering that there are
        # digital mixers.
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_A)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_A))
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_B)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_B))
        )
        
        
        ### Setup the iSWAP gate pulses
        
        # There will be a "normal" iSWAP pulse, followed by an iSWAP†
        coupler_ac_duration_iswap = \
            coupler_ac_plateau_duration_iswap + \
            2 * coupler_ac_single_edge_time_iswap
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        ## Setup the iSWAP pulse carrier.
        
        # Setup LUT
        coupler_ac_freq_iswap_if = coupler_ac_freq_nco - coupler_ac_freq_iswap
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.full_like(phases_declared, np.abs(coupler_ac_freq_iswap_if)),
            phases          = phases_declared,
            phases_q        = phases_declared + np.full_like(phases_declared, bandsign(coupler_ac_freq_iswap_if))
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
        
        # For every phase value of the coupler drive:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            phase_C = reset_phase_counter(T, coupler_ac_port, 0, phases_declared, pls)
            
            # Prepare the sought-for input state.
            # There are four options: |0+>, |+0>, |1+>, |+1>
            if prepare_input_state[-1] == '+':
                
                # Qubit B is initially prepared in the mixed state.
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                
                # Is the conditional qubit supposed to be on?
                if prepare_input_state[0] == '1':
                    # It is supposed to be on!
                    pls.output_pulse(T, control_pulse_pi_01_A)
                
                # Go to the next quantum circuit moment.
                T += control_duration_01
                
            else:
                
                # Qubit B is NOT initially prepared in the mixed state,
                # meaning that qubit A is prepared in the mixed state.
                pls.output_pulse(T, control_pulse_pi_01_half_A)
                
                # Is the conditional qubit supposed to be on?
                if prepare_input_state[1] == '1':
                    # It is supposed to be on!
                    pls.output_pulse(T, control_pulse_pi_01_B)
                
                # Go to the next quantum circuit moment.
                T += control_duration_01
            
            # Track phases!
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Apply virtual-Z to the coupler drive, a priori.
            phase_C = add_virtual_z(T, phase_C, -phase_adjustment_before_iswap_C, coupler_ac_port, 0, phases_declared, pls)
            
            # Apply iSWAP, track phases.
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Add the local phase correction following the iSWAP gate,
            # but also make sure to sweep the phase of the qubit which
            # was *not* originally prepared in |+>
            if prepare_input_state[-1] == '+':
                phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_iswap_A + control_phase_arr[ii], control_port_A, 0, phases_declared, pls)
                phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
            else:
                phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
                phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_iswap_B + control_phase_arr[ii], control_port_B, 0, phases_declared, pls)
            phase_C = add_virtual_z(T, phase_C, -phase_adjustment_after_iswap_C, coupler_ac_port, 0, phases_declared, pls)
            
            # Apply pi/2 gate on the qubit *not* prepared in |+>
            if prepare_input_state[-1] == '+':
                pls.output_pulse(T, control_pulse_pi_01_half_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_half_B)
            T += control_duration_01
            
            ## # Track phases!
            ## phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            ## phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            ## phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Commence multiplexed readout
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
        
        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   1,
            num_averages    =   num_averages,
            print_time      =   True,
        )
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
        
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
            'control_phase_arr', "rad",
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
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port_A', "",
            'control_freq_nco_A', "Hz",
            'control_freq_01_A', "Hz",
            'control_amp_01_A', "FS",
            'control_port_B', "",
            'control_freq_nco_B', "Hz",
            'control_freq_01_B', "Hz",
            'control_amp_01_B', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_freq_iswap', "Hz",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap', "s",
            
            'num_averages', "",
            
            'num_phases', "",
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            
            'phase_adjustment_after_iswap_A', "rad",
            'phase_adjustment_after_iswap_B', "rad",
            'phase_adjustment_before_iswap_C', "rad",
            'phase_adjustment_after_iswap_C', "rad",
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
                'fetched_data_arr_1', "FS",
                'fetched_data_arr_2', "FS",
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
            resonator_freq_if_arrays_to_fft = [ readout_freq_if_A, readout_freq_if_B ],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_phases,
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'conditional_cross_Ramsey',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return
 