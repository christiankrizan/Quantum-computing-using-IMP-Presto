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
    check_if_integration_window_is_legal,\
    show_user_time_remaining, \
    get_timestamp_string
from data_exporter import \
    ensure_all_keyed_elements_even, \
    export_processed_data_to_file, \
    stylise_axes, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save
from fit_phase import \
    fit_phase_offset, \
    fit_conditional_phase_offset

def cz20_sweep_amplitude_and_detuning_for_t_half(
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
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    coupler_ac_freq_nco,
    coupler_ac_freq_cz20_centre,
    coupler_ac_freq_cz20_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
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
    ''' Tune a CZ20-interaction between two qubits using
        a tuneable coupler.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    # The user provides the full gate time for the CZ₂₀ gate.
    # Here, we cut that time in 2, for the measurement ahead.
    coupler_ac_plateau_duration_cz20 = ((coupler_ac_plateau_duration_cz20 + 2*coupler_ac_single_edge_time_cz20) / 2) - (2*coupler_ac_single_edge_time_cz20)
    
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
    if ((num_freqs == 1) and (coupler_ac_freq_cz20_span != 0.0)):
        print("Note: single control frequency point requested, ignoring span parameter.")
        coupler_ac_freq_cz20_span = 0
    
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
        
        # Coupler port(s)
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
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
            phases_q     = bandsign(readout_freq_if_B)
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
        
        
        ### Setup the CZ20 gate pulse
        
        # The initially set duration will not be swept by the sequencer
        # program.
        coupler_ac_duration_cz20 = \
            2 * coupler_ac_single_edge_time_cz20 + \
            coupler_ac_plateau_duration_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        
        # Setup the CZ20 pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_cz20_centre_if = coupler_ac_freq_nco - coupler_ac_freq_cz20_centre  
        f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
        f_stop  = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
        coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_cz20_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_cz20_if_arr),
            phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, bandsign(coupler_ac_freq_cz20_centre_if)),
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
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Put the system into state |11>
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
            T += control_duration_01
            ## TODO add Vz to track phase?
            
            # Apply the CZ20 gate, with parameters being swept.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            ## TODO add Vz to track phase?
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to the next scanned frequency
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
            'coupler_ac_pulse_cz20_freq_arr', "Hz",
            'coupler_ac_amp_arr', "FS",
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
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_cz20_centre', "Hz",
            'coupler_ac_freq_cz20_span', "Hz",
            
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
            append_to_log_name_before_timestamp = '20_sweep_amplitude_and_detuning',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def cz20_sweep_amplitude_and_detuning_for_t_half_state_probability(
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
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    coupler_ac_freq_nco,
    coupler_ac_freq_cz20_centre,
    coupler_ac_freq_cz20_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['20'],
    
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
    ''' Tune a CZ20-interaction between two qubits using
        a tuneable coupler.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
        
        The fetched result will be sent into a state discriminator.
    '''
    
    # The user provides the full gate time for the CZ₂₀ gate.
    # Here, we cut that time in 2, for the measurement ahead.
    coupler_ac_plateau_duration_cz20 = ((coupler_ac_plateau_duration_cz20 + 2*coupler_ac_single_edge_time_cz20) / 2) - (2*coupler_ac_single_edge_time_cz20)
    
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
    if ((num_freqs == 1) and (coupler_ac_freq_cz20_span != 0.0)):
        print("Note: single control frequency point requested, ignoring span parameter.")
        coupler_ac_freq_cz20_span = 0
    
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
        
        # Coupler port(s)
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
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
            phases_q     = bandsign(readout_freq_if_B)
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
        
        
        ### Setup the CZ20 gate pulse
        
        # The initially set duration will not be swept by the sequencer
        # program.
        coupler_ac_duration_cz20 = \
            2 * coupler_ac_single_edge_time_cz20 + \
            coupler_ac_plateau_duration_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        
        # Setup the CZ20 pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_cz20_centre_if = coupler_ac_freq_nco - coupler_ac_freq_cz20_centre  
        f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
        f_stop  = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
        coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_cz20_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_cz20_if_arr),
            phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, bandsign(coupler_ac_freq_cz20_centre_if)),
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
        
        # For every AC coupler amplitude to sweep over:
        for jj in range(num_amplitudes):
        
            # For every frequency to sweep over:
            for ii in range(num_freqs):
                
                # Get a time reference, used for gauging the iteration length.
                T_begin = T
                
                # Put the system into state |11>
                pls.reset_phase(T, [control_port_A, control_port_B])
                pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
                T += control_duration_01
                ## TODO add Vz to track phase?
                
                # Apply the CZ20 gate, with parameters being swept.
                pls.reset_phase(T, coupler_ac_port)
                pls.output_pulse(T, coupler_ac_pulse_cz20)
                T += coupler_ac_duration_cz20
                ## TODO add Vz to track phase?
                
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
            repeat_count    =   num_single_shots,
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
            'coupler_ac_pulse_cz20_freq_arr', "Hz",
            'coupler_ac_amp_arr', "FS",
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
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_cz20_centre', "Hz",
            'coupler_ac_freq_cz20_span', "Hz",
            
            'coupler_ac_amp_min', "FS",
            'coupler_ac_amp_max', "FS",
            
            'num_freqs', "",
            'num_averages', "",
            'num_amplitudes', "",
            
            'num_single_shots', "",
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
            
            single_shot_repeats_to_discretise = num_single_shots,
            ordered_resonator_ids_in_readout_data = resonator_ids,
            get_probabilities_on_these_states = states_to_discriminate_between,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '20_sweep_amplitude_and_detuning_state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def cz20_sweep_duration_and_detuning(
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
    coupler_ac_freq_cz20_centre,
    coupler_ac_freq_cz20_span,
    coupler_ac_amp,
    
    num_freqs,
    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20_min,
    coupler_ac_plateau_duration_cz20_max,
    
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
    ''' Tune the amplitude and duration of a CZ₂₀ interaction,
        with a set gate amplitude and applied magnetostatic coupler flux.
        
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
    if ((num_freqs == 1) and (coupler_ac_freq_cz20_span != 0.0)):
        print("Note: single control frequency point requested, ignoring span parameter.")
        coupler_ac_freq_cz20_span = 0
    
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
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20_min = int(round(coupler_ac_plateau_duration_cz20_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20_max = int(round(coupler_ac_plateau_duration_cz20_max / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        # Generate an array for data storage. For all elements, round to the
        # programmable logic clock period. Then, remove duplicates and update
        # the num_time_steps parameter.
        cz20_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_cz20_min + 2 * coupler_ac_single_edge_time_cz20, \
            coupler_ac_plateau_duration_cz20_max + 2 * coupler_ac_single_edge_time_cz20, \
            num_time_steps
        )
        for jj in range(len(cz20_total_pulse_duration_arr)):
            cz20_total_pulse_duration_arr[jj] = int(round(cz20_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the array.
        cz20_total_pulse_duration_arr = np.unique( np.array(cz20_total_pulse_duration_arr) )
        num_time_steps = len(cz20_total_pulse_duration_arr)
        
        
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
            output_ports    = control_port_A,
            group           = 1,
            scales          = control_amp_12_A,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 0,
            scales          = control_amp_01_B,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 1,
            scales          = control_amp_12_B,
        )
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp, # This value will be swept!
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
            phases_q     = bandsign(readout_freq_if_B)
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
        
        # Setup control pulse carrier tones, considering that there are digital mixers
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
        
        
        ### Setup the CZ20 gate pulse
        
        # The initially set duration is temporary, and will be swept by the
        # sequencer program.
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_plateau_duration_cz20_min + \
                          2 * coupler_ac_single_edge_time_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20
        )
        
        # Setup the cz20 pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_cz20_centre_if = coupler_ac_freq_nco - coupler_ac_freq_cz20_centre  
        f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
        f_stop  = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
        coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_cz20_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_cz20_if_arr),
            phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, bandsign(coupler_ac_freq_cz20_centre_if)),
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
        
        # For every pulse duration to sweep over:
        for ii in range(num_time_steps):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Redefine the CZ20 pulse's total duration,
            # resulting in stepping said duration in time.
            coupler_ac_duration_cz20 = cz20_total_pulse_duration_arr[ii]
            coupler_ac_pulse_cz20.set_total_duration(coupler_ac_duration_cz20)
            
            # Put the system into state |11>
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
            T += control_duration_01
            ## TODO add Vz to track phase?
            
            # Apply the CZ20 gate, with parameters being swept.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            ## TODO add Vz to track phase?
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Is this the last iteration?
            if ii == num_delays-1:
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
            'cz20_total_pulse_duration_arr',"s",
            'coupler_ac_pulse_cz20_freq_arr', "Hz",
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
            'coupler_ac_amp', "FS",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_cz20_centre', "Hz",
            'coupler_ac_freq_cz20_span', "Hz",
            
            'num_freqs', "",
            'num_averages', "",
            
            'num_time_steps', "",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20_min', "s",
            'coupler_ac_plateau_duration_cz20_max', "s",
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
            append_to_log_name_before_timestamp = '20_duration_vs_frequency',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def cz20_sweep_duration_and_detuning_state_probability(
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
    coupler_ac_freq_cz20_centre,
    coupler_ac_freq_cz20_span,
    coupler_ac_amp,
    
    num_freqs,
    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20_min,
    coupler_ac_plateau_duration_cz20_max,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['20'],
    
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
    ''' Tune the amplitude and duration of a CZ₂₀ interaction,
        with a set gate amplitude and applied magnetostatic coupler flux.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
        
        The fetched result will be sent into a state discriminator.
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
    if ((num_freqs == 1) and (coupler_ac_freq_cz20_span != 0.0)):
        print("Note: single control frequency point requested, ignoring span parameter.")
        coupler_ac_freq_cz20_span = 0
    
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
        repetition_delay = int(round(repetition_delay / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        control_duration_12 = int(round(control_duration_12 / plo_clk_T)) * plo_clk_T
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20_min = int(round(coupler_ac_plateau_duration_cz20_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20_max = int(round(coupler_ac_plateau_duration_cz20_max / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        # Generate an array for data storage. For all elements, round to the
        # programmable logic clock period. Then, remove duplicates and update
        # the num_time_steps parameter.
        cz20_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_cz20_min + 2 * coupler_ac_single_edge_time_cz20, \
            coupler_ac_plateau_duration_cz20_max + 2 * coupler_ac_single_edge_time_cz20, \
            num_time_steps
        )
        for jj in range(len(cz20_total_pulse_duration_arr)):
            cz20_total_pulse_duration_arr[jj] = int(round(cz20_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the array.
        cz20_total_pulse_duration_arr = np.unique( np.array(cz20_total_pulse_duration_arr) )
        num_time_steps = len(cz20_total_pulse_duration_arr)
        
        
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
            output_ports    = control_port_A,
            group           = 1,
            scales          = control_amp_12_A,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 0,
            scales          = control_amp_01_B,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 1,
            scales          = control_amp_12_B,
        )
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            scales          = coupler_ac_amp, # This value will be swept!
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
            phases_q     = bandsign(readout_freq_if_B)
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
        
        # Setup control pulse carrier tones, considering that there are digital mixers
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
        
        
        ### Setup the CZ20 gate pulse
        
        # The initially set duration is temporary, and will be swept by the
        # sequencer program.
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_plateau_duration_cz20_min + \
                          2 * coupler_ac_single_edge_time_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20
        )
        
        # Setup the cz20 pulse carrier, this tone will be swept in frequency.
        coupler_ac_freq_cz20_centre_if = coupler_ac_freq_nco - coupler_ac_freq_cz20_centre  
        f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
        f_stop  = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
        coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate side band.
        coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_nco - coupler_ac_freq_cz20_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_cz20_if_arr),
            phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, bandsign(coupler_ac_freq_cz20_centre_if)),
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
        
        # For every AC coupler frequency to scan:
        for jj in range(num_freqs):
        
            # For every pulse duration to sweep over:
            for ii in range(num_time_steps):
                
                # Get a time reference, used for gauging the iteration length.
                T_begin = T
                
                # Redefine the CZ20 pulse's total duration,
                # resulting in stepping said duration in time.
                coupler_ac_duration_cz20 = cz20_total_pulse_duration_arr[ii]
                coupler_ac_pulse_cz20.set_total_duration(coupler_ac_duration_cz20)
                
                # Put the system into state |11>
                pls.reset_phase(T, [control_port_A, control_port_B])
                pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
                T += control_duration_01
                ## TODO add Vz to track phase?
                
                # Apply the CZ20 gate, with parameters being swept.
                pls.reset_phase(T, coupler_ac_port)
                pls.output_pulse(T, coupler_ac_pulse_cz20)
                T += coupler_ac_duration_cz20
                ## TODO add Vz to track phase?
                
                # Commence multiplexed readout
                pls.reset_phase(T, readout_stimulus_port)
                pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                pls.store(T + readout_sampling_delay) # Sampling window
                T += readout_duration
                
                # Is this the last iteration?
                if ii == num_delays-1:
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
            repeat_count    =   num_single_shots,
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
            'cz20_total_pulse_duration_arr',"s",
            'coupler_ac_pulse_cz20_freq_arr', "Hz",
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
            'coupler_ac_amp', "FS",
            'coupler_ac_freq_nco', "Hz",
            'coupler_ac_freq_cz20_centre', "Hz",
            'coupler_ac_freq_cz20_span', "Hz",
            
            'num_freqs', "",
            'num_averages', "",
            
            'num_time_steps', "",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20_min', "s",
            'coupler_ac_plateau_duration_cz20_max', "s",
            
            'num_single_shots', "",
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
            append_to_log_name_before_timestamp = '20_duration_vs_frequency_state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def cz20_tune_local_accumulated_phase(
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
    coupler_ac_freq_cz20,
    coupler_ac_amp_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_cz20_A = 0.0,
    phase_adjustment_after_cz20_B = 0.0,
    
    prepare_input_state = '+0',
    
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
    ''' This method of tuning the local accumulated phase of qubit A (or B)
        follows closely that of Fig. (C3) in M. Ganzhorn et al.
        PR Research 2 033447 (2020), https://doi.org/10.48550/arXiv.2005.05696
        
        1. Prepare |+0> (or |0+>).
        2. Run CZ₂₀, with some compensating phase phi added onto the qubit
           which was prepared in |+>.
        3. Run CZ₂₀†, yet again with some compensating phase phi added onto
           the qubit which was originally prepared in |+>
        4. Run final pi/2 pulse on the qubit originally prepared in |+>
        
        The output is a Ramsey-type oscillation, showing how much phase is
        accumulated onto qubit A (or B) during the CZ₂₀ gate execution.
        This accumulated phase stems from the dispersive shifting of qubit A
        (and B) when the tunable coupler moves due to the AC waveform
        applied onto it for t_CZ₂₀ seconds.
        
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
    assert ( (prepare_input_state == '+0') or (prepare_input_state == '0+') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'+0\' and \'0+\'"
    
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
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
            scales          = coupler_ac_amp_cz20,
        )
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            scales          = coupler_ac_amp_cz20,
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
        
        ### Setup pulses "control_pulse_pi_01_A" and "control_pulse_pi_01_B" ###
        
        # Setup control_pulse_pi_01_A and _B pulse envelopes.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
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
        
        
        ## Set up the CZ20 gate pulse
        
        # In the actual experiment, there will be a "normal" CZ20 pulse
        # followed by a CZ20†.
        coupler_ac_duration_cz20 = \
            coupler_ac_plateau_duration_cz20 + \
            2 * coupler_ac_single_edge_time_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        coupler_ac_pulse_cz20_inverted = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 1,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        
        ## Setup the CZ20 pulse carrier.
        
        # Setup LUT
        coupler_ac_freq_cz20_if = coupler_ac_freq_nco - coupler_ac_freq_cz20
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_cz20_if),
            phases          = 0.0,
            phases_q        = bandsign(coupler_ac_freq_cz20_if),
        )
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = np.abs(coupler_ac_freq_cz20_if),
            phases          = np.pi, # 180 degree inversion
            phases_q        = np.pi + bandsign(coupler_ac_freq_cz20_if), # 180 degree inversion
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
        
        # For every phase value of the final pi-half gate:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            
            # Put the system into state |+0> or |0+> with pi01_half pulses.
            if prepare_input_state == '+0':
                pls.output_pulse(T, control_pulse_pi_01_half_A)
                T += control_duration_01
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                T += control_duration_01
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Apply CZ20 gate.
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            ## phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            ## TODO Track phase of coupler or not?
            ## TODO Track phase of qubits or not? Here, I am tracking that phase.
            if prepare_input_state == '+0':
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Adjust the local accumulated phase of the qubit which
            # was originally prepared in |+>. Note that there is no
            # pulse played back, thus no real time duration spent.
            # Meaning in turn, no need to track some phase using track_phase.
            if prepare_input_state == '+0':
                phase_A = add_virtual_z(T, phase_A, control_phase_arr[ii] -phase_adjustment_after_cz20_A, control_port_A, 0, phases_declared, pls)
            else:
                phase_B = add_virtual_z(T, phase_B, control_phase_arr[ii] -phase_adjustment_after_cz20_B, control_port_B, 0, phases_declared, pls)
            
            # Apply CZ20† gate.
            pls.output_pulse(T, coupler_ac_pulse_cz20_inverted)
            T += coupler_ac_duration_cz20
            if prepare_input_state == '+0':
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Adjust the local accumulated phase of the qubit again.
            # The target is the same qubit as was phase-adjusted before.
            # Then, apply the final pi/2 gate on the adjusted qubit.
            if prepare_input_state == '+0':
                # Set the phase for the final π/2 gate at this point.
                phase_A = add_virtual_z(T, phase_A, control_phase_arr[ii] -phase_adjustment_after_cz20_A, control_port_A, 0, phases_declared, pls)
                pls.output_pulse(T, control_pulse_pi_01_half_A)
                T += control_duration_01
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                # Set the phase for the final π/2 gate at this point.
                phase_B = add_virtual_z(T, phase_B, control_phase_arr[ii] -phase_adjustment_after_cz20_B, control_port_B, 0, phases_declared, pls)
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                T += control_duration_01
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
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
            'coupler_ac_freq_cz20', "Hz",
            'coupler_ac_amp_cz20', "FS",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            
            'num_averages', "",
            'num_phases', "",
            
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            'phase_adjustment_after_cz20_A', "rad",
            'phase_adjustment_after_cz20_B', "rad",
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
            append_to_log_name_before_timestamp = '20_tune_local_accumulated_phase',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def cz20_tune_local_accumulated_phase_state_probability(
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
    coupler_ac_freq_cz20,
    coupler_ac_amp_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    
    num_averages,
    num_phases,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['10'],
    
    prepare_input_state = '+0',
    
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_cz20_A = 0.0,
    phase_adjustment_after_cz20_B = 0.0,
    
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
    ''' This method of tuning the local accumulated phase of qubit A (or B)
        follows closely that of Fig. (C3) in M. Ganzhorn et al.
        PR Research 2 033447 (2020), https://doi.org/10.48550/arXiv.2005.05696
        
        1. Prepare |+0> (or |0+>).
        2. Run CZ₂₀, with some compensating phase phi added onto the qubit
           which was prepared in |+>.
        3. Run CZ₂₀†, yet again with some compensating phase phi added onto
           the qubit which was originally prepared in |+>
        4. Run final pi/2 pulse on the qubit originally prepared in |+>
        
        The output is a Ramsey-type oscillation, showing how much phase is
        accumulated onto qubit A (or B) during the CZ₂₀ gate execution.
        This accumulated phase stems from the dispersive shifting of qubit A
        (and B) when the tunable coupler moves due to the AC waveform
        applied onto it for t_CZ₂₀ seconds.
        
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
    assert ( (prepare_input_state == '+0') or (prepare_input_state == '0+') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'+0\' and \'0+\'"
    
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
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
            scales          = coupler_ac_amp_cz20,
        )
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            scales          = coupler_ac_amp_cz20,
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
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
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
        
        
        ## Set up the CZ20 gate pulse
        
        # In the actual experiment, there will be a "normal" CZ20 pulse
        # followed by a CZ20†.
        coupler_ac_duration_cz20 = \
            coupler_ac_plateau_duration_cz20 + \
            2 * coupler_ac_single_edge_time_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        coupler_ac_pulse_cz20_inverted = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 1,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        
        
        ## Setup the CZ20 pulse carrier.
        
        # Setup LUT
        coupler_ac_freq_cz20_if = coupler_ac_freq_nco - coupler_ac_freq_cz20
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_cz20_if),
            phases          = 0.0,
            phases_q        = bandsign(coupler_ac_freq_cz20_if),
        )
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = np.abs(coupler_ac_freq_cz20_if),
            phases          = np.pi, # 180 degree inversion
            phases_q        = np.pi + bandsign(coupler_ac_freq_cz20_if), # 180 degree inversion
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
        
        # For every phase value of the final pi-half gate:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            ## TODO track phase on the coupler AC port?
            raise NotImplementedError("TODO This function should be modified to include phase tracking on the coupler port too.")
            
            # Put the system into state |+0> or |0+> with pi01_half pulses.
            if prepare_input_state == '+0':
                pls.output_pulse(T, control_pulse_pi_01_half_A)
                T += control_duration_01
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                T += control_duration_01
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Apply CZ20 gate.
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            ## phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            ## TODO Track phase of coupler or not?
            ## TODO Track phase of qubits or not? Here, I am tracking that phase.
            if prepare_input_state == '+0':
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Adjust the local accumulated phase of the qubit which
            # was originally prepared in |+>. Note that there is no
            # pulse played back, thus no real time duration spent.
            # Meaning in turn, no need to track some phase using track_phase.
            if prepare_input_state == '+0':
                phase_A = add_virtual_z(T, phase_A, control_phase_arr[ii] -phase_adjustment_after_cz20_A, control_port_A, 0, phases_declared, pls)
            else:
                phase_B = add_virtual_z(T, phase_B, control_phase_arr[ii] -phase_adjustment_after_cz20_B, control_port_B, 0, phases_declared, pls)
            
            # Apply CZ20† gate.
            pls.output_pulse(T, coupler_ac_pulse_cz20_inverted)
            T += coupler_ac_duration_cz20
            ## phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            ## TODO Track phase of coupler or not?
            ## TODO Track phase of qubits or not? Here, I am tracking that phase.
            if prepare_input_state == '+0':
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Adjust the local accumulated phase of the qubit again.
            # The target is the same qubit as was phase-adjusted before.
            # Then, apply the final pi/2 gate on the adjusted qubit.
            if prepare_input_state == '+0':
                # Set the phase for the final π/2 gate at this point.
                phase_A = add_virtual_z(T, phase_A, control_phase_arr[ii] -phase_adjustment_after_cz20_A, control_port_A, 0, phases_declared, pls)
                pls.output_pulse(T, control_pulse_pi_01_half_A)
                T += control_duration_01
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            else:
                # Set the phase for the final π/2 gate at this point.
                phase_B = add_virtual_z(T, phase_B, control_phase_arr[ii] -phase_adjustment_after_cz20_B, control_port_B, 0, phases_declared, pls)
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                T += control_duration_01
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
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
            repeat_count    =   num_single_shots,
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
            'coupler_ac_freq_cz20', "Hz",
            'coupler_ac_amp_cz20', "FS",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            
            'num_averages', "",
            'num_phases', "",
            
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            'phase_adjustment_after_cz20_A', "rad",
            'phase_adjustment_after_cz20_B', "rad",
            
            'num_single_shots', "",
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
            
            single_shot_repeats_to_discretise = num_single_shots,
            ordered_resonator_ids_in_readout_data = resonator_ids,
            get_probabilities_on_these_states = states_to_discriminate_between,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '20_tune_local_accumulated_phase_state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return

def cz20_conditional_cross_ramsey(
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
    coupler_ac_freq_cz20,
    coupler_ac_amp_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_cz20_A = 0.0,
    phase_adjustment_after_cz20_B = 0.0,
    
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
    ''' Create a Ramsey-styled interferometry while (not) transferring phase
        information from another qubit.
        
        Procedure:
        - Qubits A and B are prepared in state |0+⟩ (or |+0⟩).
        - Run CZ₂₀ gate.
            Since one qubit is in |0⟩, there is no added π rad phase
            onto the other qubit.
        - On the qubit originally prepared in |+⟩, run a final π/2 gate.
        - The expected result is a non-decaying Ramsey interferometry.
        
        Now, repeat the procedure, but preparing input state |1+⟩ (or |+1⟩).
        →   Since one qubit is in |1⟩ once the CZ₂₀ gate executes,
            a phase shift of π rad will be added onto the phase-swept qubit.
        →   The end-result is a ~π rad phase-shifted Ramsey, in comparison to
            the previous experiment where one qubit was prepared in |0⟩.
        
        Technique shown in Fig. 3 in:
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.062408
        
        Also, shown in Fig. (C7) in M. Ganzhorn et al.
        PR Research 2 033447 (2020), https://doi.org/10.48550/arXiv.2005.05696
        
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
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
            scales          = coupler_ac_amp_cz20,
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
        
        ### Setup pulses:
        ### "control_pulse_pi_01_A" and "control_pulse_pi_01_B"
        ### "control_pulse_pi_01_A_half" and "control_pulse_pi_01_B_half"
        
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
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
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
        
        ## Set up the CZ20 gate pulse
        
        # Set up CZ20 envelope
        coupler_ac_duration_cz20 = \
            coupler_ac_plateau_duration_cz20 + \
            2 * coupler_ac_single_edge_time_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        # Setup the CZ20 pulse carrier.
        coupler_ac_freq_if_cz20 = coupler_ac_freq_nco - coupler_ac_freq_cz20
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_if_cz20),
            phases          = 0.0,
            phases_q        = bandsign(coupler_ac_freq_if_cz20),
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
        
        # For every phase value of the final pi-half gate:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            
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
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
                
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
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Apply CZ20 gate.
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
            # Add the local phase correction following the CZ₂₀ gate,
            # but also make sure to sweep the phase of the qubit which
            # was originally prepared in |+>
            if prepare_input_state[-1] == '+':
                phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_cz20_A, control_port_A, 0, phases_declared, pls)
                phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_cz20_B + control_phase_arr[ii], control_port_B, 0, phases_declared, pls)
            else:
                phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_cz20_A + control_phase_arr[ii], control_port_A, 0, phases_declared, pls)
                phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_cz20_B, control_port_B, 0, phases_declared, pls)
            
            # Apply pi/2 gate on the qubit prepared in |+>
            if prepare_input_state[-1] == '+':
                pls.output_pulse(T, control_pulse_pi_01_half_B)
            else:
                pls.output_pulse(T, control_pulse_pi_01_half_A)
            T += control_duration_01
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            
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
            'coupler_ac_freq_cz20', "Hz",
            'coupler_ac_amp_cz20', "FS",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            
            'num_averages', "",
            'num_phases', "",
            
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            'phase_adjustment_after_cz20_A', "rad",
            'phase_adjustment_after_cz20_B', "rad",
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

def cz20_tune_frequency_until_pi_phase(
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
    coupler_ac_freq_cz20_centre,
    coupler_ac_freq_cz20_span,
    coupler_ac_amp_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    
    num_freqs,
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    reset_dc_to_zero_when_finished = True,
    force_device_reboot_on_connection_error = False,
    
    conditional_qubit_is = 'A',
    
    save_complex_data = True,
    save_raw_time_data = False,
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    suppress_log_browser_export_of_suboptimal_data = True,
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
    ''' Tune the frequency of the CZ₂₀ gate until one qubit
        can conditionally infer a π phase shift on another qubit.
        
        Given a frequency sweep:
        - Tune the local accumulated phase of both qubits.
        - Execute a cross-Ramsey measurement.
        - Take the phase difference between the two qubits as a datapoint
            in a plot showing CZ₂₀ gate frequency versus phase difference.
        
        conditional_qubit_is denotes which of the two available qubits
        is the one which is flipped on/off for the CZ₂₀ gate to trigger.
        Legal arguments are either 'A' or 'B'.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    # For [list of frequencies]:
    #     > For [input state |+0> and |0+>]:
    #          Run tune_local_accumulated_phase of CZ₂₀.
    #          Fit the needed phase offset.
    #     > Run cross-Ramsey.
    #     > Perform cross-Ramsey fit to get the phase difference between the
    #       two sinusoidals.
    #     > Store the phase difference for the final plot.
    # Make final plot: CZ₂₀ frequency on X-axis, Phase difference on Y-axis.
    
    ## Input sanitisation
    
    # Check that the user has denoted a legal qubit as the conditional qubit
    # in the measurement.
    assert ((conditional_qubit_is.lower() == 'a') or  \
            (conditional_qubit_is.lower() == 'b')),   \
        "Error! Illegal input argument. The conditional qubit may only be " + \
        "either qubit 'A' or qubit 'B'."
    
    # Sanitisation for whether the user has a
    # span engaged but only a single frequency.
    if ((num_freqs == 1) and (coupler_ac_freq_cz20_span != 0.0)):
        print("Note: single control frequency point requested, ignoring span parameter.")
        coupler_ac_freq_cz20_span = 0
    
    # Prepare variables for keeping track of what time to print out.
    num_tick_tocks = 0
    total_dur = 0
    
    # Declare resonator frequency stepping array.
    cz20_freq_start = coupler_ac_freq_cz20_centre - coupler_ac_freq_cz20_span/2
    cz20_freq_stop  = coupler_ac_freq_cz20_centre + coupler_ac_freq_cz20_span/2
    cz20_freq_arr   = np.linspace(cz20_freq_start, cz20_freq_stop, num_freqs)
    
    # Declare list that will be appended to, containing how much phase
    # was actually added by the alleged CZ₂₀ gate.
    cz20_added_phase_arr = []
    
    # As we go along, we will eventually find the best values.
    # Make some variables that we can update once we stumble into
    # better values.
    optimal_cz20_frequency = 0.0
    optimal_phase_adjustment_after_cz20_A = 0.0
    optimal_phase_adjustment_after_cz20_B = 0.0
    absolute_phase_offset_from_cz20_gate = 0.0
    
    # Initialise the added_phase_of_optimal_cz20_gate variable
    # to something really large. It will be used later to gauge which
    # frequency was the one that yielded the closest value to +π rad phase.
    added_phase_of_optimal_cz20_gate = 313
    
    # Go through the frequency list, and the input state list.
    conditional_cross_ramsey_input_state_list = ['0+', '1+'] if (conditional_qubit_is == 'A') else ['+0', '+1']
    for current_cz20_frequency in cz20_freq_arr:
        
        # Get a time estimate for printing "time remaining" to the user.
        tick = time.time()
        
        ## Execute a CZ₂₀ conditional cross-Ramsey for states |0+⟩ and |1+⟩!
        
        # First, we need to know the local accumulated phase correction
        # for both qubits at the current CZ₂₀ frequency.
        untuned_cz20_results = []
        for use_this_input_state in ['+0','0+']:
            
            # Run a CZ₂₀ and get the local accumulated phase of both qubits.
            ## Perform this step within a connection handler loop,
            ## to catch crashes.
            success = False
            tries = 0
            while ((not success) and (tries <= 5)):
                tries += 1
                try:
                    cz_result = cz20_tune_local_accumulated_phase(
                        ip_address = ip_address,
                        ext_clk_present = ext_clk_present,
                        
                        readout_stimulus_port = readout_stimulus_port,
                        readout_sampling_port = readout_sampling_port,
                        readout_freq_nco = readout_freq_nco,
                        readout_freq_A = readout_freq_A,
                        readout_amp_A = readout_amp_A,
                        readout_freq_B = readout_freq_B,
                        readout_amp_B = readout_amp_B,
                        readout_duration = readout_duration,
                        
                        sampling_duration = sampling_duration,
                        readout_sampling_delay = readout_sampling_delay,
                        repetition_rate = repetition_rate,
                        integration_window_start = integration_window_start,
                        integration_window_stop = integration_window_stop,
                        
                        control_port_A = control_port_A,
                        control_freq_nco_A = control_freq_nco_A,
                        control_freq_01_A = control_freq_01_A,
                        control_amp_01_A = control_amp_01_A,
                        control_port_B = control_port_B,
                        control_freq_nco_B = control_freq_nco_B,
                        control_freq_01_B = control_freq_01_B,
                        control_amp_01_B = control_amp_01_B,
                        control_duration_01 = control_duration_01,
                        
                        coupler_dc_port = coupler_dc_port,
                        coupler_dc_bias = coupler_dc_bias,
                        settling_time_of_bias_tee = settling_time_of_bias_tee,
                        
                        coupler_ac_port = coupler_ac_port,
                        coupler_ac_freq_nco = coupler_ac_freq_nco,
                        coupler_ac_freq_cz20 = current_cz20_frequency, ## Note!
                        coupler_ac_amp_cz20 = coupler_ac_amp_cz20,
                        coupler_ac_single_edge_time_cz20 = coupler_ac_single_edge_time_cz20,
                        coupler_ac_plateau_duration_cz20 = coupler_ac_plateau_duration_cz20,
                        
                        num_averages = num_averages,
                        
                        num_phases = num_phases,
                        phase_sweep_rad_min = phase_sweep_rad_min,
                        phase_sweep_rad_max = phase_sweep_rad_max,
                        
                        phase_adjustment_after_cz20_A = 0.0, # At this point in time, the required local phase correction is unknown.
                        phase_adjustment_after_cz20_B = 0.0, # At this point in time, the required local phase correction is unknown.
                        
                        prepare_input_state = use_this_input_state,
                        
                        reset_dc_to_zero_when_finished = reset_dc_to_zero_when_finished,
                        
                        save_complex_data = save_complex_data,
                        save_raw_time_data = save_raw_time_data,
                        use_log_browser_database = use_log_browser_database,
                        suppress_log_browser_export = suppress_log_browser_export_of_suboptimal_data,
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
                    )
                    
                    success = True # Done
                except ConnectionRefusedError:
                    if force_device_reboot_on_connection_error:
                        force_system_restart_over_ssh("129.16.115.184")
            assert success, "Halted! Unrecoverable crash detected."
            untuned_cz20_results.append( cz_result[0] )
        
        # At this point, untuned_cz20_results contains the filepaths of
        # the two measurements that yield the required phase correction.
        
        # By fitting, acquire the local accumulated phase
        # from the previous two measurements. Fit the phase offsets!
        local_accumulated_phase_correction = fit_phase_offset(
            raw_data_or_path_to_data = untuned_cz20_results,
            control_phase_arr = [],
            i_renamed_the_control_phase_arr_to = '',
            plot_for_this_many_seconds = 0,
            verbose = False,
        )
        
        # Dig out the sought-for phase correction values, knowing
        # what input states we ran previously: |+0> followed by |0+>
        ## I.e.: val in (val, errorbar) of res=0 of Z=0 of file list entry 0.
        ## And:  val in (val, errorbar) of res=1 of Z=0 of file list entry 1.
        ##     local_accum....[file_entry = 0][res = 0][z = 0][0 = val]
        ##     local_accum....[file_entry = 1][res = 1][z = 0][0 = val]
        
        # WITH ONE VERY IMPORTANT DETAIL: THE FIT RETURNS THE PHASE OFFSET
        # FOR THE COSINUSOID, WHEREAS WE ARE LOOKING FOR THE PHASE THAT
        # WE ADDED TO THE VIRTUAL-Z GATE. WHICH IS FIT / 2!
        # ... due to the experiment design where we got that phase value.
        
        curr_local_phase_correction_A = local_accumulated_phase_correction[0][0][0][0] / 2
        curr_local_phase_correction_B = local_accumulated_phase_correction[1][1][0][0] / 2
        
        # Remake the local_accumulated_phase_correction object.
        local_accumulated_phase_correction = \
            curr_local_phase_correction_A, \
            curr_local_phase_correction_B
        del curr_local_phase_correction_A, curr_local_phase_correction_B
        
        # Delete the suboptimal files?
        if suppress_log_browser_export_of_suboptimal_data:
            for file_to_remove in untuned_cz20_results:
                # The recent data is irrelevant, remove it.
                attempt = 0
                max_attempts = 5
                success = False
                while (attempt < max_attempts) and (not success):
                    try:
                        os.remove(os.path.abspath(file_to_remove))
                        success = True
                    except FileNotFoundError:
                        attempt += 1
                        time.sleep(0.2)
                if (not success):
                    raise OSError("Error: could not delete data file "+str(os.path.abspath(untuned_cz20_results[ii]))+" after "+str(max_attempts)+" attempts. Halting.")
        
        # Clean up.
        del untuned_cz20_results
        
        # At this point, the local_accumulated_phase_correction TUPLE contains
        # the requried local accumulated phase correction for the
        # two qubits. Run the cross-Ramsey measurements.
        tuned_conditional_cross_ramsey_results = []
        for use_this_input_state in conditional_cross_ramsey_input_state_list:
            
            # Execute conditional cross-Ramsey, using the figured-out
            # local accumulated phase corrections.
            ## Perform this step within a connection handler loop,
            ## to catch crashes.
            success = False
            tries = 0
            while ((not success) and (tries <= 5)):
                tries += 1
                try:
                    res_cz_cross_ramsey = cz20_conditional_cross_ramsey(
                        ip_address = ip_address,
                        ext_clk_present = ext_clk_present,
                        
                        readout_stimulus_port = readout_stimulus_port,
                        readout_sampling_port = readout_sampling_port,
                        readout_freq_nco = readout_freq_nco,
                        readout_freq_A = readout_freq_A,
                        readout_amp_A = readout_amp_A,
                        readout_freq_B = readout_freq_B,
                        readout_amp_B = readout_amp_B,
                        readout_duration = readout_duration,
                        
                        sampling_duration = sampling_duration,
                        readout_sampling_delay = readout_sampling_delay,
                        repetition_rate = repetition_rate,
                        integration_window_start = integration_window_start,
                        integration_window_stop = integration_window_stop,
                        
                        control_port_A = control_port_A,
                        control_freq_nco_A = control_freq_nco_A,
                        control_freq_01_A = control_freq_01_A,
                        control_amp_01_A = control_amp_01_A,
                        control_port_B = control_port_B,
                        control_freq_nco_B = control_freq_nco_B,
                        control_freq_01_B = control_freq_01_B,
                        control_amp_01_B = control_amp_01_B,
                        control_duration_01 = control_duration_01,
                        
                        coupler_dc_port = coupler_dc_port,
                        coupler_dc_bias = coupler_dc_bias,
                        settling_time_of_bias_tee = settling_time_of_bias_tee,
                        
                        coupler_ac_port = coupler_ac_port,
                        coupler_ac_freq_nco = coupler_ac_freq_nco,
                        coupler_ac_freq_cz20 = current_cz20_frequency, ## Note!
                        coupler_ac_amp_cz20 = coupler_ac_amp_cz20,
                        coupler_ac_single_edge_time_cz20 = coupler_ac_single_edge_time_cz20,
                        coupler_ac_plateau_duration_cz20 = coupler_ac_plateau_duration_cz20,
                        
                        num_averages = num_averages,
                        
                        num_phases = num_phases,
                        phase_sweep_rad_min = phase_sweep_rad_min,
                        phase_sweep_rad_max = phase_sweep_rad_max,
                        
                        phase_adjustment_after_cz20_A = local_accumulated_phase_correction[0], ## Note! This value is known now.
                        phase_adjustment_after_cz20_B = local_accumulated_phase_correction[1], ## Note! This value is known now.
                        
                        prepare_input_state = use_this_input_state, ## Note!
                        
                        reset_dc_to_zero_when_finished = reset_dc_to_zero_when_finished,
                        
                        save_complex_data = save_complex_data,
                        save_raw_time_data = save_raw_time_data,
                        use_log_browser_database = use_log_browser_database,
                        suppress_log_browser_export = suppress_log_browser_export_of_suboptimal_data,
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
                    )
                    
                    success = True # Done
                except ConnectionRefusedError:
                    if force_device_reboot_on_connection_error:
                        force_system_restart_over_ssh("129.16.115.184")
            assert success, "Halted! Unrecoverable crash detected."
            
            tuned_conditional_cross_ramsey_results.append( \
                res_cz_cross_ramsey[0]
            )
        
        ## We may not yet clear out local_accumulated_phase_correction.
        ## DO NOT del local_accumulated_phase_correction
        
        # At this point, tuned_conditional_cross_ramsey_results contains
        # the filepaths of the two conditional cross-Ramsey measurements.
        # One with the CZ₂₀ gate turned on, and one with the CZ₂₀ gate off.
        
        # Let's fit cosines into the file, to see how much phase
        # that the alleged CZ₂₀ gate actually adds.
        current_added_phase_by_cz20, current_absolute_phase_offset_from_cz20_gate = fit_conditional_phase_offset(
            raw_data_or_path_to_data_for_condition_OFF = tuned_conditional_cross_ramsey_results[0],
            raw_data_or_path_to_data_for_condition_ON  = tuned_conditional_cross_ramsey_results[1],
            conditional_qubit_is = conditional_qubit_is,
            control_phase_arr = [],
            i_renamed_the_control_phase_arr_to = '',
            plot_for_this_many_seconds = 0.0,
            verbose = False,
        )
        
        for file_to_delete in tuned_conditional_cross_ramsey_results:
            if suppress_log_browser_export_of_suboptimal_data:
                # The recent data is irrelevant, remove it.
                attempt = 0
                max_attempts = 5
                success = False
                while (attempt < max_attempts) and (not success):
                    try:
                        os.remove(os.path.abspath(file_to_delete))
                        success = True
                    except FileNotFoundError:
                        attempt += 1
                        time.sleep(0.2)
                if (not success):
                    raise OSError("Error: could not delete data file "+str(os.path.abspath( file_to_delete ))+" after "+str(max_attempts)+" attempts. Halting.")
        
        # current_added_phase_by_cz20 contains how much phase was added
        # by the current CZ₂₀ at this frequency. Append value to list.
        cz20_added_phase_arr.append( current_added_phase_by_cz20 )
        
        # At this point, we may update our tracking variables whether
        # the value was the currently best one.
        if np.abs(current_added_phase_by_cz20 - np.pi) < added_phase_of_optimal_cz20_gate:
            optimal_cz20_frequency = current_cz20_frequency
            optimal_phase_adjustment_after_cz20_A = local_accumulated_phase_correction[0]
            optimal_phase_adjustment_after_cz20_B = local_accumulated_phase_correction[1]
            added_phase_of_optimal_cz20_gate = current_added_phase_by_cz20
            absolute_phase_offset_from_cz20_gate = current_absolute_phase_offset_from_cz20_gate
        
        # Tock the clock, and show the user the time remaining.
        tock = time.time() # Get a time estimate.
        num_tick_tocks += 1
        total_dur += (tock - tick)
        average_duration_per_point = total_dur / num_tick_tocks
        calc = (len(cz20_freq_arr)-num_tick_tocks)*average_duration_per_point
        if (calc != 0.0):
            # Print "true" time remaining.
            show_user_time_remaining(calc)
    
    # At this point, we've finished building our arrays and are ready
    # to send them off into a final plot.
    ## The array cz20_added_phase_arr is our Y-axis.
    ## Let's treat it before exporting. Remember to grab the "best" values too.
    cz20_added_phase_arr = np.array(cz20_added_phase_arr)
    
    # Data to be stored.
    hdf5_steps = [
        'cz20_freq_arr', "Hz",
    ]
    hdf5_singles = [
        'optimal_cz20_frequency', "Hz",
        'optimal_phase_adjustment_after_cz20_A', "rad",
        'optimal_phase_adjustment_after_cz20_B', "rad",
        'added_phase_of_optimal_cz20_gate', "rad",
        'absolute_phase_offset_from_cz20_gate', "rad",
        
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
        'coupler_ac_freq_cz20_centre', "Hz",
        'coupler_ac_freq_cz20_span', "Hz",
        'coupler_ac_amp_cz20', "FS",
        'coupler_ac_single_edge_time_cz20', "s",
        'coupler_ac_plateau_duration_cz20', "s",
        
        'num_freqs', "",
        'num_averages', "",
        
        'num_phases', "",
        'phase_sweep_rad_min', "rad",
        'phase_sweep_rad_max', "rad",
    ]
    hdf5_logs = [
        'cz20_added_phase_arr', "rad",
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
    
    # Export the non-complex data (in a Log Browser compatible format).
    string_arr_to_return = export_processed_data_to_file(
        filepath_of_calling_script = os.path.realpath(__file__),
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        processed_data = [np.array([cz20_added_phase_arr])],
        fetched_data_scale = axes['y_scaler'],
        fetched_data_offset = axes['y_offset'],
        
        #time_vector = time_vector,   # Nothing to export here.
        #fetched_data_arr = [],       # Nothing to export here.
        timestamp = get_timestamp_string(),
        append_to_log_name_before_timestamp = '20_frequency_versus_added_phase',
        append_to_log_name_after_timestamp = '',
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export,
    )
    
    return string_arr_to_return