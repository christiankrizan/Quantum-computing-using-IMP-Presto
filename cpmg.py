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

def cpmg_known_number_of_pi_pulses_and_delay(
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
    control_amp_01,
    control_freq_nco,
    control_freq_01,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    delay_arr,
    number_of_pi_pulses,
    
    num_averages,
    
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
    ''' Perform CPMG spectroscopy.
        
        CPMG is the Carr-Purcell-Meiboom-Gill method.
        Imagine a T₂-echo measurement, but with a lot of additional refocusing
        pulses.
        
        Process:
            - Perform π/2 pulse.
            --  Perform t/2 delay (where t is some delay time).
            - Perform a sequence of π pulses.
            --  Perform 1.0 · t delay between pulses.
            --  ... except the last pulse, which has 0.5 · t delay, not 1.0·t.
            - Perform a final π/2 pulse.
            - Readout.
        
        Step duration between π/2 pulses.
        
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
        
        # Control port
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
        
        # Make all values in the delay_arr representatble.
        for jj in range(len(delay_arr)):
            delay_arr[jj] = int(round(delay_arr[jj] / plo_clk_T)) * plo_clk_T
        # Remove duplicate entries in the delay_arr.
        # Make delay_arr into a numpy array if it already isn't one.
        delay_arr = np.unique( np.array(delay_arr) )
        
        
        ''' Setup mixers '''
        
        # Readout mixer
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
        
        # Readout port amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitude
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
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
        
        
        ### Setup pulse "control_pulse_pi_01" ###
        
        # Setup control_pulse_pi_01 and control_pulse_pi_01_half pulse envelopes.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_pulse_pi_01_half = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01 / 2, # Note!
            template_q  = control_envelope_01 / 2, # Halved.
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
        
        ## Here, we define the total_cpmg_duration list.
        ## It will be filled up as we move along.
        total_cpmg_duration = []
        
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
        
        # For every delay to step through:
        for ii in range(len(delay_arr)):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            ## Get the current t for the CPMG sequence, i.e. the 
            ## waiting time between pi-pulses. And, a legal cpmg_t/2.
            cpmg_t = delay_arr[ii]
            cpmg_t_half = int(round( ((delay_arr[ii])/2) / plo_clk_T)) * plo_clk_T
            
            # Apply the frequency-swept pi01 pulses.
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse_pi_01_half)
            T += control_duration_01
            
            ## For CPMG, add a time t/2 at first.
            T += cpmg_t_half
            
            ## Then add a bunch of pi-pulses, spaced at t.
            for jj in range(number_of_pi_pulses):
                pls.output_pulse(T, control_pulse_pi_01)
                T += control_duration_01
                
                # Add t
                T += cpmg_t
            
            ## Then finally play the last pi/2 pulse, after cpmg_t/2 seconds.
            T += cpmg_t_half
            pls.output_pulse(T, control_pulse_pi_01_half)
            T += control_duration_01
            
            ## Here, we grab the time for this current instance of the
            ## CPMG sequence. The "duration" is the total time that
            ## the qubit is idling.
            total_cpmg_duration += ((T - T_begin) - (number_of_pi_pulses+2)*control_duration_01)
            
            # Commence readout
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
            repeat_count    =   1,
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
        
        ## Cast the list total_cpmg_duration into numpy.
        total_cpmg_duration = np.array( total_cpmg_duration )
        
        # Data to be stored.
        hdf5_steps = [
            'total_cpmg_duration', "s",
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
            'control_amp_01', "FS",
            'control_freq_nco', "Hz",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'number_of_pi_pulses', "",
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
            inner_loop_size = len(total_cpmg_duration),
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '01_CPMG' + str(number_of_pi_pulses) + '_pi_pulses' + with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return