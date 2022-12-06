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
from datetime import datetime
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_timestamp_string, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save


def t1_sweep_flux(
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
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_nco,
    control_freq_01,
    control_duration_01,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_averages,
    
    num_delays,
    dt_per_time_step,
    
    num_biases,
    coupler_bias_min = +0.0,
    coupler_bias_max = +1.0,
    
    save_complex_data = True,
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
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if type(coupler_dc_port) == int:
        raise TypeError( \
            "Halted! The input argument coupler_dc_port must be provided "  + \
            "as a list. Typecasting was not done for you, since some user " + \
            "setups combine several ports together galvanically. Merely "   + \
            "typecasting the input int to [int] risks damaging their "      + \
            "setups. All items in the coupler_dc_port list will be treated "+ \
            "as ports to be used for DC-biasing a coupler.")
    if num_biases < 1:
        num_biases = 1
        print("Note: num_biases was less than 1, and was thus set to 1.")
        if coupler_bias_min != 0.0:
            print("Note: the coupler bias was thus set to 0.")
            coupler_bias_min = 0.0
    elif coupler_dc_port == []:
        if num_biases != 1:
            num_biases = 1
            print("Note: num_biases was set to 1, since the coupler_port array was empty.")
        if coupler_bias_min != 0.0:
            print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
            coupler_bias_min = 0.0
    
    ## Initial array declaration
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
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
        added_delay_for_bias_tee = int(round(added_delay_for_bias_tee / plo_clk_T)) * plo_clk_T
        dt_per_time_step = int(round(dt_per_time_step / plo_clk_T)) * plo_clk_T
        
        if (integration_window_stop - integration_window_start) < plo_clk_T:
            integration_window_stop = integration_window_start + plo_clk_T
            print("Warning: an impossible integration window was defined. The window stop was moved to "+str(integration_window_stop)+" s.")
        
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
        # Coupler port amplitude (the bias)
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_amp_arr,
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
        
        for ii in range(num_delays):
        
            # Re-apply the coupler DC pulse once one tee risetime has passed.
            if coupler_dc_port != []:
                if ii > 0:
                    for bias_tone in coupler_bias_tone:
                        bias_tone.set_total_duration(control_duration_01 + readout_duration + ii * dt_per_time_step + repetition_delay)
                else:
                    for bias_tone in coupler_bias_tone:
                        bias_tone.set_total_duration(control_duration_01 + readout_duration + repetition_delay)
            
            # Output the pi01-pulse along with the coupler DC tone
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse_pi_01)
            if coupler_dc_port != []:
                pls.output_pulse(T, coupler_bias_tone)
            T += control_duration_01
            
            # Wait for the qubit to decohere some.
            T += ii * dt_per_time_step
            
            # Commence readout after said delay.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
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
        
        # Repeat the whole sequence `num_amplitudes` times,
        # then average `num_averages` times
        
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
            if coupler_dc_bias > 0.0:
                with_or_without_bias_string = "_with_bias"
            else:
                with_or_without_bias_string = ""
        
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
            'readout_freq_nco', "Hz",
            'readout_freq', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_nco', "Hz", 
            'control_freq_01', "Hz", 
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
            resonator_freq_if_arrays_to_fft = [np.abs(readout_freq_if)],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_delays,
            outer_loop_size = num_biases,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = with_or_without_bias_string,
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
        ))
    
    return string_arr_to_return


def t1_sweep_flux_multiplexed_ro0(
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
    control_amp_01_A,
    control_freq_01_A,
    control_port_B,
    control_amp_01_B,
    control_freq_01_B,
    control_duration_01,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_averages,
    
    num_delays,
    dt_per_time_step,
    
    num_biases,
    coupler_bias_min = -1.0,
    coupler_bias_max = +1.0,
    
    save_complex_data = True,
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
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if type(coupler_dc_port) == int:
        raise TypeError( \
            "Halted! The input argument coupler_dc_port must be provided "  + \
            "as a list. Typecasting was not done for you, since some user " + \
            "setups combine several ports together galvanically. Merely "   + \
            "typecasting the input int to [int] risks damaging their "      + \
            "setups. All items in the coupler_dc_port list will be treated "+ \
            "as ports to be used for DC-biasing a coupler.")
    if num_biases < 1:
        num_biases = 1
        print("Note: num_biases was less than 1, and was thus set to 1.")
        if coupler_bias_min != 0.0:
            print("Note: the coupler bias was thus set to 0.")
            coupler_bias_min = 0.0
    elif coupler_dc_port == []:
        if num_biases != 1:
            num_biases = 1
            print("Note: num_biases was set to 1, since the coupler_port array was empty.")
        if coupler_bias_min != 0.0:
            print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
            coupler_bias_min = 0.0
    
    
    ## Initial array declaration
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
    
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
        
        if (integration_window_stop - integration_window_start) < plo_clk_T:
            integration_window_stop = integration_window_start + plo_clk_T
            print("Warning: an impossible integration window was defined. The window stop was moved to "+str(integration_window_stop)+" s.")
        
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
        
        # Repeat the whole sequence `num_amplitudes` times,
        # then average `num_averages` times
        
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
        
        print("Saving data")

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
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz", 
            'control_port_B', "",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz", 
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
            resonator_freq_if_arrays_to_fft = [np.abs(readout_freq_if_A), np.abs(readout_freq_if_B)], # TODO: Automatic USB / LSB selection not considered, always set positive for now.
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_delays,
            outer_loop_size = num_biases,
            
            save_complex_data = save_complex_data,
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'multiplexed_ro',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
        ))
    
    return string_arr_to_return