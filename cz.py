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
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_timestamp_string, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save
from time_remaining_printer import show_user_time_remaining

def cz20_sweep_amplitude_and_detuning_for_t_half(
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
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_rising_edge_time_cz20,
    coupler_ac_plateau_duration_maximise_cz20,
    coupler_ac_freq_cz20_nco,
    coupler_ac_freq_cz20_centre_if,
    coupler_ac_freq_cz20_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
    save_complex_data = True,
    use_log_browser_database = True,
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
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    ## Initial array declaration
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
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
        coupler_ac_rising_edge_time_cz20 = int(round(coupler_ac_rising_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_maximise_cz20 = int(round(coupler_ac_plateau_duration_maximise_cz20 / plo_clk_T)) * plo_clk_T
        
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
            sync      = False,
        )
        # Control port mixers
        pls.hardware.configure_mixer(
            freq      = control_freq_01_A,
            out_ports = control_port_A,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_01_B,
            out_ports = control_port_B,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_cz20_nco,
            out_ports = coupler_ac_port,
            sync      = (coupler_dc_port == []),
        )
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                sync      = True,  # Sync here
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
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
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
        readout_freq_if_A = np.abs(readout_freq_nco - readout_freq_A)
        readout_freq_if_B = np.abs(readout_freq_nco - readout_freq_B)
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_A,
            phases       = np.full_like(readout_freq_if_A, 0.0),
            phases_q     = np.full_like(readout_freq_if_A, -np.pi/2), # USB!  ##+np.pi/2, # LSB
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
        
        # Setup control_pulse_pi_01 carrier tones, considering that there are digital mixers.
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        
        
        ### Setup the CZ20 gate pulse
        
        # The initially set duration will not be swept by the sequencer
        # program.
        coupler_ac_duration_cz20 = \
            2 * coupler_ac_rising_edge_time_cz20 + \
            coupler_ac_plateau_duration_maximise_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_rising_edge_time_cz20,
            fall_time   = coupler_ac_rising_edge_time_cz20,
        )
        
        # Setup the cz20 pulse carrier, this tone will be swept in frequency.
        # Since we set the mixer to some NCO value, we probably want to use
        # the lower sideband for sweeping the span (not the upper).
        f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
        f_stop = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
        coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_cz20_nco - coupler_ac_freq_cz20_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = coupler_ac_freq_cz20_if_arr,
            phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
        
            # Redefine the coupler DC pulse duration for repeated playback
            # once one tee risetime has passed.
            coupler_bias_tone.set_total_duration(
                control_duration_01 + \
                coupler_ac_duration_cz20 + \
                readout_duration + \
                repetition_delay \
            )
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):

            # Re-apply the coupler bias tone.
            if coupler_dc_port != []:
                pls.output_pulse(T, coupler_bias_tone)
            
            # Put the system into state |11>
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
            T += control_duration_01
            
            # Apply the CZ20 gate, with parameters being swept.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next scanned frequency
            pls.next_frequency(T, coupler_ac_port, group = 0)
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        
        # Increment the CZ20 pulse amplitude.
        pls.next_scale(T, coupler_ac_port, group = 0)
        
        # Move to next iteration.
        T += repetition_delay
        
        
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
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
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
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco',"Hz",
            
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
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_duration_cz20', "s",
            'coupler_ac_freq_cz20_nco', "Hz",
            'coupler_ac_freq_cz20_centre_if', "Hz",
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
                    hdf5_logs.append('Probability for state |'+statep+'>')
                    # Let the record show that I wanted to write a Unicode ket
                    # instead of the '>' character, but the Log Browser's
                    # support for anything non-bland is erratic at best.
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
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [readout_freq_if_A, readout_freq_if_B],
            
            filepath_of_calling_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_freqs,
            outer_loop_size = num_amplitudes,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = '20_sweep_amplitude_and_detuning',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return
    
def cz20_sweep_amplitude_and_detuning_for_t_half_state_probability(
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
    
    control_amp_12_A,
    control_freq_12_A,
    control_amp_12_B,
    control_freq_12_B,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_rising_edge_time_cz20,
    coupler_ac_plateau_duration_maximise_cz20,
    coupler_ac_freq_cz20_nco,
    coupler_ac_freq_cz20_centre_if,
    coupler_ac_freq_cz20_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['20'],
    
    use_log_browser_database = True,
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
        
        The readout is multiplexed between two pairwise-coupled transmons.
        The fetched result will be sent into a state discriminator.
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    ## Initial array declaration
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
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
        coupler_ac_rising_edge_time_cz20 = int(round(coupler_ac_rising_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_maximise_cz20 = int(round(coupler_ac_plateau_duration_maximise_cz20 / plo_clk_T)) * plo_clk_T
        
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
            sync      = False,
        )
        # Control port mixers
        high_res_A  = max( [control_freq_01_A, control_freq_12_A] )
        low_res_A   = min( [control_freq_01_A, control_freq_12_A] )
        control_freq_nco_A = high_res_A - (high_res_A - low_res_A)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            sync      = False,
        )
        high_res_B  = max( [control_freq_01_B, control_freq_12_B] )
        low_res_B   = min( [control_freq_01_B, control_freq_12_B] )
        control_freq_nco_B = high_res_B - (high_res_B - low_res_B)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            sync      = (coupler_dc_port == []),
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_cz20_nco,
            out_ports = coupler_ac_port,
            sync      = (coupler_dc_port == []),
        )
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                sync      = True,  # Sync here
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
            scales          = coupler_ac_amp_arr, # This value will be swept!
        )
        if coupler_dc_port != []:
            pls.setup_scale_lut(
                output_ports    = coupler_dc_port,
                group           = 0,
                scales          = coupler_dc_bias,
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
        readout_freq_if_A = np.abs(readout_freq_nco - readout_freq_A)
        readout_freq_if_B = np.abs(readout_freq_nco - readout_freq_B)
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_A,
            phases       = np.full_like(readout_freq_if_A, 0.0),
            phases_q     = np.full_like(readout_freq_if_A, -np.pi/2), # USB !  ##+np.pi/2, # LSB
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
        
        # Setup control_pulse_pi_12_A and _B pulse envelopes.
        control_ns_12 = int(round(control_duration_12 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_12 = sin2(control_ns_12)
        control_pulse_pi_12_A = pls.setup_template(
            output_port = control_port_A,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        control_pulse_pi_12_B = pls.setup_template(
            output_port = control_port_B,
            group       = 1,
            template    = control_envelope_12,
            template_q  = control_envelope_12,
            envelope    = True,
        )
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01_A = np.abs(control_freq_nco_A - control_freq_01_A)
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = control_freq_if_01_A,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12_A = np.abs(control_freq_nco_A - control_freq_12_A)
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 1,
            frequencies  = control_freq_if_12_A,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        # ... Now for the other control port
        control_freq_if_01_B = np.abs(control_freq_nco_B - control_freq_01_B)
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = control_freq_if_01_B,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        control_freq_if_12_B = np.abs(control_freq_nco_B - control_freq_12_B)
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 1,
            frequencies  = control_freq_if_12_B,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!
        )
        
        
        ### Setup the CZ20 gate pulse
        
        # The initially set duration will not be swept by the sequencer
        # program.
        coupler_ac_duration_cz20 = \
            2 * coupler_ac_rising_edge_time_cz20 + \
            coupler_ac_plateau_duration_maximise_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_rising_edge_time_cz20,
            fall_time   = coupler_ac_rising_edge_time_cz20,
        )
        
        # Setup the cz20 pulse carrier, this tone will be swept in frequency.
        # Since we set the mixer to some NCO value, we probably want to use
        # the lower sideband for sweeping the span (not the upper).
        f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
        f_stop = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
        coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_cz20_nco - coupler_ac_freq_cz20_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = coupler_ac_freq_cz20_if_arr,
            phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
        
            # Redefine the coupler DC pulse duration for repeated playback
            # once one tee risetime has passed.
            coupler_bias_tone.set_total_duration(
                control_duration_01 + \
                coupler_ac_duration_cz20 + \
                readout_duration + \
                repetition_delay \
            )
        
        # For every AC coupler amplitude to sweep over:
        for jj in range(num_amplitudes):
        
            # For every resonator stimulus pulse frequency to sweep over:
            for ii in range(num_freqs):

                # Re-apply the coupler bias tone.
                if coupler_dc_port != []:
                    pls.output_pulse(T, coupler_bias_tone)
                
                # Put the system into state |11>
                pls.reset_phase(T, [control_port_A, control_port_B])
                pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
                T += control_duration_01
                
                # Apply the CZ20 gate, with parameters being swept.
                pls.reset_phase(T, coupler_ac_port)
                pls.output_pulse(T, coupler_ac_pulse_cz20)
                T += coupler_ac_duration_cz20
                
                # Commence multiplexed readout
                pls.reset_phase(T, readout_stimulus_port)
                pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                pls.store(T + readout_sampling_delay) # Sampling window
                T += readout_duration
                
                # Move to next scanned frequency
                pls.next_frequency(T, coupler_ac_port, group = 0)
                
                # Await a new repetition, after which a new coupler DC bias tone
                # will be added - and a new frequency set for the readout tone.
                T += repetition_delay
            
            
            # Increment the CZ20 pulse amplitude.
            pls.next_scale(T, coupler_ac_port, group = 0)
        
            # Move to next iteration.
            T += repetition_delay
        
        
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
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
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
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco', "Hz",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port_A', "",
            'control_port_B', "",
            
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz",
            'control_duration_01', "s",
            
            'control_amp_12_A', "FS",
            'control_freq_12_A', "Hz",
            'control_amp_12_B', "FS",
            'control_freq_12_B', "Hz",
            'control_duration_12', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_duration_cz20', "s",
            'coupler_ac_freq_cz20_nco', "Hz",
            'coupler_ac_freq_cz20_centre_if', "Hz",
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
                    hdf5_logs.append('Probability for state |'+statep+'>')
                    # Let the record show that I wanted to write a Unicode ket
                    # instead of the '>' character, but the Log Browser's
                    # support for anything non-bland is erratic at best.
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
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [readout_freq_if_A, readout_freq_if_B],
            
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
            append_to_log_name_before_timestamp = 'state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return



"""def cz20_sweep_amplitude_and_detuning_for_t_half_state_probability(
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
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_rising_edge_time_cz20,
    coupler_ac_plateau_duration_maximise_cz20,
    coupler_ac_freq_cz20_nco,
    coupler_ac_freq_cz20_centre_if,
    coupler_ac_freq_cz20_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['20'],
    
    use_log_browser_database = True,
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
        
        The readout is multiplexed between two pairwise-coupled transmons.
        The fetched result will be sent into a state discriminator.
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    ## Initial array declaration
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
    # Set an initial num_amplitudes_per_run
    num_amplitudes_per_run = num_amplitudes
    length_of_last_run = num_amplitudes # Remember: if there's only 1 run ;)
    
    # For this script, it makes sense to get a time estimate.
    average_duration_per_point = 0.0
    num_tick_tocks = 0
    total_dur = 0.0
    use_time_printer = False
    
    ## We will very easily over-step the maximum allowed number of commands.
    ## Thus, we will remake some variables on the fly to support a large,
    ## segmented measurement.
    assembled_data_maximus = [None]
    measurement_finished = False
    measurement_was_cut_into_segments = False # TODO! For now, let's assume that there can only be 1 re-scaling of the measurement.
    there_will_be_this_many_runs = 1 # Default value is always 1.
    we_have_completed_this_many_runs = 0
    while (not measurement_finished):
        
        tick = time.time() # Get a time estimate.
        
        # Is this the last run? Then overwrite the per-run length
        # with this value.
        if we_have_completed_this_many_runs == (there_will_be_this_many_runs-1):
            num_amplitudes_per_run = length_of_last_run
            current_amp_arr_to_run = coupler_ac_amp_arr[-(length_of_last_run):]
        else:
            # What portion of the amplitude array are we covering in this run?
            current_amp_arr_to_run = coupler_ac_amp_arr[(0+num_amplitudes_per_run*we_have_completed_this_many_runs):(num_amplitudes_per_run+num_amplitudes_per_run*we_have_completed_this_many_runs)]
        
        if num_amplitudes_per_run <= 0:
            raise RuntimeError("Error: No run is supposed to be 0 length long.")  ## TODO
            
        try:
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
                coupler_ac_rising_edge_time_cz20 = int(round(coupler_ac_rising_edge_time_cz20 / plo_clk_T)) * plo_clk_T
                coupler_ac_plateau_duration_maximise_cz20 = int(round(coupler_ac_plateau_duration_maximise_cz20 / plo_clk_T)) * plo_clk_T
                
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
                    sync      = False,
                )
                # Control port mixers
                pls.hardware.configure_mixer(
                    freq      = control_freq_01_A,
                    out_ports = control_port_A,
                    sync      = False,
                )
                pls.hardware.configure_mixer(
                    freq      = control_freq_01_B,
                    out_ports = control_port_B,
                    sync      = False,
                )
                # Coupler port mixer
                pls.hardware.configure_mixer(
                    freq      = coupler_ac_freq_cz20_nco,
                    out_ports = coupler_ac_port,
                    sync      = (coupler_dc_port == []),
                )
                if coupler_dc_port != []:
                    pls.hardware.configure_mixer(
                        freq      = 0.0,
                        out_ports = coupler_dc_port,
                        sync      = True,  # Sync here
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
                    scales          = current_amp_arr_to_run, # This value will be swept!
                )
                if coupler_dc_port != []:
                    pls.setup_scale_lut(
                        output_ports    = coupler_dc_port,
                        group           = 0,
                        scales          = coupler_dc_bias,
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
                readout_freq_if_A = np.abs(readout_freq_nco - readout_freq_A)
                readout_freq_if_B = np.abs(readout_freq_nco - readout_freq_B)
                pls.setup_freq_lut(
                    output_ports = readout_stimulus_port,
                    group        = 0,
                    frequencies  = readout_freq_if_A,
                    phases       = np.full_like(readout_freq_if_A, 0.0),
                    phases_q     = np.full_like(readout_freq_if_A, -np.pi/2), # USB !  ##+np.pi/2, # LSB
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
                
                # Setup control_pulse_pi_01 carrier tones, considering that there are digital mixers.
                pls.setup_freq_lut(
                    output_ports = control_port_A,
                    group        = 0,
                    frequencies  = 0.0,
                    phases       = 0.0,
                    phases_q     = 0.0,
                )
                pls.setup_freq_lut(
                    output_ports = control_port_B,
                    group        = 0,
                    frequencies  = 0.0,
                    phases       = 0.0,
                    phases_q     = 0.0,
                )
                
                
                ### Setup the CZ20 gate pulse
                
                # The initially set duration will not be swept by the sequencer
                # program.
                coupler_ac_duration_cz20 = \
                    2 * coupler_ac_rising_edge_time_cz20 + \
                    coupler_ac_plateau_duration_maximise_cz20
                coupler_ac_pulse_cz20 = pls.setup_long_drive(
                    output_port = coupler_ac_port,
                    group       = 0,
                    duration    = coupler_ac_duration_cz20,
                    amplitude   = 1.0,
                    amplitude_q = 1.0,
                    rise_time   = coupler_ac_rising_edge_time_cz20,
                    fall_time   = coupler_ac_rising_edge_time_cz20,
                )
                
                # Setup the cz20 pulse carrier, this tone will be swept in frequency.
                # Since we set the mixer to some NCO value, we probably want to use
                # the lower sideband for sweeping the span (not the upper).
                f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
                f_stop = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
                coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
                
                # Use the lower sideband. Note the minus sign.
                coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_cz20_nco - coupler_ac_freq_cz20_if_arr
                
                # Setup LUT
                pls.setup_freq_lut(
                    output_ports    = coupler_ac_port,
                    group           = 0,
                    frequencies     = coupler_ac_freq_cz20_if_arr,
                    phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
                    phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
                
                    # Redefine the coupler DC pulse duration for repeated playback
                    # once one tee risetime has passed.
                    coupler_bias_tone.set_total_duration(
                        control_duration_01 + \
                        coupler_ac_duration_cz20 + \
                        readout_duration + \
                        repetition_delay \
                    )
                
                # For every AC coupler amplitude to sweep over:
                for jj in range(num_amplitudes_per_run):
                    
                    # For every resonator stimulus pulse frequency to sweep over:
                    for ii in range(num_freqs):
                        
                        # Re-apply the coupler bias tone.
                        if coupler_dc_port != []:
                            pls.output_pulse(T, coupler_bias_tone)
                        
                        # Put the system into state |11>
                        pls.reset_phase(T, [control_port_A, control_port_B])
                        pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
                        T += control_duration_01
                        
                        # Apply the CZ20 gate, with parameters being swept.
                        pls.reset_phase(T, coupler_ac_port)
                        pls.output_pulse(T, coupler_ac_pulse_cz20)
                        T += coupler_ac_duration_cz20
                        
                        # Commence multiplexed readout
                        pls.reset_phase(T, readout_stimulus_port)
                        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                        pls.store(T + readout_sampling_delay) # Sampling window
                        T += readout_duration
                        
                        # Move to next scanned frequency
                        pls.next_frequency(T, coupler_ac_port, group = 0)
                        
                        # Await a new repetition, after which a new coupler DC bias tone
                        # will be added - and a new frequency set for the readout tone.
                        T += repetition_delay
                    
                    
                    # Increment the CZ20 pulse amplitude.
                    pls.next_scale(T, coupler_ac_port, group = 0)
                
                    # Move to next iteration.
                    T += repetition_delay
                
                
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
                
                # We finished a run! Increment the counter.
                we_have_completed_this_many_runs += 1
                # Are we in fact done?
                if we_have_completed_this_many_runs == there_will_be_this_many_runs:
                    # We are!
                    measurement_finished = True
                
                # Create a larger array, and/or store the previous run in it.
                try:
                    if assembled_data_maximus == [None]:
                        time_vector, assembled_data_maximus = pls.get_store_data()
                except ValueError:
                    # Then there is something more in there.
                    dummy, temp_fetched_data = pls.get_store_data()
                    # Append newly-fetched data!
                    try:
                        assembled_data_maximus = np.append(assembled_data_maximus, temp_fetched_data, axis = 0)
                    except MemoryError:
                        # You're out of memory. Abort measurement.
                        # TODO  In the future, how about post-processing the
                        #       data already at this point? In order not to
                        #       store as mush data in memory.
                        print("WARNING: Out of memory, unable to allocate more. Aborting measurement and exporting data file.")
                        measurement_finished = True
                        num_amplitudes = len(assembled_data_maximus)//(num_freqs*num_single_shots)
        
        except RuntimeError as e:
            
            # TODO! This is a bit of a DEBUG catcher.
            # For now, I've assumed that there can only be one re-scaling.
            # Likely, this might be untrue.
            if (not measurement_was_cut_into_segments):
                measurement_was_cut_into_segments = True
            else:
                raise Exception("Error: there is currently an assumption in this script that there can only be one instance of segmenting the measurement run. But a RuntimeError triggered twice, which is currently not a handled exception. TODO!")
            
            # The assigned measurement is too big.
            # It will be performed in measurement segments.
            print("The assigned measurement is too big. It will be performed in measurement segments.")
            
            # Let's process the error.
            e = (str(e).lower()).replace("trying to use ","")
            e = e.replace(" commands, maximum is ",",")
            e = tuple(map(int, e.split(',')))
            
            # Update how long to make the amplitude sweep arrays.
            num_amplitudes_per_run = int( num_amplitudes // (e[0]/e[1]) )
            
            # How low will the last run be?
            length_of_last_run = int(num_amplitudes % num_amplitudes_per_run)
            
            # And how many runs will there be in total?
            there_will_be_this_many_runs = np.ceil(num_amplitudes / num_amplitudes_per_run)
            
            # Let's use the time printer, since this will take a while.
            use_time_printer = True
            
        
        if (use_time_printer) and (we_have_completed_this_many_runs > 0):
            tock = time.time() # Get a time estimate.
            num_tick_tocks += 1
            total_dur += (tock - tick)
            average_duration_per_point = total_dur / num_tick_tocks
            calc = ((there_will_be_this_many_runs)-num_tick_tocks)*average_duration_per_point
            if (calc != 0.0):
                # Print "true" time remaining.
                show_user_time_remaining(calc)
        
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        ## time_vector, fetched_data_arr = pls.get_store_data()
        fetched_data_arr = assembled_data_maximus
        
        print("Saving data")
        
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
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco',"Hz",
            
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
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_duration_cz20', "s",
            'coupler_ac_freq_cz20_nco', "Hz",
            'coupler_ac_freq_cz20_centre_if', "Hz",
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
                    hdf5_logs.append('Probability for state |'+statep+'>')
                    # Let the record show that I wanted to write a Unicode ket
                    # instead of the '>' character, but the Log Browser's
                    # support for anything non-bland is erratic at best.
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
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [readout_freq_if_A, readout_freq_if_B],
            
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
            append_to_log_name_before_timestamp = 'state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return"""
    
"""def cz20_sweep_amplitude_and_detuning_for_t_half_state_probability(
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
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_rising_edge_time_cz20,
    coupler_ac_plateau_duration_maximise_cz20,
    coupler_ac_freq_cz20_nco,
    coupler_ac_freq_cz20_centre_if,
    coupler_ac_freq_cz20_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
    num_single_shots,
    resonator_ids,
    states_to_discriminate_between = ['20'],
    
    use_log_browser_database = True,
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
        
        The readout is multiplexed between two pairwise-coupled transmons.
        The fetched result will be sent into a state discriminator.
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    ## Initial array declaration
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
    ## The Presto device will run out of commands when running this script.
    num_dividend  = 1
    curr_dividend = 1
    success = False
    attempts = 0
    max_attempts = 5
    while ((not success) and (attempts < max_attempts)):
        # Let's remake the needed ranges to not halt the measurement.
        # Example: if curr_dividend is 3, and num_dividend = 10, then
        # we want to chop up some "stepping array" into 10 pieces,
        # and run the third entry now.
        coupler_ac_amp_arr_divided = np.split(coupler_ac_amp_arr)[curr_dividend -1]
        
        try:
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
                coupler_ac_rising_edge_time_cz20 = int(round(coupler_ac_rising_edge_time_cz20 / plo_clk_T)) * plo_clk_T
                coupler_ac_plateau_duration_maximise_cz20 = int(round(coupler_ac_plateau_duration_maximise_cz20 / plo_clk_T)) * plo_clk_T
                
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
                    sync      = False,
                )
                # Control port mixers
                pls.hardware.configure_mixer(
                    freq      = control_freq_01_A,
                    out_ports = control_port_A,
                    sync      = False,
                )
                pls.hardware.configure_mixer(
                    freq      = control_freq_01_B,
                    out_ports = control_port_B,
                    sync      = False,
                )
                # Coupler port mixer
                pls.hardware.configure_mixer(
                    freq      = coupler_ac_freq_cz20_nco,
                    out_ports = coupler_ac_port,
                    sync      = (coupler_dc_port == []),
                )
                if coupler_dc_port != []:
                    pls.hardware.configure_mixer(
                        freq      = 0.0,
                        out_ports = coupler_dc_port,
                        sync      = True,  # Sync here
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
                    scales          = coupler_ac_amp_arr_divided,##coupler_ac_amp_arr, # This value will be swept!
                )
                if coupler_dc_port != []:
                    pls.setup_scale_lut(
                        output_ports    = coupler_dc_port,
                        group           = 0,
                        scales          = coupler_dc_bias,
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
                readout_freq_if_A = np.abs(readout_freq_nco - readout_freq_A)
                readout_freq_if_B = np.abs(readout_freq_nco - readout_freq_B)
                pls.setup_freq_lut(
                    output_ports = readout_stimulus_port,
                    group        = 0,
                    frequencies  = readout_freq_if_A,
                    phases       = np.full_like(readout_freq_if_A, 0.0),
                    phases_q     = np.full_like(readout_freq_if_A, -np.pi/2), # USB !  ##+np.pi/2, # LSB
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
                
                # Setup control_pulse_pi_01 carrier tones, considering that there are digital mixers.
                pls.setup_freq_lut(
                    output_ports = control_port_A,
                    group        = 0,
                    frequencies  = 0.0,
                    phases       = 0.0,
                    phases_q     = 0.0,
                )
                pls.setup_freq_lut(
                    output_ports = control_port_B,
                    group        = 0,
                    frequencies  = 0.0,
                    phases       = 0.0,
                    phases_q     = 0.0,
                )
                
                
                ### Setup the CZ20 gate pulse
                
                # The initially set duration will not be swept by the sequencer
                # program.
                coupler_ac_duration_cz20 = \
                    2 * coupler_ac_rising_edge_time_cz20 + \
                    coupler_ac_plateau_duration_maximise_cz20
                coupler_ac_pulse_cz20 = pls.setup_long_drive(
                    output_port = coupler_ac_port,
                    group       = 0,
                    duration    = coupler_ac_duration_cz20,
                    amplitude   = 1.0,
                    amplitude_q = 1.0,
                    rise_time   = coupler_ac_rising_edge_time_cz20,
                    fall_time   = coupler_ac_rising_edge_time_cz20,
                )
                
                # Setup the cz20 pulse carrier, this tone will be swept in frequency.
                # Since we set the mixer to some NCO value, we probably want to use
                # the lower sideband for sweeping the span (not the upper).
                f_start = coupler_ac_freq_cz20_centre_if - coupler_ac_freq_cz20_span / 2
                f_stop = coupler_ac_freq_cz20_centre_if + coupler_ac_freq_cz20_span / 2
                coupler_ac_freq_cz20_if_arr = np.linspace(f_start, f_stop, num_freqs)
                
                # Use the lower sideband. Note the minus sign.
                coupler_ac_pulse_cz20_freq_arr = coupler_ac_freq_cz20_nco - coupler_ac_freq_cz20_if_arr
                
                # Setup LUT
                pls.setup_freq_lut(
                    output_ports    = coupler_ac_port,
                    group           = 0,
                    frequencies     = coupler_ac_freq_cz20_if_arr,
                    phases          = np.full_like(coupler_ac_freq_cz20_if_arr, 0.0),
                    phases_q        = np.full_like(coupler_ac_freq_cz20_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
                
                    # Redefine the coupler DC pulse duration for repeated playback
                    # once one tee risetime has passed.
                    coupler_bias_tone.set_total_duration(
                        control_duration_01 + \
                        coupler_ac_duration_cz20 + \
                        readout_duration + \
                        repetition_delay \
                    )
                
                # For every AC coupler amplitude to sweep over:
                for jj in range(num_amplitudes / num_dividend):
                
                    # For every resonator stimulus pulse frequency to sweep over:
                    for ii in range(num_freqs):

                        # Re-apply the coupler bias tone.
                        if coupler_dc_port != []:
                            pls.output_pulse(T, coupler_bias_tone)
                        
                        # Put the system into state |11>
                        pls.reset_phase(T, [control_port_A, control_port_B])
                        pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
                        T += control_duration_01
                        
                        # Apply the CZ20 gate, with parameters being swept.
                        pls.reset_phase(T, coupler_ac_port)
                        pls.output_pulse(T, coupler_ac_pulse_cz20)
                        T += coupler_ac_duration_cz20
                        
                        # Commence multiplexed readout
                        pls.reset_phase(T, readout_stimulus_port)
                        pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                        pls.store(T + readout_sampling_delay) # Sampling window
                        T += readout_duration
                        
                        # Move to next scanned frequency
                        pls.next_frequency(T, coupler_ac_port, group = 0)
                        
                        # Await a new repetition, after which a new coupler DC bias tone
                        # will be added - and a new frequency set for the readout tone.
                        T += repetition_delay
                    
                    
                    # Increment the CZ20 pulse amplitude.
                    pls.next_scale(T, coupler_ac_port, group = 0)
                
                    # Move to next iteration.
                    T += repetition_delay
                
                
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
                
            # Store intermediary results
            if not pls.dry_run:
                time_vector, fetched_data_arr = pls.get_store_data()
                assert 1 == 0, 
            
            # Set the success flag to true? signalling a completed measurement.
            if curr_dividend == num_dividend
                # Then we are done. Assemble data and close the measurement.
                ## TODO HAER
                success = True
            else:
                # Then increment.
                curr_dividend += 1
        
        except RuntimeError as e:
            attempts += 1
            
            # Let's process the error.
            e = (str(e).lower()).replace("trying to use ","")
            e = e.replace(" commands, maximum is ",",")
            e = tuple(map(int, e.split(',')))
            num_dividend = np.ceil(e[0]/e[1])
            
            # ... and let's remake the problematic array so that we can
            # use numpy easily in the beginning of the while loop.
            num_amplitudes = np.floor( num_amplitudes / num_dividend ) * num_dividend
    
    # Guarantee that things worked.
    if (not success):
        # Then the while loop somehow failed.
        raise RuntimeError("Critical error: could not remake script into a runnable format. Likely, something too big for the instrument to process.")
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        ##time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
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
            'readout_duration', "s",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco',"Hz",
            
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
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_duration_cz20', "s",
            'coupler_ac_freq_cz20_nco', "Hz",
            'coupler_ac_freq_cz20_centre_if', "Hz",
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
                    hdf5_logs.append('Probability for state |'+statep+'>')
                    # Let the record show that I wanted to write a Unicode ket
                    # instead of the '>' character, but the Log Browser's
                    # support for anything non-bland is erratic at best.
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
        string_arr_to_return += save(
            timestamp = get_timestamp_string(),
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_vector = time_vector,
            fetched_data_arr = fetched_data_arr,
            fetched_data_scale = axes['y_scaler'],
            fetched_data_offset = axes['y_offset'],
            resonator_freq_if_arrays_to_fft = [readout_freq_if_A, readout_freq_if_B],
            
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
            append_to_log_name_before_timestamp = 'state_probability',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return"""