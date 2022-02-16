#####################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/upload/main
#####################################################################################

from presto import pulsed
from presto.utils import sin2
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

import os
import sys
import time
import h5py
import Labber
import shutil
import numpy as np
from datetime import datetime
from presto.utils import rotate_opt
from scipy.optimize import curve_fit


def ramsey01_readout0(
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
    control_freq_01_nco,
    control_freq_01_center_if,
    control_freq_01_span,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_freqs,
    num_averages,
    
    num_delays,
    dt_per_ramsey_iteration,
    
    save_complex_data = False,
    use_log_browser_database = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": 1.0,
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Perform a Rasey spectroscopy on a given qubit with a connected
        resonator.
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
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
    
    # Declare time delay array for saving time data.
    delay_arr = np.linspace(0.0, (num_delays * dt_per_ramsey_iteration), num_delays)
    
    
    # Instantiate the interface
    print("\nInstantiating interface!")
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
        
        # Readout output and input ports
        pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        
        # Control port
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
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
        dt_per_ramsey_iteration = int(round(dt_per_ramsey_iteration / plo_clk_T)) * plo_clk_T
        
        
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixer
        pls.hardware.configure_mixer(
            freq      = control_freq_01_nco,
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
            output_port = readout_stimulus_port,
            group       = 0,
            duration    = readout_duration,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0
        )
        
        
        ### Setup pulse "control_pulse_pi_01" ###

        # Setup control_pulse_pi_01 pulse envelope.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01_half = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01 / 2, # Note!
            template_q  = control_envelope_01 / 2, # Halved.
            envelope    = True,
        )
        
        # Setup control pulse carrier, this tone will be swept in frequency.
        f_start = control_freq_01_center_if - control_freq_01_span / 2
        f_stop = control_freq_01_center_if + control_freq_01_span / 2
        control_freq_01_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        control_pulse_01_half_freq_arr = control_freq_01_nco - control_freq_01_if_arr

        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = control_port,
            group           = 0,
            frequencies     = control_freq_01_if_arr,
            phases          = np.full_like(control_freq_01_if_arr, 0.0),
            phases_q        = np.full_like(control_freq_01_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
        
        # For every delay to step through:
        for ii in range(num_delays):
            
            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
            if coupler_dc_port != []:
                for bias_tone in coupler_bias_tone:    
                    bias_tone.set_total_duration(
                        control_duration_01 + \
                        ii * dt_per_ramsey_iteration + \
                        control_duration_01 + \
                        readout_duration + \
                        repetition_delay \
                    )
            
                # Re-apply the coupler bias tone.
                pls.output_pulse(T, coupler_bias_tone)
            
            # Apply the frequency-swept pi01 pulses.
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse_pi_01_half)
            T += control_duration_01
            
            # Await some amount of time between pulses.
            T += ii * dt_per_ramsey_iteration
            
            # Apply the last pi_01_half pulse.
            pls.output_pulse(T, control_pulse_pi_01_half)
            T += control_duration_01
            
            # Commence readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Wait for decay
            T += repetition_delay
        
        # Move to next scanned frequency
        pls.next_frequency(T, control_port)
        
        # Move to next iteration.
        T += repetition_delay
        
        
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
        
    
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()

        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
            'delay_arr', "s",
            'control_pulse_01_half_freq_arr', "Hz",
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
            'control_freq_01_nco', "Hz",
            'control_freq_01_center_if', "Hz",
            'control_freq_01_span', "Hz",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'num_freqs', "",
            'num_averages', "",
            
            'num_delays', "",
            'dt_per_ramsey_iteration', "s",
        ]
        hdf5_logs = [
            'fetched_data_arr',
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
            axes[axis] = axes[axis].replace('_23','₂₃')
            axes[axis] = axes[axis].replace('_0','₀')
            axes[axis] = axes[axis].replace('_1','₁')
            axes[axis] = axes[axis].replace('_2','₂')
            axes[axis] = axes[axis].replace('_3','₃')
        
        # Build step lists, re-scale and re-unit where necessary.
            ext_keys = []
            for ii in range(0,len(hdf5_steps),2):
                if (hdf5_steps[ii] != 'fetched_data_arr') and (hdf5_steps[ii] != 'time_matrix'):
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
                if (hdf5_singles[jj] != 'fetched_data_arr') and (hdf5_singles[jj] != 'time_matrix'):
                    temp_object = np.array( [eval(hdf5_singles[jj])] )
                    ext_keys.append(dict(name=hdf5_singles[jj], unit=hdf5_singles[jj+1], values=temp_object))
            log_dict_list = []
            for kk in range(0,len(hdf5_logs),2):
                log_entry_name = hdf5_logs[kk]
                temp_log_unit = hdf5_logs[kk+1]
                if (axes['y_name']).lower() != 'default':
                    # Replace the y-axis name
                    log_entry_name = axes['y_name']
                if axes['y_scaler'] != 1.0:
                    # Re-scale the y-axis
                    ## NOTE! Direct manipulation of the fetched_data_arr array!
                    fetched_data_arr *= axes['y_scaler']   
                if (axes['y_unit']).lower() != 'default':
                    # Change the unit on the y-axis
                    temp_log_unit = axes['y_unit']
                log_dict_list.append(dict(name=log_entry_name, unit=temp_log_unit, vector=False, complex=save_complex_data))
            
            # Save data!
            save(
                timestamp = timestamp,
                ext_keys = ext_keys,
                log_dict_list = log_dict_list,
                
                time_matrix = time_matrix,
                fetched_data_arr = fetched_data_arr,
                resonator_freq_if_arrays_to_fft = [readout_freq_if_A, readout_freq_if_B],
                
                path_to_script = os.path.realpath(__file__),
                use_log_browser_database = use_log_browser_database,
                
                integration_window_start = integration_window_start,
                integration_window_stop = integration_window_stop,
                inner_loop_size = num_delays,
                outer_loop_size = num_freqs,
                
                save_complex_data = save_complex_data,
                append_to_log_name_before_timestamp = '01_with_bias',
                append_to_log_name_after_timestamp  = '',
                select_resonator_for_single_log_export = '',
            )

def ramsey01_multiplexed_ro(
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
    control_freq_01_A_nco,
    control_freq_01_A_center_if,
    control_freq_01_A_span,
    control_port_B,
    control_amp_01_B,
    control_freq_01_B_nco,
    control_freq_01_B_center_if,
    control_freq_01_B_span,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_freqs,
    num_averages,
    
    num_delays,
    dt_per_ramsey_iteration,
    ):
    ''' Perform a Rasey spectroscopy on two qubits connected by a
        DC-tunable SQUID coupler. Readout is multiplexed on both qubits.
    '''
    
    # Declare time delay array for saving time data.
    delay_arr = np.linspace(0.0, (num_delays * dt_per_ramsey_iteration), num_delays)
    
    
    # Instantiate the interface
    print("\nInstantiating interface!")
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
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        
        # Make the user-set time variables representable
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        dt_per_ramsey_iteration = int(round(dt_per_ramsey_iteration / plo_clk_T)) * plo_clk_T
        
        
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
            freq      = control_freq_01_A_nco,
            out_ports = control_port_A,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_01_B_nco,
            out_ports = control_port_B,
            sync      = False,
        )
        # Coupler port mixer
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
        control_pulse_pi_01_A_half = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01 / 2, # Note!
            template_q  = control_envelope_01 / 2, # Halved.
            envelope    = True,
        )
        control_pulse_pi_01_B_half = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01 / 2, # Note!
            template_q  = control_envelope_01 / 2, # Halved.
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tones, considering that there are digital mixers.
        # These tones will be swept in frequency. The NCOs are set to f_01.
        f_start_A = control_freq_01_A_center_if - control_freq_01_A_span / 2
        f_start_B = control_freq_01_B_center_if - control_freq_01_B_span / 2
        f_stop_A = control_freq_01_A_center_if + control_freq_01_A_span / 2
        f_stop_B = control_freq_01_B_center_if + control_freq_01_B_span / 2
        control_freq_01_A_if_arr = np.linspace(f_start_A, f_stop_A, num_freqs)
        control_freq_01_B_if_arr = np.linspace(f_start_B, f_stop_B, num_freqs)
        
        # Use the lower sidebands. Note the minus sign.
        control_pulse_01_A_freq_arr = control_freq_01_A_nco - control_freq_01_A_if_arr
        control_pulse_01_B_freq_arr = control_freq_01_B_nco - control_freq_01_B_if_arr
        
        # Setup LUTs
        pls.setup_freq_lut(
            output_ports    = control_port_A,
            group           = 0,
            frequencies     = control_freq_01_A_if_arr,
            phases          = np.full_like(control_freq_01_A_if_arr, 0.0),
            phases_q        = np.full_like(control_freq_01_A_if_arr, +np.pi / 2),  # +pi/2 for LSB!
        )
        pls.setup_freq_lut(
            output_ports    = control_port_B,
            group           = 0,
            frequencies     = control_freq_01_B_if_arr,
            phases          = np.full_like(control_freq_01_B_if_arr, 0.0),
            phases_q        = np.full_like(control_freq_01_B_if_arr, +np.pi / 2),  # +pi/2 for LSB!
        )

        ### Setup pulse "coupler_bias_tone" ###

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
        pls.reset_phase(T, coupler_dc_port)
        pls.output_pulse(T, coupler_bias_tone)
        T += added_delay_for_bias_tee
        
        # For every delay to step through:
        for ii in range(num_delays):
            
            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
            for bias_tone in coupler_bias_tone:
                bias_tone.set_total_duration(
                    control_duration_01 + \
                    ii * dt_per_ramsey_iteration + \
                    control_duration_01 + \
                    readout_duration + \
                    repetition_delay \
                )
            
            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
            
            # Apply the frequency-swept pi01 pulses.
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_A_half, control_pulse_pi_01_B_half])
            T += control_duration_01
            
            # Await some amount of time between pulses.
            T += ii * dt_per_ramsey_iteration
            
            # Apply the last pi_01_half pulse.
            pls.output_pulse(T, [control_pulse_pi_01_A_half, control_pulse_pi_01_B_half] )
            T += control_duration_01
            
            # Commence readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Wait for decay
            T += repetition_delay
        
        # Move to next scanned frequency
        pls.next_frequency(T, [control_port_A, control_port_B])
        
        # Move to next iteration.
        T += repetition_delay
        
        
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
        
    
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()

        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # This save is done in a loop, due to quirks with Labber's log browser.
        arrays_in_loop = [
            'control_pulse_01_A_freq_arr',
            'control_pulse_01_B_freq_arr'
        ]
        for u in range(2):
        
            # Data to be stored.
            hdf5_steps = [
                'delay_arr', "s",
                arrays_in_loop[u], "Hz",
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
                'control_freq_01_A_nco', "Hz",
                'control_freq_01_A_center_if', "Hz",
                'control_freq_01_A_span', "Hz",
                'control_port_B', "",
                'control_amp_01_B', "FS",
                'control_freq_01_B_nco', "Hz",
                'control_freq_01_B_center_if', "Hz",
                'control_freq_01_B_span', "Hz",
                'control_duration_01', "s",
                
                #'coupler_dc_port', "",
                'coupler_dc_bias', "FS",
                'added_delay_for_bias_tee', "s",
                
                'num_freqs', "",
                'num_averages', "",
                
                'num_delays', "",
                'dt_per_ramsey_iteration', "s",
            ]
            hdf5_logs = [
                'fetched_data_arr',
            ]
            
            print("... building HDF5 keys.")
            
            # Assert that every key bears a corresponding unit entered above.
            number_of_keyed_elements_is_even = \
                ((len(hdf5_steps) % 2) == 0) and ((len(hdf5_singles) % 2) == 0)
            assert number_of_keyed_elements_is_even, "Error: non-even amount "  + \
                "of keys and units entered in the portion of the measurement "  + \
                "script that saves the data. Someone likely forgot a comma."
            
            # Build step lists and log lists
            ext_keys = []
            for ii in range(0,len(hdf5_steps),2):
                temp_object = np.array( eval(hdf5_steps[ii]) )
                ext_keys.append(dict(name=hdf5_steps[ii], unit=hdf5_steps[ii+1], values=temp_object))
            for jj in range(0,len(hdf5_singles),2):
                temp_object = np.array( [eval(hdf5_singles[jj])] )
                ext_keys.append(dict(name=hdf5_singles[jj], unit=hdf5_singles[jj+1], values=temp_object))
            log_dict_list = []
            for kk in range(0,len(hdf5_logs)):
                log_dict_list.append(dict(name=hdf5_logs[kk], vector=False))
            
            # Get name and time for logfile.
            path_to_script = os.path.realpath(__file__)  # Full path of current script
            current_dir, name_of_running_script = os.path.split(path_to_script)
            script_filename = os.path.splitext(name_of_running_script)[0]  # Name of current script
            timestamp = (datetime.now()).strftime("%d-%b-%Y_(%H_%M_%S)") # Date and time
            savefile_string = script_filename + '01_with_bias_'+str((u)+1)'+_of_2_' + timestamp + '.hdf5'  # Name of save file
            save_path = os.path.join(current_dir, "data", savefile_string)  # Full path of save file
            
            # Make logfile
            print("... making Log browser logfile.")
            f = Labber.createLogFile_ForData(savefile_string, log_dict_list, step_channels=ext_keys, use_database = False)
            
            # Set project name, tag, and user in logfile.
            f.setProject(script_filename)
            f.setTags('krizan')
            f.setUser('Christian Križan')
            
            print("... processing readout data.")
            
            # fetched_data_arr SHAPE: num_stores * repeat_count, num_ports, smpls_per_store
            integration_window_start = 1500 * 1e-9
            integration_window_stop  = 2000 * 1e-9
            
            # Get index corresponding to integration_window_start and integration_window_stop respectively
            integration_start_index = np.argmin(np.abs(time_matrix - integration_window_start))
            integration_stop_index = np.argmin(np.abs(time_matrix - integration_window_stop))
            integr_indices = np.arange(integration_start_index, integration_stop_index)
            
            ## Multiplexed readout
            
            # Acquire time step needed for returning the DFT sample frequencies.
            dt = time_matrix[1] - time_matrix[0]
            nr_samples = len(integr_indices)
            freq_arr = np.fft.fftfreq(nr_samples, dt)
            
            # Execute complex FFT.
            resp_fft = np.fft.fft(fetched_data_arr[:, 0, integr_indices], axis=-1) / len(integr_indices)
            
            # Get new indices for the new processing_arr arrays.
            integr_indices_1 = np.argmin(np.abs(freq_arr - readout_freq_if_A))
            integr_indices_2 = np.argmin(np.abs(freq_arr - readout_freq_if_B))
            
            # Build new processing_arr arrays.
            processing_arr_1 = 2 * resp_fft[:, integr_indices_1]
            processing_arr_2 = 2 * resp_fft[:, integr_indices_2]
            
            # Reshape the data to account for repeats.
            processing_arr_1.shape = (num_freqs, num_delays)
            processing_arr_2.shape = (num_freqs, num_delays)
            
            # Take the absolute value of the data.
            processing_arr_1 = np.abs(processing_arr_1)
            processing_arr_2 = np.abs(processing_arr_2)
            
            print("... storing processed data into the HDF5 file.")
            # TODO DEBUG This fit should not be here. The fit
            # routines as seen in the end of this file should
            # be a seperate script that takes Labber file data
            # as input, and performs the fit accordingly.
            if u == 0:
                for i in range(num_freqs):
                    f.addEntry( {"fetched_data_arr": processing_arr_1[i,:]} )
                # <FIT1 COULD GO HERE>
                
            else:
                for i in range(num_freqs):
                    f.addEntry( {"fetched_data_arr": processing_arr_2[i,:]} )
                # <FIT2 COULD GO HERE>
                
            # Check if the hdf5 file was created in the local directory.
            # If so, move it to the 'data' directory.
            if os.path.isfile(os.path.join(current_dir, savefile_string)):
                shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

            # Print final success message.
            print("Data saved, see " + save_path)
