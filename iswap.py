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

def iswap_sweep_duration_and_detuning(
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
    coupler_ac_amp,
    coupler_ac_freq_iswap_nco,
    coupler_ac_freq_iswap_centre_if,
    coupler_ac_freq_iswap_span,
    
    num_freqs,
    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    
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
    ''' Tune an iSWAP-interaction between two qubits using
        a tuneable coupler, by fixing the gate amplitude and gate bias.
        Thus, the gate duration and detuning is swept.
    '''
    
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
        
        
        ''' Make the user-set time variables representable '''
        
        ## dt_per_time_step is a bad idea and should be removed. This is likely already fixed to be honest. TODO
        ### Figure out a dt_per_time_step and make the value representable.
        ##dt_per_time_step = (coupler_ac_plateau_duration_iswap_max-coupler_ac_plateau_duration_iswap_min)/num_time_steps
        ##dt_per_time_step = int(round(dt_per_time_step / plo_clk_T)) * plo_clk_T
        
        # Generate an array for data storage. For all elements, round to the
        # programmable logic clock period. Then, remove duplicates and update
        # the num_time_steps parameter.
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
            freq      = coupler_ac_freq_iswap_nco,
            out_ports = coupler_ac_port,
            sync      = (coupler_dc_port == []),
        )
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
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
            scales          = coupler_ac_amp,
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
        # Since we set the mixer to some NCO value, we probably want to use
        # the lower sideband for sweeping the span (not the upper).
        f_start = coupler_ac_freq_iswap_centre_if - coupler_ac_freq_iswap_span / 2
        f_stop  = coupler_ac_freq_iswap_centre_if + coupler_ac_freq_iswap_span / 2
        coupler_ac_freq_iswap_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        coupler_ac_pulse_iswap_freq_arr = coupler_ac_freq_iswap_nco - coupler_ac_freq_iswap_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = coupler_ac_freq_iswap_if_arr,
            phases          = np.full_like(coupler_ac_freq_iswap_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_iswap_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
        
        # For every pulse duration to sweep over:
        ##for ii in range(dt_per_time_step): TODO I removed this, likely bugged, see Rabi_12 duration sweep
        for ii in iswap_total_pulse_duration_arr:

            ## # Redefine the iSWAP pulse's total duration,
            ## # resulting in stepping said duration in time.
            ## coupler_ac_duration_iswap = \
            ##     2 * coupler_ac_single_edge_time_iswap + \
            ##     coupler_ac_plateau_duration_iswap_min + \
            ##     ii * dt_per_time_step
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
            T += repetition_delay
        
        # Move to the next scanned frequency
        pls.next_frequency(T, coupler_ac_port, group = 0)
        
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
            'iswap_total_pulse_duration_arr',"s",
            'coupler_ac_pulse_iswap_freq_arr', "Hz",
        ]
        hdf5_singles = [
            'readout_stimulus_port',"",
            'readout_sampling_port',"",
            'readout_duration',"s",
            'readout_freq_A',"Hz",
            'readout_amp_A',"FS",
            'readout_freq_B',"Hz",
            'readout_amp_B',"FS",
            'sampling_duration',"s",
            'readout_sampling_delay',"s",
            'repetition_delay',"s",
            'control_port_A',"",
            'control_amp_01_A',"FS",
            'control_freq_01_A',"Hz",
            'control_port_B,',"",
            'control_amp_01_B',"FS",
            'control_freq_01_B',"Hz",
            'control_duration_01',"s",
            #'coupler_dc_port',"",
            'coupler_dc_bias',"FS",
            'added_delay_for_bias_tee',"s",
            'coupler_ac_port',"",
            'coupler_ac_amp',"FS",
            'coupler_ac_freq_iswap_nco',"Hz",
            'coupler_ac_freq_iswap_centre_if',"Hz",
            'coupler_ac_freq_iswap_span',"Hz",
            'num_freqs',"",
            'num_averages',"",
            'num_time_steps',"",
            'coupler_ac_single_edge_time_iswap',"s",
            'coupler_ac_plateau_duration_iswap_min',"s",
            'coupler_ac_plateau_duration_iswap_max',"s",
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
            inner_loop_size = num_time_steps,
            outer_loop_size = num_freqs,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'sweep_duration_and_detuning',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return
    

def iswap_sweep_duration_and_amplitude(
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
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    coupler_ac_freq_iswap,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,

    num_averages,
    num_amplitudes,
    num_time_steps,
    
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
    ''' Tune an iSWAP-interaction between two qubits, where it is known at
        what gate frequency the iSWAP interaction takes place (and with
        what coupler bias), but not the iSWAP gate amplitude nor the gate
        duration.
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
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_min = int(round(coupler_ac_plateau_duration_iswap_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_max = int(round(coupler_ac_plateau_duration_iswap_max / plo_clk_T)) * plo_clk_T
        
        
        ''' Make the user-set time variables representable '''
        
        ## # Figure out a dt_per_time_step and make the value representable.
        ## dt_per_time_step = (coupler_ac_plateau_duration_iswap_max-coupler_ac_plateau_duration_iswap_min)/num_time_steps
        ## dt_per_time_step = int(round(dt_per_time_step / plo_clk_T)) * plo_clk_T
        
        # Generate an array for data storage. For all elements, round to the
        # programmable logic clock period. Then, remove duplicates and update
        # the num_time_steps parameter.
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
            freq      = coupler_ac_freq_iswap, # Fixed value for this sweep.
            out_ports = coupler_ac_port,
            sync      = (coupler_dc_port == []),
        )
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
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
        
        # Setup the iSWAP pulse carrier. This tone will not be swept in
        # frequency.
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = 0.0,
            phases          = 0.0,
            phases_q        = 0.0, # TODO! See comment below.
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
        
        # For every resonator stimulus pulse frequency to sweep over:
        ##for ii in range(num_time_steps): TODO I removed this, dt_per_time_step is likely bugged. See Rabi_12 duration sweep.
        for ii in iswap_total_pulse_duration_arr:
        
            ## # Redefine the iSWAP pulse's total duration,
            ## # resulting in stepping said duration in time.
            ## coupler_ac_duration_iswap = \
            ##     2 * coupler_ac_single_edge_time_iswap + \
            ##     coupler_ac_plateau_duration_iswap_min + \
            ##     ii * dt_per_time_step
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
            T += repetition_delay
        
        # Increment the iSWAP pulse amplitude.
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
            'readout_freq_nco',"Hz",
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
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_iswap', "Hz",
            
            'num_amplitudes', "",
            'coupler_ac_amp_min', "FS",
            'coupler_ac_amp_max', "FS",

            'num_averages',"",
            'num_time_steps',"",
            'coupler_ac_single_edge_time_iswap',"s",
            'coupler_ac_plateau_duration_iswap_min',"s",
            'coupler_ac_plateau_duration_iswap_max',"s",
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
            inner_loop_size = num_time_steps,
            outer_loop_size = num_amplitudes,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'sweep_duration_and_amplitude',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return
    

def iswap_sweep_amplitude_and_detuning(
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
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    coupler_ac_freq_iswap_nco,
    coupler_ac_freq_iswap_centre_if,
    coupler_ac_freq_iswap_span,
    
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
    ''' Tune an iSWAP-interaction between two qubits using
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
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
        
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
            freq      = coupler_ac_freq_iswap_nco,
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
        # Since we set the mixer to some NCO value, we probably want to use
        # the lower sideband for sweeping the span (not the upper).
        f_start = coupler_ac_freq_iswap_centre_if - coupler_ac_freq_iswap_span / 2
        f_stop = coupler_ac_freq_iswap_centre_if + coupler_ac_freq_iswap_span / 2
        coupler_ac_freq_iswap_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        coupler_ac_pulse_iswap_freq_arr = coupler_ac_freq_iswap_nco - coupler_ac_freq_iswap_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = coupler_ac_freq_iswap_if_arr,
            phases          = np.full_like(coupler_ac_freq_iswap_if_arr, 0.0),
            phases_q        = np.full_like(coupler_ac_freq_iswap_if_arr, +np.pi / 2),  # +pi/2 for LSB!
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
                coupler_ac_duration_iswap + \
                readout_duration + \
                repetition_delay \
            )
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):

            # Re-apply the coupler bias tone.
            if coupler_dc_port != []:
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
            
            # Move to next scanned frequency
            pls.next_frequency(T, coupler_ac_port, group = 0)
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        
        # Increment the iSWAP pulse amplitude.
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
            'coupler_ac_duration_iswap', "s",
            'coupler_ac_freq_iswap_nco', "Hz",
            'coupler_ac_freq_iswap_centre_if', "Hz",
            'coupler_ac_freq_iswap_span', "Hz",
            
            'coupler_ac_amp_min', "FS",
            'coupler_ac_amp_max', "FS",
            
            'num_freqs', "",
            'num_averages', "",
            'num_amplitudes', "",
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
            append_to_log_name_before_timestamp = 'sweep_amplitude_and_detuning',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return
    
    
def tune_local_qubit_phases_of_iswap(
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
    coupler_ac_amp_iswap,
    coupler_ac_freq_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,

    num_averages,
    num_phases,
    
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
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
    ''' Tune the phases of the individual qubits partaking in an iSWAP gate.
        Based on the methods from:
        
        M. Ganzhorn et al. PR Research 2 033447 (2020)
        https://doi.org/10.48550/arXiv.2005.05696
        
        ... with theoretical assistance found in:
        
        Deanna M. Abrams et al. Nature electronics, 3, December 2020 pp.744–750
        https://doi.org/10.48550/arXiv.1912.04424
        
        The method is similar to a Ramsey sequence.
        The input state is prepared to |+0⟩ (or |0+⟩).
        Then, we run an iSWAP, followed by an iSWAP†.
        We then put |+0⟩ (or |0+⟩) back to |10⟩ (or |01⟩).
        And then readout in Z.
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    # Adjust the phase _min and _max parameters.
    phase_sweep_rad_min = phase_sweep_rad_min % (2*np.pi)
    phase_sweep_rad_max = phase_sweep_rad_max % (2*np.pi)
    if phase_sweep_rad_max == 0.0:
        phase_sweep_rad_max = (2*np.pi)
    
    
    ## Initial array declaration
    
    # Declare phase array for the last pi/2 to be swept
    control_phase_arr = np.linspace(phase_sweep_rad_min, phase_sweep_rad_max, num_phases)
    
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
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
        
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
            freq      = coupler_ac_freq_iswap, # Fixed value for this sweep.
            out_ports = coupler_ac_port,
            sync      = (coupler_dc_port == []),
        )
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
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
            scales          = control_amp_01_A,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 0,
            scales          = control_amp_01_B,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 1,
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
        readout_freq_if_A = np.abs(readout_freq_nco - readout_freq_A)
        readout_freq_if_B = np.abs(readout_freq_nco - readout_freq_B)
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_A,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!  ##+np.pi/2, # LSB
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = readout_freq_if_B,
            phases       = 0.0,
            phases_q     = -np.pi/2, # USB!  ##+np.pi/2, # LSB
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
        control_pulse_pi_01_half_phase_swept_A = pls.setup_template(
            output_port = control_port_A,
            group       = 1,
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
        control_pulse_pi_01_half_phase_swept_B = pls.setup_template(
            output_port = control_port_B,
            group       = 1,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tones, considering that there are digital mixers.
        # Two sets are used. One "normal" LUT for the "normal" pi_half-pulses.
        # And, a phase-shifted LUT for the phase sweep.
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 1,
            frequencies  = np.full_like(control_phase_arr, 0.0),
            phases       = control_phase_arr,
            phases_q     = control_phase_arr,
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 1,
            frequencies  = np.full_like(control_phase_arr, 0.0),
            phases       = control_phase_arr,
            phases_q     = control_phase_arr,
        )
        
        
        ### Setup the iSWAP gate pulses
        
        # There will be a "normal" iSWAP pulse, followed by an iSWAP†
        coupler_ac_duration_iswap = coupler_ac_plateau_duration_iswap + \
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
        
        
        # Setup the iSWAP pulse carrier. This tone will not be swept in
        # frequency.
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = 0.0,
            phases          = 0.0,
            phases_q        = 0.0,
        )
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = 0.0,
            phases          = np.pi, # 180 degree inversion
            phases_q        = np.pi, # 180 degree inversion
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
            
            # Redefine the coupler bias tone duration.
            '''Note! The second coupler_ac_duration_iswap row is not a bug.'''
            coupler_bias_tone.set_total_duration(
                control_duration_01 + \
                coupler_ac_duration_iswap + \
                coupler_ac_duration_iswap + \
                control_duration_01 + \
                readout_duration + \
                repetition_delay \
            )
        
        # For every phase value of the final pi-half gate:
        for ii in range(num_phases):
            
            # Re-apply the coupler bias tone, to keep on playing once
            # the bias tee has charged.
            if coupler_dc_port != []:
                pls.output_pulse(T, coupler_bias_tone)
            
            # Put the system into state |+0> or |0+> with pi01_half pulses.
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_half_A, control_pulse_pi_01_half_B])
            T += control_duration_01    
            
            # Apply the first, normal, iSWAP gate.
            pls.reset_phase(T, coupler_ac_port)
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            
            # Apply the second, inverted, iSWAP gate.
            pls.output_pulse(T, coupler_ac_pulse_iswap_inverted)
            T += coupler_ac_duration_iswap
            
            # Put the system into state |10> or |01>, with some phase error.
            pls.output_pulse(T, [control_pulse_pi_01_half_phase_swept_A, control_pulse_pi_01_half_phase_swept_B])
            T += control_duration_01 
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            ''' Do note: having to use a bias-tee is an extreme limitation.
                If the Presto biasing card was working, I could have set the
                phase sweep in the "sequence execution" instead of having to
                loop over the phases in the run.
            '''
            
            # Move to next phase in the sweep
            pls.next_frequency(T, [control_port_A, control_port_B], group = 1)
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        ## Not needed, since everything is executed in one iteration.
        ## # Move to next iteration.
        ## T += repetition_delay
        
        
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
        
        # Data to be stored.
        hdf5_steps = [
            'control_phase_arr', "rad",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_freq_nco',"Hz",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
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
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_freq_iswap', "Hz",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap', "s",
            
            'num_averages', "",
            'num_phases', "",
            
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
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
            inner_loop_size = num_phases,
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'tune_local_qubit_phase',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
        )
    
    return string_arr_to_return
    
    
    
    
    
def tune_coupler_drive_phase_of_iswap():
    ''' Tune the phase of the iSWAP coupler drive ("AC flux driveline")
        that applies an iSWAP gate between two pairwise-coupled qubits.
        Based on the methods from:
        
        M. Ganzhorn et al. PR Research 2 033447 (2020)
        https://doi.org/10.48550/arXiv.2005.05696
        
        ... with theoretical assistance found in:
        
        Deanna M. Abrams et al. Nature electronics, 3, December 2020 pp.744–750
        https://doi.org/10.48550/arXiv.1912.04424
    '''
    assert 1 == 0, ("The missing pieces: https://www.nature.com/articles/s41928-020-00498-1 " +\
            "and Ganzhorn PR Research 2 033447 (2020). The idea is to tune the iSWAP's "+\
            "individual qubit phases. Like a Ramsey in a way. With, and without "+\
            "iSWAP _tensor_ iSWAP-dagger. Where, iSWAP-dagger is an iSWAP 180degrees "+\
            "out of phase. Same as \"minusing\" the amplitude, aka. -A. BUT: "+\
            "every iSWAP must execute with the iSWAP coupler drive being (phi1 - phi2) degrees. "+\
            "Aka. the coupler drive must always update to the difference-phase between "+\
            "the qubits. The Ganzhorn paper has the methods. After one has tuned the "+\
            "phases, one must also run iRB to optimise a phase offset "+\
            "(added onto the coupler drive I think, check paper) "+\
            "vs. fidelity. So it's not necessarily so that the best fidelity comes "+\
            "from running the \"phase optimal\" values.")
    