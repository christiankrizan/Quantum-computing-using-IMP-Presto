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

import log_browser_exporter


def find_f_ro01_sweep_coupler(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_center_if,
    readout_freq_span,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_freqs,
    num_averages,
    
    num_biases,
    coupler_bias_min = -1.0,
    coupler_bias_max = +1.0,
    
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
    ''' Find the optimal readout frequency for reading out qubits in state |0>,
        as a function of a swept pairwise coupler bias.
    '''
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
    print("Instantiating interface")
    
    # Instantiate the interface
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
        
        # Coupler port
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,   # readout_freq_nco is set as the mixer NCO
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False, # Sync at next call
        )
        # Coupler port mixer
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
        # Coupler bias amplitude (to be swept)
        pls.setup_scale_lut(
            output_ports    = coupler_dc_port,
            group           = 0,
            scales          = coupler_amp_arr,
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

        # Setup readout carrier, this tone will be swept in frequency.
        # The user provides an intended span.
        f_start = readout_freq_center_if - readout_freq_span / 2
        f_stop  = readout_freq_center_if + readout_freq_span / 2
        readout_freq_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        readout_pulse_freq_arr = readout_freq_nco - readout_freq_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_arr,
            phases       = np.full_like(readout_freq_if_arr, 0.0),
            phases_q     = np.full_like(readout_freq_if_arr, +np.pi / 2), # +pi/2 = LSB
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
        
        # Redefine the coupler DC pulse duration for repeated playback
        # once one tee risetime has passed.
        for bias_tone in coupler_bias_tone:
            bias_tone.set_total_duration(readout_duration + repetition_delay)
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):

            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
            
            # Commence readout, swept in frequency.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next scanned frequency
            pls.next_frequency(T, readout_stimulus_port)
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        # Increment the coupler port's DC bias.
        pls.next_scale(T, coupler_dc_port)
        
        # Move to next iteration.
        T += repetition_delay


        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################

        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   num_biases,
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
            'readout_pulse_freq_arr', "Hz",
            'coupler_amp_arr', "FS",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_nco', "Hz",
            'readout_freq_center_if', "Hz",
            'readout_freq_span', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s", 
            #'coupler_dc_port', "",
            'added_delay_for_bias_tee', "s",
            'num_freqs', "",
            'num_averages', "",
            'num_biases', "",
            'coupler_bias_min', "FS",
            'coupler_bias_max', "FS",
        ]
        hdf5_logs = [
            'fetched_data_arr', "FS",
        ]
        
        # Assert that the received keys bear (an even number of) entries,
        # implying whether a unit is missing.
        number_of_keyed_elements_is_even = \
            ((len(hdf5_steps) % 2) == 0) and \
            ((len(hdf5_singles) % 2) == 0) and \
            ((len(hdf5_logs) % 2) == 0)
        assert number_of_keyed_elements_is_even, "Error: non-even amount "  + \
            "of keys and units provided. Someone likely forgot a comma."
        
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
        log_browser_exporter.save(
            ext_keys = ext_keys,
            log_dict_list = log_dict_list,
            
            time_matrix = time_matrix,
            fetched_data_arr = fetched_data_arr,
            resonator_freq_if_arrays_to_fft = [],
            axes = axes,
            
            path_to_script = os.path.realpath(__file__),
            use_log_browser_database = use_log_browser_database,
            
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            inner_loop_size = num_freqs,
            outer_loop_size = num_biases,
            
            save_complex_data = save_complex_data,
            append_to_log_name_before_timestamp = '',
            append_to_log_name_after_timestamp  = '',
        )

 
def find_f_ro01_sweep_power(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_center_if,
    readout_freq_span,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,

    num_freqs,
    num_averages,
    
    num_amplitudes,
    readout_amp_min = -1.0,
    readout_amp_max = +1.0,
    save_complex_data = False,
    ):
    ''' Plot the readout frequency versus swept readout amplitude, pulsed.
    '''
    
    # Declare amplitude array for sweeping the power
    readout_amp_arr = np.linspace(readout_amp_min, readout_amp_max, num_amplitudes)
    
    print("Instantiating interface")
    
    # Instantiate the interface
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
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,   # readout_freq_nco is set as the mixer NCO
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = True, # Sync here
        )
        
        
        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp_arr,
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

        # Setup readout carrier, this tone will be swept in frequency.
        # The user provides an intended span.
        f_start = readout_freq_center_if - readout_freq_span / 2
        f_stop  = readout_freq_center_if + readout_freq_span / 2
        readout_freq_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        readout_pulse_freq_arr = readout_freq_nco - readout_freq_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_arr,
            phases       = np.full_like(readout_freq_if_arr, 0.0),
            phases_q     = np.full_like(readout_freq_if_arr, +np.pi / 2), # +pi/2 = LSB
        )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################

        # Start of sequence
        T = 0.0  # s
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):
            
            # Commence readout, swept in frequency.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next scanned frequency
            pls.next_frequency(T, readout_stimulus_port)
            
            # Wait for decay
            T += repetition_delay
            
        # Move to next amplitude
        pls.next_scale(T, readout_stimulus_port)
        
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
        
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")        
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
            'readout_pulse_freq_arr', "Hz",
            'readout_amp_arr', "FS",
        ]
        hdf5_singles = [
            'readout_stimulus_port', "",
            'readout_sampling_port', "",
            'readout_freq_nco', "Hz",
            'readout_freq_center_if', "Hz",
            'readout_freq_span', "Hz",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",

            'num_freqs', "",
            'num_amplitudes', "",
            'num_averages', "",
            
            'readout_amp_min', "FS",
            'readout_amp_max', "FS",
        ]
        
        # Assert that every key bears a corresponding unit entered above.
        number_of_keyed_elements_is_even = \
            ((len(hdf5_steps) % 2) == 0) and ((len(hdf5_singles) % 2) == 0)
        assert number_of_keyed_elements_is_even, "Error: non-even amount "  + \
            "of keys and units entered in the portion of the measurement "  + \
            "script that saves the data. Someone likely forgot a comma."
        
        # Build step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            if (hdf5_steps[ii] != 'fetched_data_arr') and (hdf5_steps[ii] != 'time_matrix'):
                temp_object = np.array( eval(hdf5_steps[ii]) )
                ext_keys.append(dict(name=hdf5_steps[ii], unit=hdf5_steps[ii+1], values=temp_object))
        for jj in range(0,len(hdf5_singles),2):
            if (hdf5_singles[jj] != 'fetched_data_arr') and (hdf5_singles[jj] != 'time_matrix'):
                temp_object = np.array( [eval(hdf5_singles[jj])] )
                ext_keys.append(dict(name=hdf5_singles[jj], unit=hdf5_singles[jj+1], values=temp_object))
        
        # Get name and time for logfile.
        path_to_script = os.path.realpath(__file__)  # Full path of current script
        current_dir, name_of_running_script = os.path.split(path_to_script)
        script_filename = os.path.splitext(name_of_running_script)[0]  # Name of current script
        timestamp = (datetime.now()).strftime("%d-%b-%Y_(%H_%M_%S)") # Date and time
        savefile_string = script_filename + '_power_sweep_' + timestamp + '.hdf5'  # Name of save file
        save_path = os.path.join(current_dir, "data", savefile_string)  # Full path of save file
        
        # Make logfile
        log_dict_list = [dict(name='fetched_data_arr', vector=False, complex=save_complex_data)]# , dict(name='time_matrix', vector=True)]
        f = Labber.createLogFile_ForData(savefile_string, log_dict_list, step_channels=ext_keys, use_database = False)
        
        # Set project name, tag, and user in logfile.
        f.setProject(script_filename)
        f.setTags('krizan')
        f.setUser('Christian Križan')
        
        # Split fetched_data_arr into repeats:
        # fetched_data_arr SHAPE: num_stores * repeat_count, num_ports, smpls_per_store
        integration_window_start = 1500 * 1e-9
        integration_window_stop  = 2000 * 1e-9
        t_span = integration_window_stop - integration_window_start
        
        # Get index corresponding to integration_window_start and integration_window_stop respectively
        integration_start_index = np.argmin(np.abs(time_matrix - integration_window_start))
        integration_stop_index = np.argmin(np.abs(time_matrix - integration_window_stop))
        integr_indices = np.arange(integration_start_index, integration_stop_index)
        
        # Construct a matrix, where every row is an integrated sampling
        # sequence corresponding to exactly one bias point.
        
        if(save_complex_data):
            angles = np.angle(fetched_data_arr[:, 0, integr_indices], deg=False)
            rows, cols = np.shape(angles)
            for row in range(rows):
                for col in range(cols):
                    #if angles[row][col] < 0.0:
                    angles[row][col] += (2.0 * np.pi)
            angles_mean = np.mean(angles, axis=-1)
            mean_len = len(angles_mean)
            for row in range(mean_len):
                angles_mean[row] -= (2.0 * np.pi)
            processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * angles_mean)
            ##processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(angles, axis=-1))
            ## processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(np.angle(fetched_data_arr[:, 0, integr_indices]), axis=-1))
        else:
            processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1)
        
        # Reshape depending on the repeat variable, as well as the inner loop
        # of the sequencer program.
        processing_arr.shape = (num_amplitudes, num_freqs)
        
        for i in range(num_amplitudes):
            f.addEntry( {"fetched_data_arr": processing_arr[i,:]} )
        # TODO: "time_matrix does not exist."
        #f.addEntry( {"time_matrix": time_matrix} )
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)
        
        
def find_f_ro12_sweep_coupler(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_nco,
    readout_freq_center_if,
    readout_freq_span,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_freqs,
    num_averages,
    
    num_biases,
    coupler_bias_min = -1.0,
    coupler_bias_max = +1.0,
    save_complex_data = False,
    ):
    ''' Find the optimal readout frequency for reading out qubits in state |1>,
        as a function of a swept pairwise coupler bias.
    '''
    
    # Declare amplitude array for the coupler to be swept
    coupler_amp_arr = np.linspace(coupler_bias_min, coupler_bias_max, num_biases)
    
    print("Instantiating interface")
    
    # Instantiate the interface
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
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Coupler port
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        
        ''' Setup mixers '''
        
        # Readout port
        pls.hardware.configure_mixer(
            freq      = readout_freq_nco,   # readout_freq_nco is set as the mixer NCO
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False, # Sync at next call
        )
        # Control port mixer
        pls.hardware.configure_mixer(
            freq      = control_freq_01,
            out_ports = control_port,
            sync      = False,
        )
        # Coupler port mixer
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
        # Control port amplitudes
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_01,
        )
        # Coupler bias amplitude (to be swept)
        pls.setup_scale_lut(
            output_ports    = coupler_dc_port,
            group           = 0,
            scales          = coupler_amp_arr,
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

        # Setup readout carrier, this tone will be swept in frequency.
        # The user provides an intended span.
        f_start = readout_freq_center_if - readout_freq_span / 2
        f_stop  = readout_freq_center_if + readout_freq_span / 2
        readout_freq_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the lower sideband. Note the minus sign.
        readout_pulse_freq_arr = readout_freq_nco - readout_freq_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = readout_freq_if_arr,
            phases       = np.full_like(readout_freq_if_arr, 0.0),
            phases_q     = np.full_like(readout_freq_if_arr, +np.pi / 2), # +pi/2 = LSB
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
        
        # Setup control_pulse_pi_01 carrier tone, considering that there are digital mixers.
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        
        
        
        
        ### Setup pulse "coupler_bias_tone" ###

        # Setup the coupler tone bias.
        coupler_bias_tone = pls.setup_long_drive(
            output_port = coupler_dc_port,
            group       = 0,
            duration    = added_delay_for_bias_tee,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = 0e-9,
            fall_time   = 0e-9
        )
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
        
        # Redefine the coupler DC pulse duration for repeated playback
        # once one tee risetime has passed.
        coupler_bias_tone.set_total_duration(readout_duration + repetition_delay)
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):

            # Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
            
            # Put the system into state |1>
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse_pi_01)
            T += control_duration_01
            
            # Commence readout, swept in frequency.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next scanned frequency
            pls.next_frequency(T, readout_stimulus_port)
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        
        # Increment the coupler port's DC bias.
        pls.next_scale(T, coupler_dc_port)
        
        # Move to next iteration.
        T += repetition_delay


        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################

        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   num_biases,
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
            'readout_pulse_freq_arr', "Hz",
            'coupler_amp_arr', "FS",
        ]
        hdf5_singles = [
            'readout_freq_nco', "Hz",
            'readout_freq_center_if', "Hz",
            'readout_freq_span', "Hz",
            'readout_amp', "FS",
            'readout_duration', "s",
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_delay', "s",
            
            'control_port', "",
            'control_amp_01', "FS",
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'added_delay_for_bias_tee', "s",
            'num_freqs', "",
            'num_averages', "",
            'num_biases', "",
            'coupler_bias_min', "FS",
            'coupler_bias_max', "FS",
        ]
        
        # Assert that every key bears a corresponding unit entered above.
        number_of_keyed_elements_is_even = \
            ((len(hdf5_steps) % 2) == 0) and ((len(hdf5_singles) % 2) == 0)
        assert number_of_keyed_elements_is_even, "Error: non-even amount "  + \
            "of keys and units entered in the portion of the measurement "  + \
            "script that saves the data. Someone likely forgot a comma."
        
        # Build step lists
        ext_keys = []
        for ii in range(0,len(hdf5_steps),2):
            if (hdf5_steps[ii] != 'fetched_data_arr') and (hdf5_steps[ii] != 'time_matrix'):
                temp_object = np.array( eval(hdf5_steps[ii]) )
                ext_keys.append(dict(name=hdf5_steps[ii], unit=hdf5_steps[ii+1], values=temp_object))
        for jj in range(0,len(hdf5_singles),2):
            if (hdf5_singles[jj] != 'fetched_data_arr') and (hdf5_singles[jj] != 'time_matrix'):
                temp_object = np.array( [eval(hdf5_singles[jj])] )
                ext_keys.append(dict(name=hdf5_singles[jj], unit=hdf5_singles[jj+1], values=temp_object))
        
        # Get name and time for logfile.
        path_to_script = os.path.realpath(__file__)  # Full path of current script
        current_dir, name_of_running_script = os.path.split(path_to_script)
        script_filename = os.path.splitext(name_of_running_script)[0]  # Name of current script
        timestamp = (datetime.now()).strftime("%d-%b-%Y_(%H_%M_%S)") # Date and time
        savefile_string = script_filename + '_12_' + timestamp + '.hdf5'  # Name of save file
        save_path = os.path.join(current_dir, "data", savefile_string)  # Full path of save file
        
        # Make logfile
        log_dict_list = [dict(name='fetched_data_arr', vector=False, complex=save_complex_data)]# , dict(name='time_matrix', vector=True)]
        f = Labber.createLogFile_ForData(savefile_string, log_dict_list, step_channels=ext_keys, use_database = False)
        
        # Set project name, tag, and user in logfile.
        f.setProject(script_filename)
        f.setTags('krizan')
        f.setUser('Christian Križan')
        
        # Split fetched_data_arr into repeats:
        # fetched_data_arr SHAPE: num_stores * repeat_count, num_ports, smpls_per_store
        integration_window_start = 1500 * 1e-9
        integration_window_stop  = 2000 * 1e-9
        t_span = integration_window_stop - integration_window_start
        
        # Get index corresponding to integration_window_start and integration_window_stop respectively
        integration_start_index = np.argmin(np.abs(time_matrix - integration_window_start))
        integration_stop_index = np.argmin(np.abs(time_matrix - integration_window_stop))
        integr_indices = np.arange(integration_start_index, integration_stop_index)
        
        # Construct a matrix, where every row is an integrated sampling
        # sequence corresponding to exactly one bias point.
        
        if(save_complex_data):
            angles = np.angle(fetched_data_arr[:, 0, integr_indices], deg=False)
            rows, cols = np.shape(angles)
            for row in range(rows):
                for col in range(cols):
                    #if angles[row][col] < 0.0:
                    angles[row][col] += (2.0 * np.pi)
            angles_mean = np.mean(angles, axis=-1)
            mean_len = len(angles_mean)
            for row in range(mean_len):
                angles_mean[row] -= (2.0 * np.pi)
            processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * angles_mean)
            ##processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(angles, axis=-1))
            ## processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(np.angle(fetched_data_arr[:, 0, integr_indices]), axis=-1))
        else:
            processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1)
        
        # Reshape depending on the repeat variable, as well as the inner loop
        # of the sequencer program.
        processing_arr.shape = (num_biases, num_freqs)
        
        for i in range(num_biases):
            f.addEntry( {"fetched_data_arr": processing_arr[i,:]} )
        # TODO: "time_matrix does not exist."
        #f.addEntry( {"time_matrix": time_matrix} )
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)
           

