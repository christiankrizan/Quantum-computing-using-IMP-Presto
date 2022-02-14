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


def iswap_sweep_duration_and_detuning(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_duration,
    readout_freq_A,
    readout_amp_A,
    readout_freq_B,
    readout_amp_B,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
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
    coupler_ac_freq_iswap_center_if,
    coupler_ac_freq_iswap_span,
    
    num_freqs,
    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    
    ):
    ''' Tune an iSWAP-interaction between two qubits using
        a tuneable coupler, by fixing the gate amplitude and gate bias.
        Thus, the gate duration and detuning is swept.
    '''
    
    
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
        
        # Coupler port(s)
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)

        
        ''' But first, something completely different '''

        # Calculate the dt_per_time_step for the iSWAP plateau duration.
        # The instrument can only represent times that are multiples of
        # pls.get_clk_T().
        # Solution: force the following parameters to be multiples of
        # pls.get_clk_T():
        #   - dt_per_time_step
        #   - coupler_ac_plateau_duration_iswap_min
        #   - coupler_ac_plateau_duration_iswap_max
        # Then, generate the time array that dictates the axis in the
        # data storage subroutine. Finally, set the num_time_steps
        # to the amount of entries in this list.
        ## 1.   Make coupler_ac_plateau_duration_iswap_min and _max into
        ##      representable values.
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        ## TODO THIS PIECE OF CODE SHOULD BE ADDED HERE, BUT I HAVE NOT
        ##      DONE A MEASUREMENT WITH THIS CODE PIECE IN PLACE:
        ##      THIS ONE -->> coupler_ac_single_edge_time_iswap     = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_min = int(round(coupler_ac_plateau_duration_iswap_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_max = int(round(coupler_ac_plateau_duration_iswap_max / plo_clk_T)) * plo_clk_T
        ## 2.   Figure out a dt_per_time_step and make the value representable.
        dt_per_time_step = (coupler_ac_plateau_duration_iswap_max-coupler_ac_plateau_duration_iswap_min)/num_time_steps
        dt_per_time_step = int(round(dt_per_time_step / plo_clk_T)) * plo_clk_T
        ## 3.   Generate an array for data storage.
        iswap_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_iswap_min + 2*coupler_ac_single_edge_time_iswap, \
            coupler_ac_plateau_duration_iswap_max + 2*coupler_ac_single_edge_time_iswap, \
            num_time_steps)
        ## 4.   For all elements, round to the programmable logic clock period.
        for jj in range(len(iswap_total_pulse_duration_arr)):
            iswap_total_pulse_duration_arr[jj] = int(round(iswap_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        ## 5.   Remove duplicates
        new_list = []
        for kk in range(len(iswap_total_pulse_duration_arr)):
            if not (iswap_total_pulse_duration_arr[kk] in new_list):
                new_list.append(iswap_total_pulse_duration_arr[kk])
        iswap_total_pulse_duration_arr = new_list
        ## 6.   Update the num_time_steps parameter.
        num_time_steps = len(iswap_total_pulse_duration_arr)
        
        
        ''' Setup mixers '''
        
        # Readout port, multiplexed, calculate an optimal NCO frequency.
        ##readout_freq_nco = max(readout_freq_A, readout_freq_B) + readout_band_nco_offset
        readout_freq_nco = readout_freq_A
        ##readout_freq_nco = readout_freq_A ## TODO_theoneworking
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
            sync      = False,
        )
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
            scales          = coupler_ac_amp,
        )
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
            frequencies  = readout_freq_if_A, ## TODO
            phases       = np.full_like(readout_freq_if_A, 0.0),
            phases_q     = np.full_like(readout_freq_if_A, -np.pi/2),##+np.pi/2), # LSB
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = readout_freq_if_B, ## TODO
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
                          2*coupler_ac_single_edge_time_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        
        # Setup the iSWAP pulse carrier, this tone will be swept in frequency.
        # Since we set the mixer to some NCO value, we probably want to use
        # the lower sideband for sweeping the span (not the upper).
        f_start = coupler_ac_freq_iswap_center_if - coupler_ac_freq_iswap_span / 2
        f_stop = coupler_ac_freq_iswap_center_if + coupler_ac_freq_iswap_span / 2
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
        
        # For every resonator stimulus pulse duration to sweep over:
        for ii in range(num_time_steps):

            # Redefine the iSWAP pulse's total duration,
            # resulting in stepping said duration in time.
            coupler_ac_duration_iswap = \
                2*coupler_ac_single_edge_time_iswap + \
                coupler_ac_plateau_duration_iswap_min + \
                ii*dt_per_time_step
            coupler_ac_pulse_iswap.set_total_duration(coupler_ac_duration_iswap)

            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
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
        
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        
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
            'coupler_dc_port',"",
            'coupler_dc_bias',"FS",
            'added_delay_for_bias_tee',"s",
            'coupler_ac_port',"",
            'coupler_ac_amp',"FS",
            'coupler_ac_freq_iswap_nco',"Hz",
            'coupler_ac_freq_iswap_center_if',"Hz",
            'coupler_ac_freq_iswap_span',"Hz",
            'num_freqs',"",
            'num_averages',"",
            'num_time_steps',"",
            'coupler_ac_single_edge_time_iswap',"s",
            'coupler_ac_plateau_duration_iswap_min',"s",
            'coupler_ac_plateau_duration_iswap_max',"s",
        ]
        hdf5_logs = [
            'fetched_data_arr_1',
            'fetched_data_arr_2',
        ]
        
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
        savefile_string = script_filename + '_sweep_time_and_detuning_' + timestamp + '.hdf5'  # Name of save file
        save_path = os.path.join(current_dir, "data", savefile_string)  # Full path of save file
        
        # Make logfile
        f = Labber.createLogFile_ForData(savefile_string, log_dict_list, step_channels=ext_keys, use_database = False)
        
        # Set project name, tag, and user in logfile.
        f.setProject(script_filename)
        f.setTags('krizan')
        f.setUser('Christian Križan')
        
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
        processing_arr_1.shape = (num_freqs, num_time_steps)
        processing_arr_2.shape = (num_freqs, num_time_steps)
        
        # Take the absolute value of the data.
        processing_arr_1 = np.abs(processing_arr_1)
        processing_arr_2 = np.abs(processing_arr_2)
        
        for i in range(num_freqs):
            f.addEntry( {"fetched_data_arr_1": processing_arr_1[i,:], "fetched_data_arr_2": processing_arr_2[i,:] } )
            ## TODO Storing several arrays would likely need some sort of string making followed by evil Python evals.
        # TODO: "time_matrix does not exist."
        ## TODO Actually while on this topic, likely I just need to add another
        ## f.addEntry bearing time_matrix. Whether it has to have the same
        ## amount of values in some axis n, I don't know as of writing.
        #f.addEntry( {"time_matrix": time_matrix} )
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)


def iswap_sweep_duration_and_amplitude(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_duration,
    readout_freq_A,
    readout_amp_A,
    readout_freq_B,
    readout_amp_B,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
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
    coupler_ac_freq_iswap,
    
    num_amplitudes,
    coupler_ac_amp_min,
    coupler_ac_amp_max,

    num_averages,
    
    num_time_steps,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap_min,
    coupler_ac_plateau_duration_iswap_max,
    
    ):
    ''' Tune an iSWAP-interaction between two qubits, where it is known at
        what gate frequency the iSWAP interaction takes place (and with
        what coupler bias), but not the iSWAP gate amplitude nor the gate
        duration.
    '''
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
    
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
        
        # Coupler port(s)
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        
        ''' Make the user-set time variables representable '''
        ## 1.   Make coupler_ac_plateau_duration_iswap_min and _max into
        ##      representable values.
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        coupler_ac_single_edge_time_iswap     = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_min = int(round(coupler_ac_plateau_duration_iswap_min / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap_max = int(round(coupler_ac_plateau_duration_iswap_max / plo_clk_T)) * plo_clk_T
        ## 2.   Figure out a dt_per_time_step and make the value representable.
        dt_per_time_step = (coupler_ac_plateau_duration_iswap_max-coupler_ac_plateau_duration_iswap_min)/num_time_steps
        dt_per_time_step = int(round(dt_per_time_step / plo_clk_T)) * plo_clk_T
        ## 3.   Generate an array for data storage.
        iswap_total_pulse_duration_arr = np.linspace( \
            coupler_ac_plateau_duration_iswap_min + 2*coupler_ac_single_edge_time_iswap, \
            coupler_ac_plateau_duration_iswap_max + 2*coupler_ac_single_edge_time_iswap, \
            num_time_steps)
        ## 4.   For all elements, round to the programmable logic clock period.
        for jj in range(len(iswap_total_pulse_duration_arr)):
            iswap_total_pulse_duration_arr[jj] = int(round(iswap_total_pulse_duration_arr[jj] / plo_clk_T)) * plo_clk_T
        ## 5.   Remove duplicates
        new_list = []
        for kk in range(len(iswap_total_pulse_duration_arr)):
            if not (iswap_total_pulse_duration_arr[kk] in new_list):
                new_list.append(iswap_total_pulse_duration_arr[kk])
        iswap_total_pulse_duration_arr = new_list
        ## 6.   Update the num_time_steps parameter.
        num_time_steps = len(iswap_total_pulse_duration_arr)
        
        
        ''' Setup mixers '''
        
        # Readout port, multiplexed, calculate an optimal NCO frequency.
        ##TODO readout_freq_nco = (np.abs(readout_freq_A) + np.abs(readout_freq_B))/2
        readout_freq_nco = readout_freq_A
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
            sync      = False,
        )
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
            frequencies  = readout_freq_if_A, ## TODO
            phases       = np.full_like(readout_freq_if_A, 0.0),
            phases_q     = np.full_like(readout_freq_if_A, -np.pi/2),##+np.pi/2), # LSB
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = readout_freq_if_B, ## TODO
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
                          2*coupler_ac_single_edge_time_iswap,
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
        # TODO To what extent do I need to set a +pi/2 q-phase?
        # Should I have it activated to match the phase of the
        # frequency sweeps I have done previously? I am not sure.
        # TODO! UPDATE from 2022-02-08
        # Many old files have their multiplexed readout set wrongly.
        # These should all be USB as of writing.
        
        
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
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_time_steps):
        
            # Redefine the iSWAP pulse's total duration,
            # resulting in stepping said duration in time.
            coupler_ac_duration_iswap = \
                2*coupler_ac_single_edge_time_iswap + \
                coupler_ac_plateau_duration_iswap_min + \
                ii*dt_per_time_step
            coupler_ac_pulse_iswap.set_total_duration(coupler_ac_duration_iswap)
            
            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
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
        
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        
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
            'iswap_total_pulse_duration_arr',"s",
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
            
            'coupler_dc_port', "",
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
            'fetched_data_arr_1',
            'fetched_data_arr_2',
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
        savefile_string = script_filename + '_sweep_amplitude_and_detuning_' + timestamp + '.hdf5'  # Name of save file
        save_path = os.path.join(current_dir, "data", savefile_string)  # Full path of save file
        
        # Make logfile
        print("... making Log browser logfile.")
        f = Labber.createLogFile_ForData(savefile_string, log_dict_list, step_channels=ext_keys, use_database = False)
        
        # Set project name, tag, and user in logfile.
        f.setProject(script_filename)
        f.setTags('krizan')
        f.setUser('Christian Križan')
        
        print("... processing multiplexed readout data.")
        
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
        processing_arr_1.shape = (num_amplitudes, num_time_steps)
        processing_arr_2.shape = (num_amplitudes, num_time_steps)
        
        # Take the absolute value of the data.
        processing_arr_1 = np.abs(processing_arr_1)
        processing_arr_2 = np.abs(processing_arr_2)
        
        print("... storing processed data into the HDF5 file.")
        
        ## NOTE! For this specific routine, the order of storing
        ## arrays vs. columns has been reversed, in order to put frequency
        ## on the 2D Y-axis.
        for i in range(num_amplitudes):
            f.addEntry( {"fetched_data_arr_1": processing_arr_1[i,:], "fetched_data_arr_2": processing_arr_2[i,:] } )
            ## TODO Storing several arrays would likely need some sort of string making followed by evil Python evals.
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)



def iswap_sweep_amplitude_and_detuning(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_duration,
    readout_freq_A,
    readout_amp_A,
    readout_freq_B,
    readout_amp_B,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
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
    coupler_ac_freq_iswap_center_if,
    coupler_ac_freq_iswap_span,
    
    coupler_ac_amp_min,
    coupler_ac_amp_max,
    
    num_freqs,
    num_averages,
    num_amplitudes,
    
    ):
    ''' Tune an iSWAP-interaction between two qubits using
        a tuneable coupler.
    '''
    
    # Declare amplitude array for the AC coupler tone to be swept
    coupler_ac_amp_arr = np.linspace(coupler_ac_amp_min, coupler_ac_amp_max, num_amplitudes)
    
    
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
        
        # Coupler port(s)
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        
        ''' Make the user-set time variables representable '''
        
        ## TODO This should be done with the XY-pulses as well.
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
        ''' Setup mixers '''
        
        # Readout port, multiplexed, calculate an optimal NCO frequency.
        ##TODO readout_freq_nco = (np.abs(readout_freq_A) + np.abs(readout_freq_B))/2
        readout_freq_nco = readout_freq_A
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
            sync      = False,
        )
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
            frequencies  = readout_freq_if_A, ## TODO
            phases       = np.full_like(readout_freq_if_A, 0.0),
            phases_q     = np.full_like(readout_freq_if_A, -np.pi/2),##+np.pi/2), # LSB
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = readout_freq_if_B, ## TODO
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
            2*coupler_ac_single_edge_time_iswap + \
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
        f_start = coupler_ac_freq_iswap_center_if - coupler_ac_freq_iswap_span / 2
        f_stop = coupler_ac_freq_iswap_center_if + coupler_ac_freq_iswap_span / 2
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
        coupler_bias_tone.set_total_duration(
            control_duration_01 + \
            coupler_ac_duration_iswap + \
            readout_duration + \
            repetition_delay \
        )
        
        # For every resonator stimulus pulse frequency to sweep over:
        for ii in range(num_freqs):

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
        
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        
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
            'coupler_ac_amp_arr', "FS",
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
            
            'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_duration_iswap', "s",
            'coupler_ac_freq_iswap_nco', "Hz",
            'coupler_ac_freq_iswap_center_if', "Hz",
            'coupler_ac_freq_iswap_span', "Hz",
            
            'coupler_ac_amp_min', "FS",
            'coupler_ac_amp_max', "FS",
            
            'num_freqs', "",
            'num_averages', "",
            'num_amplitudes', "",
        ]
        hdf5_logs = [
            'fetched_data_arr_1',
            'fetched_data_arr_2',
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
        savefile_string = script_filename + '_sweep_amplitude_and_detuning_' + timestamp + '.hdf5'  # Name of save file
        save_path = os.path.join(current_dir, "data", savefile_string)  # Full path of save file
        
        # Make logfile
        print("... making Log browser logfile.")
        f = Labber.createLogFile_ForData(savefile_string, log_dict_list, step_channels=ext_keys, use_database = False)
        
        # Set project name, tag, and user in logfile.
        f.setProject(script_filename)
        f.setTags('krizan')
        f.setUser('Christian Križan')
        
        print("... processing multiplexed readout data.")
        
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
        processing_arr_1.shape = (num_amplitudes, num_freqs)
        processing_arr_2.shape = (num_amplitudes, num_freqs)
        
        # Take the absolute value of the data.
        processing_arr_1 = np.abs(processing_arr_1)
        processing_arr_2 = np.abs(processing_arr_2)
        
        print("... storing processed data into the HDF5 file.")
        
        ## NOTE! For this specific routine, the order of storing
        ## arrays vs. columns has been reversed, in order to put frequency
        ## on the 2D Y-axis.
        for i in range(num_freqs):
            f.addEntry( {"fetched_data_arr_1": processing_arr_1[:,i], "fetched_data_arr_2": processing_arr_2[:,i] } )
            ## TODO Storing several arrays would likely need some sort of string making followed by evil Python evals.
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)
