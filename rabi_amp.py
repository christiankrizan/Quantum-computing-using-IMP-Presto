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

import load_rabi_amp

def oscillation01_with_coupler_bias(
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
    control_freq_01,
    control_duration_01,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_amplitudes,
    num_biases,
    num_averages,
    
    control_amp_01_min = 0.0,
    control_amp_01_max = 1.0,
    
    coupler_dc_bias_min = 0.0,
    coupler_dc_bias_max = 1.0,
    ):
    ''' Perform a Rabi oscillation experiment between states |0> and |1>.
        The energy is found by sweeping the amplitude, instead of
        sweeping the pulse duration. While sweeping, a bias voltage
        is applied onto a connected coupler.
    '''

    # Declare amplitude array for the Rabi experiment.
    control_amp_arr = np.linspace(control_amp_01_min, control_amp_01_max, num_amplitudes)
    
    # Declare amplitude array for the coupler sweep.
    coupler_amp_arr = np.linspace(coupler_dc_bias_min, coupler_dc_bias_max, num_biases)

    print("Instantiating interface")

    # Instantiate interface class
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
        
        # Coupler port(s)
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)


        ''' Setup mixers '''
        
        # Readout mixer
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixer
        pls.hardware.configure_mixer(
            freq      = control_freq_01,  # Note that the 01 transition freq. is set as the mixer NCO
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
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = control_amp_arr,
        )
        # Coupler port amplitude(s) (to be swept)
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
        # Setup readout carrier, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup pulse "control_pulse_pi_01" ###

        # Setup control_pulse_pi_01 pulse envelope
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        # Setup control_pulse_pi_01 carrier tone, considering that there is a digital mixer
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
            duration    = added_delay_for_bias_tee, # Set a minimum initial delay, will be changed.
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
        T = 0.0 # s
        
        # Charge the bias tee.
        pls.reset_phase(T, coupler_dc_port)
        pls.output_pulse(T, coupler_bias_tone)
        T += added_delay_for_bias_tee
        
        for ii in range(num_amplitudes):
        
            # Re-apply the coupler DC pulse once one tee risetime has passed.
            coupler_bias_tone.set_total_duration(control_duration_01 + readout_duration + repetition_delay)
            
            # Output the pi01-pulse to be characterised,
            # along with the coupler DC tone
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, [control_pulse_pi_01, coupler_bias_tone])
            T += control_duration_01
            
            # Readout pulse starts right after control pulse
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next Rabi amplitude
            pls.next_scale(T, control_port)
            
            # Wait for decay
            T += repetition_delay
        
        # Increment the coupler port's DC bias.
        pls.next_scale(T, coupler_dc_port)
        
        # Move to next iteration.
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        # Repeat the whole sequence 'num_amplitudes' times,
        # then average 'num_averages' times
        pls.run(
            period       = T,
            repeat_count = num_biases,
            num_averages = num_averages,
            print_time   = True,
        )
    
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
        ## TODO: Do not save as h5py-HDF5. Instead, run a Rabi fit on
        ##       every row of the arr.shape(bla, bla).
        ##       Save all of this data in a vector of the log browser-styled
        ##       hdf5.
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
            'control_amp_arr', "FS",
            'coupler_amp_arr', "FS",
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
            'control_freq_01', "Hz",
            'control_duration_01', "s",
            
            'coupler_dc_port', "",
            'added_delay_for_bias_tee', "s",
            
            'num_amplitudes', "",
            'num_biases', "",
            'num_averages', "",
            
            'control_amp_01_min', "FS",
            'control_amp_01_max', "FS",
            
            'coupler_dc_bias_min', "FS",
            'coupler_dc_bias_max', "FS",
        ]
        hdf5_logs = [
            'fetched_data_arr',
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
        savefile_string = script_filename + '01_sweep_bias_' + timestamp + '.hdf5'  # Name of save file
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
        
        # Construct a matrix, where every row is an integrated sampling
        # sequence corresponding to exactly one bias point.
        processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1)
        processing_arr.shape = (num_biases, num_amplitudes)
        
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


def oscillation01_with_coupler_bias_multiplexed_ro(
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
    
    control_port_A,
    control_freq_01_A,
    control_port_B,
    control_freq_01_B,
    control_duration_01,
    
    coupler_dc_port,
    added_delay_for_bias_tee,
    
    num_amplitudes,
    num_biases,
    num_averages,
    
    control_amp_01_A_min = -1.0,
    control_amp_01_A_max = +1.0,
    control_amp_01_B_min = -1.0,
    control_amp_01_B_max = +1.0,
    
    coupler_dc_bias_min = -1.0,
    coupler_dc_bias_max = +1.0,
    ):
    ''' Perform a Rabi oscillation experiment between states |0> and |1>.
        The energy is found by sweeping the amplitude, instead of
        sweeping the pulse duration. While sweeping, a bias voltage
        is applied onto a connected coupler.
        
        The readout is multiplexed between two pairwise-coupled transmons.
    '''

    # Declare amplitude arrays for the Rabi experiment.
    control_amp_arr_A = np.linspace(control_amp_01_A_min, control_amp_01_A_max, num_amplitudes)
    control_amp_arr_B = np.linspace(control_amp_01_B_min, control_amp_01_B_max, num_amplitudes)
    
    # Declare amplitude array for the coupler sweep.
    coupler_amp_arr = np.linspace(coupler_dc_bias_min, coupler_dc_bias_max, num_biases)

    print("Instantiating interface")

    # Instantiate interface class
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
            scales          = control_amp_arr_A,
        )
        pls.setup_scale_lut(
            output_ports    = control_port_B,
            group           = 0,
            scales          = control_amp_arr_B,
        )
        # Coupler port amplitude (to be swept)
        pls.setup_scale_lut(
            output_ports    = coupler_dc_port,
            group           = 0,
            scales          = coupler_amp_arr,
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
        T = 0.0 # s
        
        # Charge the bias tee.
        pls.reset_phase(T, coupler_dc_port)
        pls.output_pulse(T, coupler_bias_tone)
        T += added_delay_for_bias_tee
        
        # Redefine the coupler DC pulse duration for repeated playback
        # once one tee risetime has passed.
        for bias_tone in coupler_bias_tone:
            bias_tone.set_total_duration(control_duration_01 + readout_duration + repetition_delay)
        
        # For all amplitudes to sweep over:
        for ii in range(num_amplitudes):
            
            # Re-apply the coupler DC pulse once one tee risetime has passed.
            pls.output_pulse(T, coupler_bias_tone)
            
            # Output the pi01-pulses to be characterised
            pls.reset_phase(T, [control_port_A, control_port_B])
            pls.output_pulse(T, [control_pulse_pi_01_A, control_pulse_pi_01_B])
            T += control_duration_01
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Move to next Rabi amplitude
            pls.next_scale(T, [control_port_A, control_port_B])
            
            # Wait for decay
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
            period       = T,
            repeat_count = num_biases,
            num_averages = num_averages,
            print_time   = True,
        )
    
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")

        ## TODO: Do not save as h5py-HDF5. Instead, run a Rabi fit on
        ##       every row of the arr.shape(bla, bla).
        ##       Save all of this data in a vector of the log browser-styled
        ##       hdf5.
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # This save is done in a loop, due to quirks with Labber's log browser.
        arrays_in_loop = [
            'control_amp_arr_A',
            'control_amp_arr_B'
        ]
        for u in range(2):
        
            # Data to be stored.
            hdf5_steps = [
                arrays_in_loop[u], "FS",
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
                'control_port_B', "",
                'control_freq_01_B', "Hz",
                'control_duration_01', "s",
                
                #'coupler_dc_port', "",
                'added_delay_for_bias_tee', "s",
                
                'num_amplitudes', "",
                'num_biases', "",
                'num_averages', "",
                
                'control_amp_01_A_min', "FS",
                'control_amp_01_A_max', "FS",
                'control_amp_01_B_min', "FS",
                'control_amp_01_B_max', "FS",
                
                'coupler_dc_bias_min', "FS",
                'coupler_dc_bias_max', "FS",
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
            savefile_string = script_filename + '01_sweep_bias_multiplexed_'+str((u)+1)+'_of_2_' + timestamp + '.hdf5'  # Name of save file
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
            processing_arr_1.shape = (num_biases, num_amplitudes)
            processing_arr_2.shape = (num_biases, num_amplitudes)
            
            # Take the absolute value of the data.
            processing_arr_1 = np.abs(processing_arr_1)
            processing_arr_2 = np.abs(processing_arr_2)
            
            print("... storing processed data into the HDF5 file.")
            
            
            # Check if the hdf5 file was created in the local directory.
            # If so, move it to the 'data' directory.
            if os.path.isfile(os.path.join(current_dir, savefile_string)):
                shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

            # Print final success message.
            print("Data saved, see " + save_path)




def oscillation01(
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
    control_freq_01,
    control_duration_01,
    
    num_amplitudes,
    num_averages,
    
    control_amp_01_min = 0.0,
    control_amp_01_max = 1.0,
    ):
    ''' Perform a Rabi oscillation experiment between states |0> and |1>.
        The energy is found by sweeping the amplitude, instead of
        sweeping the pulse duration.
    '''

    # Declare amplitude array for the Rabi experiment
    control_amp_arr = np.linspace(control_amp_01_min, control_amp_01_max, num_amplitudes)

    print("Instantiating interface")

    # Instantiate interface class
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


        ''' Setup mixers '''
        
        # Readout mixer
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False, # Sync at next call
        )
        # Control port mixer
        pls.hardware.configure_mixer(
            freq      = control_freq_01,  # Note that the 01 transition freq. is set as the mixer NCO
            out_ports = control_port,
            sync      = True,  # Sync here
        )
        

        ''' Setup scale LUTs '''
        
        # Readout amplitude
        pls.setup_scale_lut(
            output_ports=readout_stimulus_port,
            group=0,
            scales=readout_amp,
        )
        # Control port amplitude sweep for pi_01
        pls.setup_scale_lut(
            output_ports=control_port,
            group=1,
            scales=control_amp_arr,
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
        pls.setup_freq_lut(
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        
        
        ### Setup pulse "control_pulse_pi_01" ###

        # Setup control_pulse_pi_01 pulse envelope
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01 = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tone, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
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
        
        # Pi_01 pulse to be characterised
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi_01)
        T += control_duration_01
        
        # Readout pulse starts right after control pulse
        pls.reset_phase(T, readout_stimulus_port)
        pls.output_pulse(T, readout_pulse)
        pls.store(T + readout_sampling_delay) # Sampling window
        
        # Move to next Rabi amplitude
        pls.next_scale(T, control_port)
        
        # Wait for decay
        T += repetition_delay
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        # Repeat the whole sequence `num_amplitudes` times,
        # then average `num_averages` times
        pls.run(
            period       = T,
            repeat_count = num_amplitudes,
            num_averages = num_averages,
            print_time   = True,
        )
    
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        

        print("Saving data")

        
        ####################
        ''' SAVE AS HDF5 '''
        ####################
        assert 1 == 0, "Error: H5PY save method deprecated. Change to Log Browser"
        
        

