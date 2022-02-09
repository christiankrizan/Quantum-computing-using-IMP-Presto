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


def establish_drag_coefficient_alpha(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_duration,
    readout_freq,
    readout_amp,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_averages,
    
    num_unitary_pulse_pairs_min,
    num_unitary_pulse_pairs_max,
    num_unitary_pulse_pairs_step_size,
    
    drag_coefficient_step_size,
    drag_coefficient_min = -1.0,
    drag_coefficient_max = +1.0,
    
    ):
    ''' Perform single-qubit DRAG tune-up, allowing for biasing one
        connected SQUID coupler. The goal is to establish the DRAG coefficient
        alpha.
    '''
    
    # Make the num_unitary_pulse_pairs_min argument legal.
    if num_unitary_pulse_pairs_min < 0:
        num_unitary_pulse_pairs_min = 0
    
    # Declare array with the number of unitary pulse pairs to step over
    # in the main sequencer loop. And, make the array legal.
    num_unitary_pairs_arr = np.arange(num_unitary_pulse_pairs_min, num_unitary_pulse_pairs_max, num_unitary_pulse_pairs_step_size)
    num_unitary_pairs_arr = (np.unique(np.round(num_unitary_pairs_arr))).astype(int)
    
    # Declare array bearing the DRAG coefficients, with a resolution
    # as given by user input.
    drag_coefficient_arr = np.arange(drag_coefficient_min, drag_coefficient_max, drag_coefficient_step_size)
    num_drag_coefficients = len(drag_coefficient_arr)
    
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
        
        # Coupler port(s)
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        
        ''' Setup mixers '''
        
        # Readout port,
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
        )
        # Control port mixers
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
        
        # Readout port amplitude
        pls.setup_scale_lut(
            output_ports    = readout_stimulus_port,
            group           = 0,
            scales          = readout_amp,
        )
        # Control port amplitudes
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 0,
            scales          = np.full(num_drag_coefficients, control_amp_01),
        )
        pls.setup_scale_lut(
            output_ports    = control_port,
            group           = 1,
            scales          = np.full(num_drag_coefficients, control_amp_01) * drag_coefficient_arr,
        )
        # Coupler port amplitudes
        pls.setup_scale_lut(
            output_ports    = coupler_dc_port,
            group           = 0,
            scales          = coupler_dc_bias,
        )
        
        
        
        ### Setup readout pulses ###
        
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
            output_ports =  readout_stimulus_port,
            group        =  0,
            frequencies  =  0.0,
            phases       =  0.0,
            phases_q     =  0.0,
        )
        

        ### Setup pulse "control_pulse_pi_01"
        
        ##  This pulse setup will be different in order to use a scaler
        ##  on the Q-portion.
        
        # Setup control_pulse_pi_01 and control_pulse_pi_inv_01 pulse envelopes.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope_01 = sin2(control_ns_01)
        control_envelope_01_inv = -1 * sin2(control_ns_01)
        
        control_pulse_pi_01_I = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = np.full_like(control_envelope_01, 0.0),
            envelope    = True,
        )
        control_pulse_pi_01_Q = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = np.full_like(control_envelope_01, 0.0),
            template_q  = control_envelope_01,
            envelope    = True,
        )
        
        control_pulse_pi_01_inv_I = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01_inv,
            template_q  = np.full_like(control_envelope_01_inv, 0.0),
            envelope    = True,
        )
        control_pulse_pi_01_inv_Q = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = np.full_like(control_envelope_01_inv, 0.0),
            template_q  = control_envelope_01_inv,
            envelope    = True,
        )
        
        # Setup control_pulse_pi_01 carrier tones,
        # considering that there are digital mixers
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
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
        T = 0.0  # s

        # Charge the bias tee.
        pls.reset_phase(T, coupler_dc_port)
        pls.output_pulse(T, coupler_bias_tone)
        T += added_delay_for_bias_tee
        
        # For all numbers of unitary pair lengths
        for ii in num_unitary_pairs_arr:
            
            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
            for bias_tone in coupler_bias_tone:
                bias_tone.set_total_duration(
                    2*ii * control_duration_01 + \
                    readout_duration + \
                    repetition_delay \
                )
            
            ## Re-apply the coupler bias tone.
            pls.output_pulse(T, coupler_bias_tone)
            
            # Apply the unitary pairs.
            pls.reset_phase(T, control_port)
            for dummy in range(ii):
                pls.output_pulse(T, [control_pulse_pi_01_I, control_pulse_pi_01_Q])
                T += control_duration_01
                pls.output_pulse(T, [control_pulse_pi_01_inv_I, control_pulse_pi_01_inv_Q])
                T += control_duration_01
            
            # Commence readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        # Increment the control port scalars, groups 0 (I) and 1 (Q)
        pls.next_scale(T, control_port)
        
        # Move to next iteration.
        T += repetition_delay
        
        
        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################

        # Average the measurement over 'num_averages' averages
        pls.run(
            period          =   T,
            repeat_count    =   num_drag_coefficients,
            num_averages    =   num_averages,
            print_time      =   True,
            enable_compression = True # Feature!
        )
        
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
            'num_unitary_pairs_arr',"",
            'drag_coefficient_arr', "",
        ]
        hdf5_singles = [
            'readout_stimulus_port',"",
            'readout_sampling_port',"",
            'readout_duration',"s",
            'readout_freq',"Hz",
            'readout_amp',"FS",
            
            'sampling_duration',"s",
            'readout_sampling_delay',"s",
            'repetition_delay',"s",
            
            'control_port',"",
            'control_amp_01',"FS",
            'control_freq_01',"Hz",            
            'control_duration_01',"s",
            
            #'coupler_dc_port',"",
            'coupler_dc_bias',"FS",
            'added_delay_for_bias_tee',"s",
            
            'num_averages', "",
            'num_drag_coefficients', "",
            
            'num_unitary_pulse_pairs_min',"",
            'num_unitary_pulse_pairs_max',"",
            'num_unitary_pulse_pairs_step_size',"",
            'drag_coefficient_min',"",
            'drag_coefficient_max',"",
            
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
        savefile_string = script_filename + '_find_alpha_' + timestamp + '.hdf5'  # Name of save file
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
        processing_arr.shape = (num_drag_coefficients, len(num_unitary_pairs_arr))
        
        for i in range(num_drag_coefficients):
            f.addEntry( {"fetched_data_arr": processing_arr[i,:]} )
        # TODO: "time_matrix does not exist."
        #f.addEntry( {"time_matrix": time_matrix} )
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)
