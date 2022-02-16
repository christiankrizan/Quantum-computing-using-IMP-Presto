#####################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/upload/main
#####################################################################################

from presto import pulsed
from presto.utils import sin2, get_sourcecode
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

import os
import sys
import time
import math
import Labber
import shutil
import random
import numpy as np
from datetime import datetime

def get_theta_phi_of_next_pulse(i_am_here = [0,0], going_here = [0,0]):
    ''' Figures out how many degrees to apply onto
        the i_am_here coordinate to rotate the state vector
        along the shortest direction to the going_here coordinate.
        
        Figuring out how to return to |0> is easy using this method.
        Just call this subroutine using the starting coordinates
        as the going_here coordinates. See "starting_coordinate_deg"
        in randomised_benchmarking_single_qubit for instance.
    '''
    
    # Adjust input values into units of degrees.
    for ii in [0,1]:
        
        # If >=360, reduce with 360 until <360
        while going_here[ii] >= 360:
            going_here[ii] -= 360
        while i_am_here[ii] >= 360:
            i_am_here[ii] -= 360
        
        # If <0, add 360 until >=0
        while going_here[ii] < 0:
            going_here[ii] += 360
        while i_am_here[ii] < 0:
            i_am_here[ii] += 360
    
    # Get theta and phi for the state vector's next journey.
    # Figure out whether a negative or positive rotation will be faster.
    trip = [0,0]
    
    ## Below, clockwise/counter-clockwise is seen from
    ## "Sitting at +Z, looking down onto the XY plane"
    
    for jj in [0,1]:
        # Rotate both vectors to a "common" reference.
        going_here[jj] -= i_am_here[jj]
        i_am_here[jj]  -= i_am_here[jj]
        
        # i_am_here will now be 0. But going_here might be negative.
        # Re-normalise to 0-360 degrees.
        if going_here[jj] < 0:
            going_here[jj] += 360
        
        # Is the counter-clockwise arc smaller than the clockwise arc?
        if (360 - going_here[jj]) < going_here[jj]:
            # Then rotate counter-clockwise
            trip[jj] = -1 * (360 - going_here[jj])
        else:
            # Otherwise, go clockwise
            trip[jj] = going_here[jj]

    return trip
    

def randomised_benchmarking_01_single_qubit(
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
    
    num_cliffords,
    num_cliffords_rb_step_size,
    randomisation_seed,
    state_coordinates_deg = [[0,0],[90,0],[90,90],[90,180],[90,270],[180,0]],
    starting_coordinate_deg = [180,0],
    
    ):
    ''' Randomised benchmarking using Clifford-group single-qubit gates.
        The coordinate system, as in what state number corresponds to
        what theta and what phi, I have set as default to be:
            0: +Z
            1: +X
            2: +Y
            3: -X
            4: -Y
            5: -Z
        ... however, this is overwritable when calling the function.
        
        See the subroutine get_theta_phi_of_next_pulse to figure out
        how theta and phi is figured out for the upcoming pulse.
        
        The Randomised benchmarking schema is done via applying random
        gates from the Clifford group, N times. Finally, gate N+1 is applied,
        which will always be a "return to where I started" gate.
    '''
    
    # First, we acquire the number of states available
    # from the number of state coordinates entered.
    num_of_states = len(state_coordinates_deg)
    
    # We need to sanitise the user input.
    for uic in range(num_of_states):
        for tf in [0,1]:
            # If >=360, reduce with 360 until <360
            while (state_coordinates_deg[uic])[tf] >= 360:
                (state_coordinates_deg[uic])[tf] -= 360
            
            # If <0, add 360 until >=0
            while (state_coordinates_deg[uic])[tf] < 0:
                (state_coordinates_deg[uic])[tf] += 360
    
    
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
        
        # Control port(s)
        pls.hardware.set_dac_current(control_port, 40_500)
        pls.hardware.set_inv_sinc(control_port, 0)
        
        # Coupler port(s)
        pls.hardware.set_dac_current(coupler_dc_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_dc_port, 0)
        
        
        ''' Setup mixers '''
        
        # Readout port mixers
        pls.hardware.configure_mixer(
            freq      = readout_freq,
            in_ports  = readout_sampling_port,
            out_ports = readout_stimulus_port,
            sync      = False,
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
        
        ## For Randomised benchmarking, I am going to want to change
        ## the scale of the control pulse to match a newly found theta.
        control_amp_arr = np.linspace(0, control_amp_01, 181)
        
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
            scales          = control_amp_arr,
        )
        # Coupler port amplitude
        pls.setup_scale_lut(
            output_ports    = coupler_dc_port,
            group           = 0,
            scales          = coupler_dc_bias,
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
        

        ### Setup pulse "control_pulse_pi_01"
        
        ## ... likewise for Randomised Benchmarking, I am going to
        ## want to change phases at known times T. So I can declare
        ## a LUT of phases to change between.
        phase_arr = np.linspace(0, 2*np.pi, 361)
        
        # Setup the control_pulse_pi_01 pulse envelope.
        control_ns_01 = int(round(control_duration_01 * pls.get_fs("dac")))  # Number of samples in the control template.
        control_envelope_01 = sin2(control_ns_01)
        control_pulse_pi_01_pos = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
            envelope    = True,
        )
        control_pulse_pi_01_neg = pls.setup_template(
            output_port = control_port,
            group       = 0,
            template    = -control_envelope_01,
            template_q  = -control_envelope_01,
            envelope    = True,
        )
        # Setup control_pulse_pi_01 carrier tone, considering the mixer.
        ## Also, considering the phase-selectability in the RB sequence.
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 0,
            frequencies  = np.full_like(phase_arr, 0.0),
            phases       = phase_arr,   # Note: already converted to radians!
            phases_q     = phase_arr,   # e.g. phase_arr[180] = np.pi!
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
        
        # Preparing for the randomised benchmarking, we set the i_am_here
        # coordinate to the starting coordinate.
        i_am_here = starting_coordinate_deg
        going_here = i_am_here
        
        # For every number of cliffords to include in a sequence:
        for ii in range(0,num_cliffords,num_cliffords_rb_step_size):
            
            # The length of the bias pulse may differ depending on
            # the number of Cliffords that are made virtual.
            # Thus, we note here at what T the bias tone should be played,
            # and redefine its length once we know the total control duration.
            T_at_sequence_start = T
            
            ## The re-application of the bias tone is done after declaring
            ## the RB sequence further down.
            
            # Get the next RB sequence and apply it.
            # First, re-seed the random number generator.
            random.seed(randomisation_seed)
            
            # Reset the phase of the mixer initially to have
            # a known reference here.
            pls.reset_phase(T, control_port)
            
            # Schedule output of ii Clifford gates.
            # The for-loop below will schedule a whole RB-sequence, note that
            # this includes the final n+1 gate that returns the vector back
            # to where we started.
            for cliff_to_schedule in range(ii+1):
                
                # Now get how many degrees you need to go in directions
                # [theta, phi]. Note! If this is the very final
                # clifford gate that takes us back to where we started,
                # then don't choose a random coordinate.
                if cliff_to_schedule < ii:
                    going_here = state_coordinates_deg[math.floor(random.random() * num_of_states)]
                else:
                    # If we are at the ii+1 iteration, then we are going home.
                    going_here = starting_coordinate_deg
                trip = get_theta_phi_of_next_pulse( i_am_here, going_here )
                
                ## At this point, going_here already contains the phase of where
                ## I'm supposed to end up after some phi rotation trip[1].
                ## However, if I were to schedule several virtual-Z
                ## gates at the same time T, I may cause a bamboozle in the
                ## device. Since i_am_here is updated with going_here on
                ## every new iteration of this loop, I may in fact simply
                ## ignore changing anything unless there is a change in theta.
                
                # Will there be any pulse played? ( = Is there a theta change?)
                # If yes, schedule a pulse.
                if trip[0] != 0:

                    # Will this sequence involve a phi shift too?
                    # Then apply a virtual Z-gate here.
                    if trip[1] != 0:
                        pls.select_frequency(T, going_here[1], control_port)
                    
                    # Set the amplitude scaler to correspond to theta (trip[0])
                    pls.select_scale(T, abs(round(trip[0])), control_port)
                    
                    # Schedule amplitude positive or amplitude negative pulse:
                    if trip[0] > 0:
                        pls.output_pulse(T, control_pulse_pi_01_pos)
                    else:
                        pls.output_pulse(T, control_pulse_pi_01_neg)
                    T += control_duration_01
                
                # Update where you have ended up.
                i_am_here = going_here
            
            # At this point we know how long the RB sequence is.
            # Redefine the coupler DC pulse duration to keep on playing once
            # the bias tee has charged.
            coupler_bias_tone.set_total_duration(
                (T - T_at_sequence_start) + \
                readout_duration + \
                repetition_delay \
            )
            
            # Re-apply the coupler bias tone.
            pls.output_pulse(T_at_sequence_start, coupler_bias_tone)
            
            # Commence readout after the full RB+1 sequence.
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, readout_pulse)
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Await a new repetition, after which a new coupler DC bias tone
            # will be added - and a new frequency set for the readout tone.
            T += repetition_delay
        
        
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
        
    if not pls.dry_run:
        time_matrix, fetched_data_arr = pls.get_store_data()
        
        
        # For RB, make a vector showing the number of Cliffords that ran.
        num_cliffords_arr = np.arange(0,num_cliffords,num_cliffords_rb_step_size)
        
        print("Saving data")

        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
            'num_cliffords_arr', "",
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
            
            'coupler_dc_port',"",
            'coupler_dc_bias',"FS",
            'added_delay_for_bias_tee',"s",
            
            'num_averages',"",
            
            'num_cliffords',"",
            'num_cliffords_rb_step_size',"",
            'randomisation_seed',"",
            #state_coordinates_deg = [[0,0],[90,0],[90,90],[90,180],[90,270],[180,0]],
            #starting_coordinate_deg = [0,0],
        ]
        hdf5_logs = [
            'fetched_data_arr_1',
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
        savefile_string = script_filename + '_'+str(len(num_cliffords_arr))+'x'+str(num_cliffords_rb_step_size)+'_cliffords_' + timestamp + '.hdf5'  # Name of save file
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
        ##processing_arr.shape = (num_biases, num_freqs)
        
        ##for i in range(num_biases):
        f.addEntry( {"fetched_data_arr_1": processing_arr[:]} )
        # TODO: "time_matrix does not exist."
        #f.addEntry( {"time_matrix": time_matrix} )
        
        # Check if the hdf5 file was created in the local directory.
        # If so, move it to the 'data' directory.
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , os.path.join(current_dir, "data", savefile_string))

        # Print final success message.
        print("Data saved, see " + save_path)


