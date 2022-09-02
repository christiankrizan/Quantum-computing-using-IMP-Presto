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
from phase_calculator import \
    reset_phase_counter, \
    get_legal_phase, \
    bandsign, \
    add_virtual_z
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_timestamp_string, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save

##from time_remaining_printer import show_user_time_remaining

def execute(
    quantum_circuit,
    
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
    control_freq_nco_A,
    control_port_B,
    control_amp_01_B,
    control_freq_01_B,
    control_freq_nco_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_freq_nco,
    
    coupler_ac_amp_cz20,
    coupler_ac_freq_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    phase_adjustment_after_cz20_A,
    phase_adjustment_after_cz20_B,
    phase_adjustment_coupler_ac_cz20,
    
    coupler_ac_amp_iswap,
    coupler_ac_freq_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    phase_adjustment_after_iswap_A,
    phase_adjustment_after_iswap_B,
    phase_adjustment_coupler_ac_iswap,
    
    num_averages,
    
    save_complex_data = True,
    use_log_browser_database = True,
    suppress_log_browser_export = False,
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
    ''' Execute an arbitrary quantum circuit, of course within the limitations
        of the native gate set.
    '''
    
    ## Input sanitisation
    
    # Acquire legal values regarding the coupler port settings.
    if ((coupler_dc_port == []) and (coupler_dc_bias != 0.0)):
        print("Note: the coupler bias was set to 0, since the coupler_port array was empty.")
        coupler_dc_bias = 0.0
    
    ## Initial array declaration
    
    # Declare what phases are available
    phases_declared = np.linspace(0, 2*np.pi, 512)
    
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
        dry_run      =   True #False
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_iswap / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_iswap / plo_clk_T)) * plo_clk_T
        
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
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler port mixer
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = (coupler_dc_port == []),
        )
        if coupler_dc_port != []:
            pls.hardware.configure_mixer(
                freq      = 0.0,
                out_ports = coupler_dc_port,
                tune      = True,
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
            scales          = coupler_ac_amp_iswap,
        )
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            scales          = coupler_ac_amp_cz20,
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
        readout_freq_if_A = readout_freq_nco - readout_freq_A
        readout_freq_if_B = readout_freq_nco - readout_freq_B
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 0,
            frequencies  = np.abs(readout_freq_if_A),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if_A),
        )
        pls.setup_freq_lut(
            output_ports = readout_stimulus_port,
            group        = 1,
            frequencies  = np.abs(readout_freq_if_B),
            phases       = 0.0,
            phases_q     = bandsign(readout_freq_if_B)
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
        
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
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
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_A)),
            phases       = phases_declared,
            phases_q     = phases_declared + bandsign(control_freq_if_01_A),
        )
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_B)),
            phases       = phases_declared,
            phases_q     = phases_declared + bandsign(control_freq_if_01_B),
        )
        
        
        ## Set up the iSWAP gate pulse
        
        # Set up iSWAP envelope
        coupler_ac_duration_iswap = \
            coupler_ac_plateau_duration_iswap + \
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
        # Setup the iSWAP pulse carrier.
        coupler_ac_freq_if_iswap = coupler_ac_freq_nco - coupler_ac_freq_iswap
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 0,
            frequencies     = np.abs(coupler_ac_freq_if_iswap),
            phases          = phase_adjustment_coupler_ac_iswap, # TODO NOT FINISHED, this shouldn't work.
            phases_q        = phase_adjustment_coupler_ac_iswap + bandsign(coupler_ac_freq_if_iswap), # TODO NOT FINISHED, this shouldn't work.
        )
        
        
        ## Set up the CZ20 gate pulse
        
        # Set up CZ20 envelope
        coupler_ac_duration_cz20 = \
            coupler_ac_plateau_duration_cz20 + \
            2 * coupler_ac_single_edge_time_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 1,
            duration    = coupler_ac_duration_cz20,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_cz20,
            fall_time   = coupler_ac_single_edge_time_cz20,
        )
        # Setup the CZ20 pulse carrier.
        coupler_ac_freq_if_cz20 = coupler_ac_freq_nco - coupler_ac_freq_cz20
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = np.abs(coupler_ac_freq_if_cz20),
            phases          = phase_adjustment_coupler_ac_cz20, # TODO NOT FINISHED, this shouldn't work.
            phases_q        = phase_adjustment_coupler_ac_cz20 + bandsign(coupler_ac_freq_if_cz20), # TODO NOT FINISHED, this shouldn't work.
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
        
            # Redefine the coupler DC pulse duration.
            coupler_bias_tone.set_total_duration(
                quantum_circuit[-1][0][3] # Get the longest channel's length.
            )
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        ## TODO Some kind of loop counter should go here
        ##      in order to assure that the phase remains coherent.
        
        # Get a time reference, used for gauging the iteration length.
        T_begin = T
        
        # (Re-)apply the coupler bias tone.
        if coupler_dc_port != []:
            pls.output_pulse(T, coupler_bias_tone)
        
        # Reset phases
        pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
        phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
        phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
        
        # Schedule gates from the quantum circuit.
        for all_barrier_moments in quantum_circuit:
            for operation_code in all_barrier_moments:
                
                # Get current operation's scheduled execution time.
                T = operation_code[3]
                
                # Do things depending on the operation.
                # The most common gates expected are (in order):
                #    Rotate-Z, sqrt(X), Pauli-X, iSWAP, CZ, Measure
                if operation_code[0] == 'rz':
                    # In the 'rz' instruction, we expect a parameter showing
                    # how much phase to rotate with. This parameter is seen
                    # in operation_code[1][0]
                    if operation_code[2][0] == 0: # Qubit A or B for this interface platform?
                        add_virtual_z(T, phase_A, operation_code[1][0], control_port_A, 0, phases_declared, pls)
                    else:
                        add_virtual_z(T, phase_B, operation_code[1][0], control_port_B, 0, phases_declared, pls)
                
                elif operation_code[0] == 'sx':
                    if operation_code[2][0] == 0: # Qubit A or B for this interface platform?
                        pls.output_pulse(T, control_pulse_pi_01_half_A)
                    else:
                        pls.output_pulse(T, control_pulse_pi_01_half_B)
                
                elif operation_code[0] == 'x':
                    if operation_code[2][0] == 0: # Qubit A or B for this interface platform?
                        pls.output_pulse(T, control_pulse_pi_01_A)
                    else:
                        pls.output_pulse(T, control_pulse_pi_01_B)
                
                elif operation_code[0] == 'iswap':
                    pls.output_pulse(T, coupler_ac_pulse_iswap)
                    add_virtual_z(T, phase_A, phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
                    add_virtual_z(T, phase_B, phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
                    # TODO How does one phase-adjust the coupler tone? Not done yet.
                
                elif operation_code[0] == 'cz':
                    pls.output_pulse(T, coupler_ac_pulse_cz20)
                    add_virtual_z(T, phase_A, phase_adjustment_after_cz20_A, control_port_A, 0, phases_declared, pls)
                    add_virtual_z(T, phase_B, phase_adjustment_after_cz20_B, control_port_B, 0, phases_declared, pls)
                
                elif operation_code[0] == 'measure':
                    # Commence multiplexed readout
                    pls.reset_phase(T, readout_stimulus_port)
                    pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                    pls.store(T + readout_sampling_delay) # Sampling window
                    T += readout_duration
                    
                elif operation_code[0] == 'total_duration':
                    # print('Finished scheduling gate sequence, will commence execution.')
                    pass
                
                else:
                    raise NotImplementedError("Error! An unknown gate operation was encountered")
        
        # Get our last time reference
        T_last = T
        
        # Is the iteration longer than the repetition delay?
        if (T_last - T_begin) >= repetition_delay:
            while (T_last - T_begin) >= repetition_delay:
                # If this happens, then the iteration does not fit within
                # one decreed repetion period.
                T = repetition_delay * repetition_counter
                repetition_counter += 1
                
                # Move reference
                T_begin = T
        else:
            # Then all is good.
            T = repetition_delay * repetition_counter
            repetition_counter += 1
        
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
    
    assert 1 == 0, "HALTED, NOT FINISHED" #TODO
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        
        print("Saving data")
        
        ###########################################
        ''' SAVE AS LOG BROWSER COMPATIBLE HDF5 '''
        ###########################################
        
        # Data to be stored.
        hdf5_steps = [
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
            'coupler_ac_freq_nco', "Hz",
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
                    hdf5_logs.append('Probability for state |'+statep+'⟩')
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
            inner_loop_size = 1, # TODO Yeah how should this be done... Remains to be seen soon.
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            append_to_log_name_before_timestamp = 'quantum_circuit',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
        ))
    
    return string_arr_to_return
    