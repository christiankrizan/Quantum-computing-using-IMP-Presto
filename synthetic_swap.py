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
from bias_calculator import \
    sanitise_dc_bias_arguments, \
    get_dc_dac_range_integer, \
    change_dc_bias
from time_calculator import \
    check_if_integration_window_is_legal, \
    get_timestamp_string
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save
from single_qubit_gates import \
    hadamard_gate, \
    s_dagger_gate


def iswap_then_cz20_conditional_cross_ramsey(
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
    repetition_rate,
    integration_window_start,
    integration_window_stop,
    
    control_port_A,
    control_freq_nco_A,
    control_freq_01_A,
    control_amp_01_A,
    control_port_B,
    control_freq_nco_B,
    control_freq_01_B,
    control_amp_01_B,
    control_duration_01,
    
    coupler_dc_port,
    coupler_dc_bias,
    settling_time_of_bias_tee,
    
    coupler_ac_port,
    coupler_ac_freq_nco,
    
    coupler_ac_freq_cz20,
    coupler_ac_amp_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    
    coupler_ac_freq_iswap,
    coupler_ac_amp_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    
    num_averages,
    
    num_phases,
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    phase_adjustment_after_cz20_A = 0.0,
    phase_adjustment_after_cz20_B = 0.0,
    phase_adjustment_after_cz20_C = 0.0,
    phase_adjustment_after_iswap_A = 0.0,
    phase_adjustment_after_iswap_B = 0.0,
    phase_adjustment_after_iswap_C = 0.0,
    
    prepare_input_state = '0+',
    
    reset_dc_to_zero_when_finished = True,
    
    save_complex_data = True,
    save_raw_time_data = False,
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
    ''' Demonstrate that iSWAP followed by CZ₂₀ behaves like a SWAP gate.
        However, this statement does not hold for every input state.
        SWAP:   --x--
                  |  
                --x--
                
                =
                
                ----- --o-- --H-- --o-- --H-- --o-- -----
                        |           |           |        
                --H-- --o-- --H-- --o-- --H-- --o-- --H--
                
                =
                
                --H-- --S†- -|X|- --o-- -----
                              |     |        
                --S†- ----- -|X|- --o-- --H--
        
        ... where:
        →  S† = S-dagger
        →  H  = Hadamard
        → |X| = iSWAP
        →  o  = CZ₂₀
        
        ... thus, also perform the needed single-qubit gates every
        time a SWAP gate is requested.
        
        Legal input states for the cross-Ramsey scheme include:
        → '0+'
        → '1+'
        → '+0'
        → '+1'
        
        Even though the measurement is the same as a conditional cross-Ramsey
        experiment, note that the SWAP gate is not conditional. Which,
        is the point of this experiment; to show that there is no difference
        whether you initialise |0+⟩ or |1+⟩. The |1⟩ state gets swapped
        onto the other qubit.
        
        The quantum circuit for this measurement is the following:
        →   Initialise in |0+⟩, |1+⟩, |+0⟩ or |+1⟩
        →   Perform the SWAP gate equivalent.
            The SWAP equivalent is SWAP( 𝛙 ) = iSWAP(CZ₂₀( 𝛙 ))
                BUT with the added condition that that equivalent
                is not true for every input state, hence the extra
                S† and H gates.
        →   On the qubit NOT initially prepared in |+⟩, now perform
            a final π/2 gate. NOTE: this final π/2 gate, is phase-swept
            using a virtual-Z gate.
        →   Readout. The expected output as the phase of the final
            π/2 gate is swept, is a Ramsey-styled measurement without decay.
            
        The point of the measurement is to show that
        the SWAP simply SWAPs away whatever state the
        "conditional" qubit was in, and replaces it with the |+⟩ state
        from the other qubit.
        
        repetition_rate is the time multiple at which every single
        measurement is repeated at. Example: a repetition rate of 300 µs
        means that single iteration of a measurement ("a shot") begins anew
        every 300 µs. If the measurement itself cannot fit into a 300 µs
        window, then the next iteration will happen at the next integer
        multiple of 300 µs.
    '''
    
    ## Input sanitisation
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    ## Assure that the requested input state to prepare, is valid.
    assert ( \
        (prepare_input_state == '0+') or \
        (prepare_input_state == '+0') or \
        (prepare_input_state == '1+') or \
        (prepare_input_state == '+1') ),\
        "Error! Invalid request for input state to prepare. " + \
        "Legal values are \'0+\', \'+0\', \'1+\' and \'+1\'"
    
    ## Initial array declaration
    
    # Declare what phases are available
    phases_declared = np.linspace(0, 2*np.pi, 512)
    
    # Declare phase array to sweep, and make it legal.
    control_phase_arr = np.linspace(phase_sweep_rad_min, phase_sweep_rad_max, num_phases)
    control_phase_arr = legalise_phase_array( control_phase_arr, phases_declared )
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   False,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G10, DacFSample.G6, DacFSample.G6],
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
        pls.hardware.set_dac_current(coupler_ac_port, 40_500)
        pls.hardware.set_inv_sinc(coupler_ac_port, 0)
        
        # Configure the DC bias. Also, let's charge the bias-tee.
        if coupler_dc_port != []:
            pls.hardware.set_dc_bias( \
                coupler_dc_bias, \
                coupler_dc_port, \
                range_i = get_dc_dac_range_integer(coupler_dc_bias) \
            )
            time.sleep( settling_time_of_bias_tee )
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        readout_duration  = int(round(readout_duration / plo_clk_T)) * plo_clk_T
        sampling_duration = int(round(sampling_duration / plo_clk_T)) * plo_clk_T
        readout_sampling_delay = int(round(readout_sampling_delay / plo_clk_T)) * plo_clk_T
        repetition_rate = int(round(repetition_rate / plo_clk_T)) * plo_clk_T
        control_duration_01 = int(round(control_duration_01 / plo_clk_T)) * plo_clk_T
        settling_time_of_bias_tee = int(round(settling_time_of_bias_tee / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_single_edge_time_iswap = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_iswap = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
        ''' Setup mixers '''
        
        # Readout port mixer
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
            scales          = coupler_ac_amp_cz20,
        )
        pls.setup_scale_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            scales          = coupler_ac_amp_iswap,
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
            phases_q     = bandsign(readout_freq_if_B),
        )
        
        ### Setup pulses:
        ### "control_pulse_pi_01_A" and "control_pulse_pi_01_B"
        ### "control_pulse_pi_01_A_half" and "control_pulse_pi_01_B_half"
        
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
        control_pulse_pi_01_half_A = pls.setup_template(
            output_port = control_port_A,
            group       = 0,
            template    = control_envelope_01/2, # Halved!
            template_q  = control_envelope_01/2, # Halved!
            envelope    = True,
        )
        control_pulse_pi_01_B = pls.setup_template(
            output_port = control_port_B,
            group       = 0,
            template    = control_envelope_01,
            template_q  = control_envelope_01,
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
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_A)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_A))
        )
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.full_like(phases_declared, np.abs(control_freq_if_01_B)),
            phases       = phases_declared,
            phases_q     = phases_declared + np.full_like(phases_declared, bandsign(control_freq_if_01_B))
        )
        
        
        ## Set up the CZ20 gate pulse
        
        # Set up CZ20 envelope
        coupler_ac_duration_cz20 = \
            coupler_ac_plateau_duration_cz20 + \
            2 * coupler_ac_single_edge_time_cz20
        coupler_ac_pulse_cz20 = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 0,
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
            group           = 0,
            frequencies     = np.full_like(phases_declared, np.abs(coupler_ac_freq_if_cz20)),
            phases          = phases_declared,
            phases_q        = phases_declared + np.full_like(phases_declared, bandsign(coupler_ac_freq_if_cz20))
        )
        
        
        ### Setup the iSWAP gate pulse
        
        # Set up iSWAP envelope
        coupler_ac_duration_iswap = \
            coupler_ac_plateau_duration_iswap + \
            2 * coupler_ac_single_edge_time_iswap
        coupler_ac_pulse_iswap = pls.setup_long_drive(
            output_port = coupler_ac_port,
            group       = 1,
            duration    = coupler_ac_duration_iswap,
            amplitude   = 1.0,
            amplitude_q = 1.0,
            rise_time   = coupler_ac_single_edge_time_iswap,
            fall_time   = coupler_ac_single_edge_time_iswap
        )
        # Setup LUT
        coupler_ac_freq_iswap_if = coupler_ac_freq_nco - coupler_ac_freq_iswap
        pls.setup_freq_lut(
            output_ports    = coupler_ac_port,
            group           = 1,
            frequencies     = np.full_like(phases_declared, np.abs(coupler_ac_freq_iswap_if)),
            phases          = phases_declared,
            phases_q        = phases_declared + np.full_like(phases_declared, bandsign(coupler_ac_freq_iswap_if))
        )
        
        ### Setup sampling window ###
        pls.set_store_ports(readout_sampling_port)
        pls.set_store_duration(sampling_duration)
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0  # s
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # For every phase value of the final pi-half gate:
        for ii in range(num_phases):
            
            # Get a time reference, used for gauging the iteration length.
            T_begin = T
            
            # Reset phases.
            pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
            phase_A = reset_phase_counter(T, control_port_A, 0, phases_declared, pls)
            phase_B = reset_phase_counter(T, control_port_B, 0, phases_declared, pls)
            phase_C = reset_phase_counter(T, coupler_ac_port, None, phases_declared, pls)
            
            # Prepare the sought-for input state.
            # There are four options: |0+>, |+0>, |1+>, |+1>
            if prepare_input_state[-1] == '+':
                
                # Qubit B is initially prepared in the mixed state.
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                
                # Is the conditional qubit supposed to be on?
                if prepare_input_state[0] == '1':
                    # It is supposed to be on!
                    pls.output_pulse(T, control_pulse_pi_01_A)
                
                # Go to the next quantum circuit moment.
                T += control_duration_01
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
                phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
                
            else:
                
                # Qubit B is NOT initially prepared in the mixed state,
                # meaning that qubit A is prepared in the mixed state.
                pls.output_pulse(T, control_pulse_pi_01_half_A)
                
                # Is the conditional qubit supposed to be on?
                if prepare_input_state[1] == '1':
                    # It is supposed to be on!
                    pls.output_pulse(T, control_pulse_pi_01_B)
                
                # Go to the next quantum circuit moment.
                T += control_duration_01
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
                phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            #############   < Apply SWAP-gate equivalent >  #############
            
            # Apply Hadamard gate on the qubit which was initially
            # NOT prepared in |+⟩. And, an S† gate on the other qubit.
            # However, which qubit is which is supposed to not matter.
            # These H and S† gates are here to really make the SWAP equivalent
            # valid for any input state.
            if prepare_input_state[-1] == '+':
                # We're doing H(S†( qubit_A )) and S†( qubit_B )
                T_moment = T
                T_after_H, phase_A = hadamard_gate(
                    starting_time_of_current_measurement_iteration = T_begin,
                    execute_gate_at_this_time = T_moment,
                    
                    phase_of_qubit = phase_A,
                    pi_pulse_object = control_pulse_pi_01_A,
                    pi_half_pulse_object = control_pulse_pi_01_half_A,
                    duration_of_pi_pulse = control_duration_01,
                    duration_of_pi_half_pulse = control_duration_01,
                    rf_carrier_frequency_of_qubit = control_freq_01_A,
                    
                    port = control_port_A,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                T_after_Sdg, phase_B = s_dagger_gate(
                    execute_gate_at_this_time = T_moment,
                    
                    phase_of_qubit = phase_B,
                    port = control_port_B,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                
                # Set the current time following this moment. Track phases!
                T = np.max( [T_after_H, T_after_Sdg] )
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
                phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
                
                # Do final S† gate on the qubit NOT initially prepared in |+⟩
                T, phase_A = s_dagger_gate(
                    execute_gate_at_this_time = T,
                    
                    phase_of_qubit = phase_A,
                    port = control_port_A,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                # The S†-gate is instantaneous, no need to track phases.
            
            else:
                # We're doing H(S†( qubit_B )) and S†( qubit_A )
                T_moment = T
                T_after_H, phase_B = hadamard_gate(
                    starting_time_of_current_measurement_iteration = T_begin,
                    execute_gate_at_this_time = T_moment,
                    
                    phase_of_qubit = phase_B,
                    pi_pulse_object = control_pulse_pi_01_B,
                    pi_half_pulse_object = control_pulse_pi_01_half_B,
                    duration_of_pi_pulse = control_duration_01,
                    duration_of_pi_half_pulse = control_duration_01,
                    rf_carrier_frequency_of_qubit = control_freq_01_B,
                    
                    port = control_port_B,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                T_after_Sdg, phase_A = s_dagger_gate(
                    execute_gate_at_this_time = T_moment,
                    
                    phase_of_qubit = phase_A,
                    port = control_port_A,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                
                # Set the current time following this moment. Track phases!
                T = np.max( [T_after_H, T_after_Sdg] )
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
                phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
                
                # Do final S† gate on the qubit NOT initially prepared in |+⟩
                T, phase_B = s_dagger_gate(
                    execute_gate_at_this_time = T,
                    
                    phase_of_qubit = phase_B,
                    port = control_port_B,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                # The S†-gate is instantaneous, no need to track phases.
            
            # Apply iSWAP gate
            pls.output_pulse(T, coupler_ac_pulse_iswap)
            T += coupler_ac_duration_iswap
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_iswap, phase_C)
            
            # Add the local phase correction following the iSWAP gate
            raise NotImplementedError("Halted! No coupler phase adjustment before iSWAP.")
            phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
            phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
            phase_C = add_virtual_z(T, phase_C, -phase_adjustment_after_iswap_C, coupler_ac_port, None, phases_declared, pls)
            
            # Apply CZ₂₀ gate
            pls.output_pulse(T, coupler_ac_pulse_cz20)
            T += coupler_ac_duration_cz20
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            phase_C = track_phase(T - T_begin, coupler_ac_freq_cz20, phase_C)
            
            # Add the local phase correction following the CZ₂₀ gate
            phase_A = add_virtual_z(T, phase_A, -phase_adjustment_after_iswap_A, control_port_A, 0, phases_declared, pls)
            phase_B = add_virtual_z(T, phase_B, -phase_adjustment_after_iswap_B, control_port_B, 0, phases_declared, pls)
            ##phase_C = add_virtual_z(T, phase_C, -phase_adjustment_after_iswap_C, coupler_ac_port, None, phases_declared, pls)
            
            # And, as art of making a SWAP gate that is legal for any input
            # state, we apply a final Hadamard gate on the qubit NOT initially
            # prepared in |+⟩.
            if prepare_input_state[-1] == '+':
                # We're doing H( qubit_B )
                T, phase_B = hadamard_gate(
                    starting_time_of_current_measurement_iteration = T_begin,
                    execute_gate_at_this_time = T,
                    
                    phase_of_qubit = phase_B,
                    pi_pulse_object = control_pulse_pi_01_B,
                    pi_half_pulse_object = control_pulse_pi_01_half_B,
                    duration_of_pi_pulse = control_duration_01,
                    duration_of_pi_half_pulse = control_duration_01,
                    rf_carrier_frequency_of_qubit = control_freq_01_B,
                    
                    port = control_port_B,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                # Track phases!
                phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
                ##phase_C = track_phase(T - T_begin, coupler_ac_freq_cz20, phase_C)
                
            else:
                # We're doing H( qubit_A )
                T, phase_A = hadamard_gate(
                    starting_time_of_current_measurement_iteration = T_begin,
                    execute_gate_at_this_time = T,
                    
                    phase_of_qubit = phase_A,
                    pi_pulse_object = control_pulse_pi_01_A,
                    pi_half_pulse_object = control_pulse_pi_01_half_A,
                    duration_of_pi_pulse = control_duration_01,
                    duration_of_pi_half_pulse = control_duration_01,
                    rf_carrier_frequency_of_qubit = control_freq_01_A,
                    
                    port = control_port_A,
                    group_holding_phase_LUT = 0,
                    phases_declared = phases_declared,
                    pulse_object = pls,
                    
                    virtual_z_will_come_right_after_this_gate = False,
                )
                # Track phases!
                phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
                ##phase_C = track_phase(T - T_begin, coupler_ac_freq_cz20, phase_C)
            
            #############  </ Apply SWAP-gate equivalent >  #############
            
            # At this point, we've already applied the virtual-Z gate that
            # gives us our cross-Ramsey-like behaviour.
            
            # Apply π/2 gate on the qubit NOT initially prepared in |+⟩
            if prepare_input_state[-1] == '+':
                pls.output_pulse(T, control_pulse_pi_01_half_A)
            else:
                pls.output_pulse(T, control_pulse_pi_01_half_B)
            T += control_duration_01
            phase_A = track_phase(T - T_begin, control_freq_01_A, phase_A)
            phase_B = track_phase(T - T_begin, control_freq_01_B, phase_B)
            ##phase_C = track_phase(T - T_begin, coupler_ac_freq_cz20, phase_C)
            
            # Commence multiplexed readout
            pls.reset_phase(T, readout_stimulus_port)
            pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
            pls.store(T + readout_sampling_delay) # Sampling window
            T += readout_duration
            
            # Get T that aligns with the repetition rate.
            T, repetition_counter = get_repetition_rate_T(
                T_begin, T, repetition_rate, repetition_counter,
            )
        
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
        
        # Reset the DC bias port(s).
        if (coupler_dc_port != []) and reset_dc_to_zero_when_finished:
            pls.hardware.set_dc_bias(0.0, coupler_dc_port)
    
    # Declare path to whatever data will be saved.
    string_arr_to_return = []
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        print("Raw data downloaded to PC.")

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
            'readout_freq_nco', "Hz",
            'readout_freq_A', "Hz",
            'readout_amp_A', "FS",
            'readout_freq_B', "Hz",
            'readout_amp_B', "FS",
            'readout_duration', "s",
            
            'sampling_duration', "s",
            'readout_sampling_delay', "s",
            'repetition_rate', "s",
            'integration_window_start', "s",
            'integration_window_stop', "s",
            
            'control_port_A', "",
            'control_freq_nco_A', "Hz",
            'control_freq_01_A', "Hz",
            'control_amp_01_A', "FS",
            'control_port_B', "",
            'control_freq_nco_B', "Hz",
            'control_freq_01_B', "Hz",
            'control_amp_01_B', "FS",
            'control_duration_01', "s",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "V",
            'settling_time_of_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_nco', "Hz",
            
            'coupler_ac_freq_cz20', "Hz",
            'coupler_ac_amp_cz20', "FS",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            
            'coupler_ac_freq_iswap', "Hz",
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap', "s",
            
            'num_averages', "",
            
            'num_phases', "",
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
            
            'phase_adjustment_after_cz20_A', "rad",
            'phase_adjustment_after_cz20_B', "rad",
            'phase_adjustment_after_cz20_C', "rad",
            'phase_adjustment_after_iswap_A', "rad",
            'phase_adjustment_after_iswap_B', "rad",
            'phase_adjustment_after_iswap_C', "rad",
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
            inner_loop_size = num_phases,
            outer_loop_size = 1,
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = '_cz_following_iswap_conditional_cross_Ramsey',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return



    
def synthetic_swap_prep_10_cross_Ramsey_DEPRECATED(
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
    control_amp_01_Vz_A,
    control_freq_01_A,
    control_amp_01_B,
    ##control_amp_01_Vz_B
    control_freq_01_B,
    control_duration_01,
    
    control_freq_12_A,
    control_freq_12_B,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    coupler_ac_port,
    coupler_ac_freq_nco,
    
    coupler_ac_amp_iswap,
    coupler_ac_freq_iswap,
    coupler_ac_single_edge_time_iswap,
    coupler_ac_plateau_duration_iswap,
    
    coupler_ac_amp_cz20,
    coupler_ac_freq_cz20,
    coupler_ac_single_edge_time_cz20,
    coupler_ac_plateau_duration_cz20,
    
    phase_adjustment_coupler_ac_iswap,
    phase_adjustment_coupler_ac_cz20,
    
    num_averages,
    
    num_phases,    
    phase_sweep_rad_min = 0.0,
    phase_sweep_rad_max = 6.2831853071795864769252867665590057683943387987502116419498891846,
    
    reset_dc_to_zero_when_finished = True,
    
    save_complex_data = True,
    save_raw_time_data = False,
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
    ''' Prepare |10>, run a cross-Ramsey experiment.
        The SWAP gate is made by running both the CZ and the iSWAP
        gates simultaneously.
    '''
    
    assert 1 == 0, "Halted! This function's DC biasing has not been modernised."
    
    ## Input sanitisation
    
    # DC bias argument sanitisation.
    coupler_bias_min, coupler_bias_max, num_biases, coupler_dc_bias, \
    with_or_without_bias_string = sanitise_dc_bias_arguments(
        coupler_dc_port  = coupler_dc_port,
        coupler_bias_min = None,
        coupler_bias_max = None,
        num_biases       = None,
        coupler_dc_bias  = coupler_dc_bias
    )
    
    ## Initial array declaration
    
    # Declare what phases are available
    phases_declared = np.linspace(0, 2*np.pi, 512)
    
    # Declare phase array to sweep, and make it legal.
    control_phase_arr = np.linspace(phase_sweep_rad_min, phase_sweep_rad_max, num_phases)
    control_phase_arr = legalise_phase_array( control_phase_arr, phases_declared )
    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        force_reload =   False,
        address      =   ip_address,
        ext_ref_clk  =   ext_clk_present,
        adc_mode     =   AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample  =   AdcFSample.G2,  # 2 GSa/s
        dac_mode     =   [DacMode.Mixed42, DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample  =   [DacFSample.G10, DacFSample.G10, DacFSample.G6, DacFSample.G6],
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
        coupler_ac_single_edge_time_cz20 = int(round(coupler_ac_single_edge_time_cz20 / plo_clk_T)) * plo_clk_T
        coupler_ac_plateau_duration_cz20 = int(round(coupler_ac_plateau_duration_cz20 / plo_clk_T)) * plo_clk_T
        
        # Check whether the integration window is legal.
        integration_window_stop = check_if_integration_window_is_legal(
            sample_rate = 1e9,
            sampling_duration = sampling_duration,
            integration_window_start = integration_window_start,
            integration_window_stop  = integration_window_stop
        )
        
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
        high_res_A  = max( [control_freq_01_A, control_freq_12_A] )
        low_res_A   = min( [control_freq_01_A, control_freq_12_A] )
        control_freq_nco_A = high_res_A - (high_res_A - low_res_A)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_A,
            out_ports = control_port_A,
            tune      = True,
            sync      = False,
        )
        high_res_B  = max( [control_freq_01_B, control_freq_12_B] )
        low_res_B   = min( [control_freq_01_B, control_freq_12_B] )
        control_freq_nco_B = high_res_B - (high_res_B - low_res_B)/2 -250e6
        pls.hardware.configure_mixer(
            freq      = control_freq_nco_B,
            out_ports = control_port_B,
            tune      = True,
            sync      = False,
        )
        # Coupler ports, calculate an optimal NCO frequency.
        ##high_gate_f  = max( [coupler_ac_freq_iswap, coupler_ac_freq_cz20] ) ## TODO Not working, await IMP patch.
        ##low_gate_f   = min( [coupler_ac_freq_iswap, coupler_ac_freq_cz20] ) ## TODO Not working, await IMP patch.
        ##coupler_ac_freq_nco = high_gate_f - (high_gate_f - low_gate_f)/2 -250e6  ## TODO Not working, await IMP patch.
        pls.hardware.configure_mixer(
            freq      = coupler_ac_freq_nco,
            out_ports = coupler_ac_port,
            tune      = True,
            sync      = (coupler_dc_port == []),
        )
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
        ''' Almost the same thing for the Vz-swept gates '''
        pls.setup_scale_lut(
            output_ports    = control_port_A,
            group           = 1,
            scales          = control_amp_01_Vz_A,
        )
        ##pls.setup_scale_lut(
        ##    output_ports    = control_port_B,
        ##    group           = 1,
        ##    scales          = control_amp_01_Vz_B,
        ##)
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
            phases_q     = bandsign(readout_freq_if_B),
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
        ##control_pulse_pi_01_B = pls.setup_template(
        ##    output_port = control_port_B,
        ##    group       = 0,
        ##    template    = control_envelope_01/2, # Halved!
        ##    template_q  = control_envelope_01/2, # Halved!
        ##    envelope    = True,
        ##)
        
        ##control_pulse_pi_01_half_A = pls.setup_template(
        ##    output_port = control_port_A,
        ##    group       = 0,
        ##    template    = control_envelope_01/2, # Halved!
        ##    template_q  = control_envelope_01/2, # Halved!
        ##    envelope    = True,
        ##)
        control_pulse_pi_01_half_Vz_A = pls.setup_template(
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
        ##control_pulse_pi_01_half_Vz_B = pls.setup_template(
        ##    output_port = control_port_B,
        ##    group       = 1,
        ##    template    = control_envelope_01/2, # Halved!
        ##    template_q  = control_envelope_01/2, # Halved!
        ##    envelope    = True,
        ##)
        
        
        # Setup control pulse carrier tones, considering that there is a digital mixer
        control_freq_if_01_A = control_freq_nco_A - control_freq_01_A
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_A),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_A),
        )
        pls.setup_freq_lut(
            output_ports = control_port_A,
            group        = 1,
            frequencies  = np.full_like(control_phase_arr, np.abs(control_freq_if_01_A)),
            phases       = control_phase_arr,
            phases_q     = control_phase_arr + bandsign(control_freq_if_01_A),
        )
        control_freq_if_01_B = control_freq_nco_B - control_freq_01_B
        pls.setup_freq_lut(
            output_ports = control_port_B,
            group        = 0,
            frequencies  = np.abs(control_freq_if_01_B),
            phases       = 0.0,
            phases_q     = bandsign(control_freq_if_01_B),
        )
        ##pls.setup_freq_lut(
        ##    output_ports = control_port_B,
        ##    group        = 1,
        ##    frequencies  = np.full_like(control_phase_arr, np.abs(control_freq_if_01_B)),
        ##    phases       = control_phase_arr,
        ##    phases_q     = control_phase_arr + bandsign(control_freq_if_01_B),
        ##)
        
        
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
            phases          = -phase_adjustment_coupler_ac_iswap,
            phases_q        = -phase_adjustment_coupler_ac_iswap + bandsign(coupler_ac_freq_if_iswap),
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
            phases          = -phase_adjustment_coupler_ac_cz20,
            phases_q        = -phase_adjustment_coupler_ac_cz20 + bandsign(coupler_ac_freq_if_cz20),
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
            # Note that only one 2q-gate's worth of duration is added.
            coupler_bias_tone.set_total_duration(
                control_duration_01 + \
                coupler_ac_duration_iswap + \
                control_duration_01 + \
                readout_duration + \
                repetition_delay \
            )
        
        # Define repetition counter for T.
        repetition_counter = 1
        
        # For both cases, where the control qubit can either be on/off
        for turn_on_control_qubit in [True, False]:
        
            # For every phase value of the final pi-half gate:
            for ii in range(num_phases):
                
                # Get a time reference, used for gauging the iteration length.
                T_begin = T
                
                # Re-apply the coupler bias tone, to keep on playing once
                # the bias tee has charged.
                if coupler_dc_port != []:
                    pls.output_pulse(T, coupler_bias_tone)
                
                # Reset phases
                pls.reset_phase(T, [control_port_A, control_port_B, coupler_ac_port])
                
                # Prepare the input state
                if turn_on_control_qubit:
                    pls.output_pulse(T, control_pulse_pi_01_A)
                    ## T += control_duration_01
                    ## This delay is its seperate moment in this paper, I'm unsure why.
                    ## https://journals.aps.org/pra/pdf/10.1103/PhysRevA.102.062408
                
                # Set other qubit to |+>
                pls.output_pulse(T, control_pulse_pi_01_half_B)
                T += control_duration_01
                
                # Apply synthetic SWAP interaction, both gates simultaneously.
                pls.output_pulse(T, [coupler_ac_pulse_iswap, coupler_ac_pulse_cz20])
                T += coupler_ac_duration_cz20  # Same as coupler_ac_duration_iswap
                
                # Apply virtual-Z-swept pi/2 gate on qubit A.
                pls.output_pulse(T, control_pulse_pi_01_half_Vz_A)
                T += control_duration_01
                
                # Commence multiplexed readout
                pls.reset_phase(T, readout_stimulus_port)
                pls.output_pulse(T, [readout_pulse_A, readout_pulse_B])
                pls.store(T + readout_sampling_delay) # Sampling window
                T += readout_duration
                
                # Move to next phase in the sweep
                pls.next_frequency(T, control_port_A, group = 1)
                
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
    
    if not pls.dry_run:
        time_vector, fetched_data_arr = pls.get_store_data()
        print("Raw data downloaded to PC.")

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
            'control_port_B', "",
            
            'control_amp_01_A', "FS",
            'control_freq_01_A', "Hz",
            'control_amp_01_B', "FS",
            'control_freq_01_B', "Hz",
            'control_duration_01', "s",
            
            'control_freq_12_A', "Hz",
            'control_freq_12_B', "Hz",
            
            #'coupler_dc_port', "",
            'coupler_dc_bias', "FS",
            'added_delay_for_bias_tee', "s",
            
            'coupler_ac_port', "",
            'coupler_ac_freq_nco', "Hz",
            
            'coupler_ac_amp_iswap', "FS",
            'coupler_ac_freq_iswap', "Hz",
            'coupler_ac_single_edge_time_iswap', "s",
            'coupler_ac_plateau_duration_iswap', "s",
            
            'coupler_ac_amp_cz20', "FS",
            'coupler_ac_freq_cz20', "Hz",
            'coupler_ac_single_edge_time_cz20', "s",
            'coupler_ac_plateau_duration_cz20', "s",
            
            'phase_adjustment_coupler_ac_iswap', "rad",
            'phase_adjustment_coupler_ac_cz20', "rad",
            
            'num_averages', "",
            
            'num_phases', "",
            'phase_sweep_rad_min', "rad",
            'phase_sweep_rad_max', "rad",
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
            inner_loop_size = num_phases,
            outer_loop_size = 2, # One for |00> and one for |10> as the input state.
            
            save_complex_data = save_complex_data,
            source_code_of_executing_file = '', #get_sourcecode(__file__),
            default_exported_log_file_name = default_exported_log_file_name,
            append_to_log_name_before_timestamp = 'cross_Ramsey',
            append_to_log_name_after_timestamp  = '',
            select_resonator_for_single_log_export = '',
            
            suppress_log_browser_export = suppress_log_browser_export,
            log_browser_tag  = log_browser_tag,
            log_browser_user = log_browser_user,
            save_raw_time_data = save_raw_time_data,
        ))
    
    return string_arr_to_return