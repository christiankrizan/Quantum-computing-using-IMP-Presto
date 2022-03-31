#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import os
import time

import numpy as np
from numpy import hanning as von_hann

from presto import pulsed
from presto.utils import sin2, get_sourcecode
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

def outputSingleWave(
    ip_address,
    ext_clk_present,
    
    repetition_delay,
    
    control_port,
    control_freq,
    control_duration,
    
    num_amplitudes,
    num_averages,
    
    fake_readout_sampling_port,
    fake_sampling_duration,
    
    control_amp_min = 0.0,
    control_amp_max = 1.0,
    ):
    ''' Output a simple single waveform according to provided parameters.
    '''

    # Declare amplitude array for the Rabi experiment
    control_amp_arr = np.linspace(control_amp_min, control_amp_max, num_amplitudes)

    
    # Instantiate the interface
    print("\nConnecting to "+str(ip_address)+"...")
    with pulsed.Pulsed(
        address     = ip_address,
        ext_ref_clk = ext_clk_present,
        adc_mode    = AdcMode.Mixed,  # Use mixers for downconversion
        adc_fsample = AdcFSample.G2,  # 2 GSa/s
        dac_mode    = [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample = [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        dry_run     = False,
    ) as pls:
        print("Connected. Setting up...")
        #pls.hardware.set_adc_attenuation(readout_sampling_port, 0.0)
        #pls.hardware.set_dac_current(readout_stimulus_port, 40_500)
        #pls.hardware.set_dac_current(control_port, 40_500)
        #pls.hardware.set_inv_sinc(readout_stimulus_port, 0)
        pls.hardware.set_inv_sinc(control_port, 0)

        ''' Setup mixers '''
        
        # Control port mixer
        pls.hardware.configure_mixer(
            freq        =   control_freq,
            out_ports   =   control_port,
            sync        =   True,  # Sync here
        )


        ''' Setup scale LUTs '''
        
        # Control port amplitude sweep for pi
        pls.setup_scale_lut(
            output_ports=control_port,
            group=1,
            scales=control_amp_arr,
        )


        ### Setup pulse "control_pulse" ###

        # Setup control_pulse_pi pulse envelope
        control_ns = int(round(control_duration * pls.get_fs("dac")))  # Number of samples in the control template
        control_envelope = sin2(control_ns)
        control_pulse_pi = pls.setup_template(
            output_port = control_port,
            group       = 1,
            template    = control_envelope,
            template_q  = control_envelope,
            envelope    = True,
        )
        # Setup control_pulse_pi carrier tone, considering that there is a digital mixer
        pls.setup_freq_lut(
            output_ports = control_port,
            group        = 1,
            frequencies  = 0.0,
            phases       = 0.0,
            phases_q     = 0.0,
        )
        
        
        ###   TODO Setup a fake sampling window to   ###
        ### avoid a NoneType exception in Presto 2.0 ###
        pls.set_store_ports(fake_readout_sampling_port)
        pls.set_store_duration(fake_sampling_duration)
        

        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        # Start of sequence
        T = 0.0  # s
        
        # Pi pulse to be characterised
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse_pi)
        T += control_duration
        
        # Move to next amplitude
        pls.next_scale(T, control_port)
        
        # TODO Do a fake store
        pls.store(T) # Sampling window
        
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
        
        # TODO Get fake store data
        time_vector, fetched_data_arr = pls.get_store_data()
        