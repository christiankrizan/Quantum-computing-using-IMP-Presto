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
from phase_calculator import bandsign
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_timestamp_string, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save

def output_pulse_sweep_frequency(
    ip_address,
    ext_clk_present,
    
    waveform_port,
    waveform_freq_nco,
    waveform_freq_centre,
    waveform_freq_span,
    waveform_amp,
    waveform_duration,
    
    repetition_delay,
    
    num_freqs,
    num_averages,
    
    fake_sampling_port,
    fake_sampling_duration,
    
    round_nco_freq_to_representable_number_to_avoid_phase_drift = False,
    
    ):
    ''' Output a simple single waveform according to provided parameters.
    '''
    
    ## Input sanitisation
    
    # Sanitisation for whether the user has a
    # span engaged but only a single frequency.
    if ((num_freqs == 1) and (waveform_freq_span != 0.0)):
        print("Note: single waveform frequency point requested, ignoring span parameter.")
        waveform_freq_span = 0.0
    
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
        pls.hardware.set_adc_attenuation(fake_sampling_port, 0.0)
        pls.hardware.set_dac_current(waveform_port, 40_500)
        pls.hardware.set_inv_sinc(waveform_port, 0)
        
        # Sanitise user-input time arguments
        plo_clk_T = pls.get_clk_T() # Programmable logic clock period.
        waveform_duration  = int(round(waveform_duration / plo_clk_T)) * plo_clk_T
        
        
        ''' Setup mixers '''
        
        # Waveform mixer
        pls.hardware.configure_mixer(
            freq      = waveform_freq_nco,
            in_ports  = fake_sampling_port,
            out_ports = waveform_port,
            tune      = round_nco_freq_to_representable_number_to_avoid_phase_drift,
            sync      = True,
        )
        
        
        ''' Setup scale LUTs '''
        
        # Waveform amplitude
        pls.setup_scale_lut(
            output_ports  = waveform_port,
            group         = 0,
            scales        = waveform_amp,
        )
        
        # Setup waveform pulse envelope
        waveform_pulse = pls.setup_long_drive(
            output_port =   waveform_port,
            group       =   0,
            duration    =   waveform_duration,
            amplitude   =   1.0,
            amplitude_q =   1.0,
            rise_time   =   0e-9,
            fall_time   =   0e-9
        )
        
        # Setup waveform carrier, this tone will be swept in frequency.
        # The user provides an intended span.
        waveform_freq_centre_if = waveform_freq_nco - waveform_freq_centre
        f_start = waveform_freq_centre_if - waveform_freq_span / 2
        f_stop  = waveform_freq_centre_if + waveform_freq_span / 2
        waveform_freq_if_arr = np.linspace(f_start, f_stop, num_freqs)
        
        # Use the appropriate sideband.
        waveform_pulse_freq_arr = waveform_freq_nco - waveform_freq_if_arr
        
        # Setup LUT
        pls.setup_freq_lut(
            output_ports = waveform_port,
            group        = 0,
            frequencies  = np.abs(waveform_freq_if_arr),
            phases       = np.full_like(waveform_freq_if_arr, 0.0),
            phases_q     = bandsign(waveform_freq_if_arr),
        )
        
        
        ###      Setup a fake sampling window to      ###
        ###   avoid a NoneType exception in the API   ###
        pls.set_store_ports(fake_sampling_port)
        pls.set_store_duration(fake_sampling_duration)
        
        
        #################################
        ''' PULSE SEQUENCE STARTS HERE'''
        #################################
        
        print("Outputting waveform on port "+str(waveform_port))
        
        # Start of sequence
        T = 0.0  # s
        
        # Pi pulse to be characterised
        pls.reset_phase(T, waveform_port)
        pls.output_pulse(T, waveform_pulse)
        T += waveform_duration
        
        # Move to next amplitude
        pls.next_frequency(T, waveform_port)
        
        # Do a fake store
        pls.store(T) # Sampling window
        
        # Wait for decay
        T += repetition_delay

        ################################
        ''' EXPERIMENT EXECUTES HERE '''
        ################################
        
        pls.run(
            period       = T,
            repeat_count = num_freqs,
            num_averages = num_averages,
            print_time   = True,
        )
        
        # Get fake store data
        time_vector, fetched_data_arr = pls.get_store_data()
        