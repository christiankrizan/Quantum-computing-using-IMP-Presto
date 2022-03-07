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
import Labber
import shutil
import numpy as np
from datetime import datetime
from log_browser_exporter import save

def get_iq_for_readout_optimisation_states_g_e_f(
    ip_address,
    ext_clk_present,
    
    readout_stimulus_port,
    readout_sampling_port,
    readout_freq_g,
    readout_freq_e,
    readout_freq_f,
    readout_amp,
    readout_duration,
    
    sampling_duration,
    readout_sampling_delay,
    repetition_delay,
    integration_window_start,
    integration_window_stop,
    
    control_port,
    control_amp_01,
    control_freq_01,
    control_duration_01,
    control_amp_12,
    control_freq_12,
    control_duration_12,
    
    coupler_dc_port,
    coupler_dc_bias,
    added_delay_for_bias_tee,
    
    num_shots,
    
    save_complex_data = False,
    use_log_browser_database = True,
    axes =  {
        "x_name":   'default',
        "x_scaler": 1.0,
        "x_unit":   'default',
        "y_name":   'default',
        "y_scaler": [1.0],
        "y_offset": [0.0],
        "y_unit":   'default',
        "z_name":   'default',
        "z_scaler": 1.0,
        "z_unit":   'default',
        }
    ):
    ''' Perform single-shot readouts for gauging where in the IQ plane
        one finds the |g> , |e> and |f> states.
    '''
    
    assert 1 == 0, "Not implemented."
    
    return string_arr_to_return