#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

'''from presto import pulsed
from presto.utils import sin2, get_sourcecode
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode

import os
import sys
import time
import shutil'''
import numpy as np
from numpy import hanning as von_hann
from phase_calculator import \
    legalise_phase_array, \
    reset_phase_counter, \
    add_virtual_z, \
    track_phase, \
    bandsign
'''from bias_calculator import \
    sanitise_dc_bias_arguments, \
    get_dc_dac_range_integer, \
    change_dc_bias
from repetition_rate_calculator import get_repetition_rate_T
from time_calculator import \
    check_if_integration_window_is_legal, \
    get_timestamp_string
from data_exporter import \
    ensure_all_keyed_elements_even, \
    stylise_axes, \
    get_dict_for_step_list, \
    get_dict_for_log_list, \
    save'''





def s_gate(
    execute_gate_at_this_time,
    
    phase_of_qubit,
    port,
    group_holding_phase_LUT,
    phases_declared,
    pulse_object,
    
    virtual_z_will_come_right_after_this_gate = False,
    ):
    ''' Using the provided pulse object, perform an S gate.
        An S gate is a +90° rotation about the Z-axis on the Bloch sphere.
    '''
    
    # Remake some arguments (TODO into pointers?)
    T = execute_gate_at_this_time
    
    # Do S-gate using virtual-Z.
    if (not virtual_z_will_come_right_after_this_gate):
        phase_of_qubit = add_virtual_z( \
            T, \
            phase_of_qubit, \
            +np.pi/2, \
            port, \
            group_holding_phase_LUT, \
            phases_declared, \
            pulse_object \
        )
    else:
        # Skip the set_frequency call. Instead, just update the notion
        # of where the qubit phase is supposed to be. The actual virtual-Z
        # gate call after this gate, will take care of the phase
        # update accordingly.
        phase_of_qubit += np.pi/2
        if phase_of_qubit >= 2*np.pi:
            phase_of_qubit -= 2*np.pi
    
    # Return current sequencer time and the current phase of the qubit.
    return T, phase_of_qubit

def s_dagger_gate(
    execute_gate_at_this_time,
    
    phase_of_qubit,
    port,
    group_holding_phase_LUT,
    phases_declared,
    pulse_object,
    
    virtual_z_will_come_right_after_this_gate = False,
    ):
    ''' Using the provided pulse object, perform an S† gate.
        An S† gate is a -90° rotation about the Z-axis on the Bloch sphere.
    '''
    
    # Remake some arguments (TODO into pointers?)
    T = execute_gate_at_this_time
    
    # Do S† gate using virtual-Z.
    if (not virtual_z_will_come_right_after_this_gate):
        phase_of_qubit = add_virtual_z( \
            T, \
            phase_of_qubit, \
            -np.pi/2, \
            port, \
            group_holding_phase_LUT, \
            phases_declared, \
            pulse_object \
        )
    else:
        # Skip the set_frequency call. Instead, just update the notion
        # of where the qubit phase is supposed to be. The actual virtual-Z
        # gate call after this gate, will take care of the phase
        # update accordingly.
        phase_of_qubit -= np.pi/2
        if phase_of_qubit < 0:
            phase_of_qubit += 2*np.pi
    
    # Return current sequencer time and the current phase of the qubit.
    return T, phase_of_qubit