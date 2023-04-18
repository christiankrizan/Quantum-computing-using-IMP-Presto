#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
from phase_calculator import \
    add_virtual_z, \
    track_phase

def hadamard_gate(
    starting_time_of_current_measurement_iteration,
    execute_gate_at_this_time,
    
    phase_of_qubit,
    pi_pulse_object,
    pi_half_pulse_object,
    duration_of_pi_pulse,
    duration_of_pi_half_pulse,
    rf_carrier_frequency_of_qubit,
    
    port,
    group_holding_phase_LUT,
    phases_declared,
    pulse_object,
    
    virtual_z_will_come_right_after_this_gate = False,
    ):
    ''' Using the provided pulse object, perform a Hadamard gate.
        A Hadamard gate is a 90° rotation about the Y-axis, followed
        by a 180° rotation about the X-axis.
        
        virtual_z_will_come_right_after_this_gate controls whether to
        apply the command apply_virtual_z in this hadamard gate function.
        If there always would have been a virtual-Z gate applied at the
        end of this hadamard gate function, then you'd might end up
        in a case such as an Rz gate following a Hadamard gate,
        meaning that two virtual-Z gates wouldbe scheduled at the same time T.
        Meaning that in the API, there would be two calls to select_frequency
        for the same time T. To avoid any mishaps, please set
        virtual_z_will_come_right_after_this_gate = True if there will be
        a virtual-Z applied right after the Hadamard gate.
    '''
    
    # Remake some arguments (TODO into pointers?)
    T = execute_gate_at_this_time
    T_begin = starting_time_of_current_measurement_iteration
    
    # Perform 90° rotation about the Y-axis.
    # Example: |0⟩ is rotated to |+⟩.
    pulse_object.output_pulse(T, pi_half_pulse_object)
    T += duration_of_pi_half_pulse
    phase_of_qubit = track_phase(T - T_begin, rf_carrier_frequency_of_qubit, phase_of_qubit)
    
    # Perform 180° rotation about the X axis.
    ## Such an action is done by rotating +π/2 radians about the Z axis,
    ## doing a full π-pulse, and finally rotating back with -π/2 radians.
    
    # Do virtual-Z
    phase_of_qubit = add_virtual_z( \
        T, \
        phase_of_qubit, \
        +np.pi/2, \
        port, \
        group_holding_phase_LUT, \
        phases_declared, \
        pulse_object \
    )
    
    # Do π-pulse
    pulse_object.output_pulse(T, pi_pulse_object)
    T += duration_of_pi_pulse
    phase_of_qubit = track_phase(T - T_begin, rf_carrier_frequency_of_qubit, phase_of_qubit)
    
    # Twist the qubit back with another virtual-Z gate.
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

def s_dagger_gate(
    execute_gate_at_this_time,
    
    phase_of_qubit,
    port,
    group_holding_phase_LUT,
    phases_declared,
    pulse_object,
    
    virtual_z_will_come_right_after_this_gate = False,
    )
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
