###############################################################################
#  Qiskit © Copyright IBM 2017, 2021.
#  
#  This code is licensed under the Apache License, Version 2.0. You may
#  obtain a copy of this license in the LICENSE.txt file in the root directory
#  of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#  
#  Any modifications or derivative works of this code must retain this
#  copyright notice, and modified files need to carry a notice indicating
#  that they have been altered from the originals.
###############################################################################
''' The author of this file you have opened and are reading from
    (known as "this file" henceforth),
    Christian Križan (known as "I" and "my" henceforth),
    dictates my intentions on the 2022-08-26 as the following:
    
        This file containing this file header is licensed under
        Apache license 2.0.
    
        This file uses Qiskit code, and I thus release this file under
        the Apache 2.0 license. Should this license somehow "license infect"
        the other files I have provided relating to the Indecorum project,
        which is the project I offer at
        https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/,
        ... then I decree my intent that it is the Apache 2.0 files that
        are incorrectly licensed by me, and not any other file in the
        Indecorum project, since my intent is that the Indecorum project
        files are all licensed as 
      Creative Commons Attribution Non Commercial Share Alike 4.0 International
      CC-BY-NC-SA-4.0
      https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
      https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/
      
        ... bearing an appropriate LICENSE text that can readily be found at
        my Indecorum project's repository links above.
        
        To the fullest extent defendable, any and all files hosted in the
        Indecorum project, does not magically become Apache 2.0 licensed
        because of the accidental existence of this file you are reading from.
        
        If the implications of including an Apache 2.0-licensed file in the
        project suddenly risks infecting the rest of the files with
        an Apache 2.0 license, then the author hereby clearly dictates my
        intentions that the project is supposed to be licensed under
        CC-BY-NC-SA-4.0, and any files with licenses that suddenly try to
        change this fact, are incorrectly licensed files.
        
        My intent is that any sudden change of license to Apache 2.0 or
        similar, is accidental, and sudden licensing errors are to be
        rectified, rather than a user suddenly taking advantage of a
        stupid legal fluke thinking they get a carte blanche to do
        whatever the fuck they want to.
        
        Indecorum is © Copyright Christian Križan 2021,
        all rights reserved wherever possible, to the extent that the
        licensing of Indecorum under CC-BY-NC-SA-4.0 is unhindered.
'''

import numpy as np
from qiskit import transpile
from qiskit_experiments.library import StandardRB

# For simulation
#from qiskit.providers.aer import AerSimulator
#from qiskit.providers.fake_provider import FakeParis

def generate_rb_sequence(
    num_random_quantum_circuits_per_tick,
    list_of_num_clifford_gates_per_x_axis_tick,
    native_gate_set,
    qubit_indices,
    randomisation_seed = None,
    optimisation_level = 0,
    sample_cliffords_independently_for_all_lengths = True
    ):
    ''' Using Qiskit, generate a Randomised benchmarking experiment.
        
        num_random_quantum_circuits_per_tick:
            Each x-axis tick defines the length of a randomised
            benchmarking sequence. This argument defines how many
            samples will be taken at a certain x-tick.
            
            Example: 3
        
        list_of_num_clifford_gates_per_x_axis_tick:
            This argument is the x-axis. Every point defines
            the length of the RB sequences run at that tick.
        
            Example: [1,5,10]
        
        native_gate_set:
            List of strings, defining the gates present in this QPU
            backend's native gate set.
            
            Example: ['x','sz','rz','cz']
            
            Example of supported input strings:
                'cx', 'iswap', 'id', 'rz', 'h', 's', 'sx', 'sxdg', 'x', 'y', 'z', 'kraus', 'qerror_loc', 'quantum_channel', 'roerror', 'save_amplitudes', 'save_amplitudes_sq', 'save_clifford',
                'save_density_matrix', 'save_expval', 'save_expval_var', 'save_matrix_product_state', 'save_probabilities', 'save_probabilities_dict', 'save_stabilizer', 'save_state', 'save_statevector',
                'save_statevector_dict', 'save_superop', 'save_unitary', 'set_density_matrix', 'set_matrix_product_state', 'set_stabilizer', 'set_statevector', 'set_superop', 'set_unitary', 'snapshot', 'superop',
        
        qubit_indices:
            List defining the indexes of the qubits used in the run.
            
            Example: [0]    or    [0,1]
        
        randomisation_seed:
            Set specific seed for the RB sequence randomiser.
            
            Example: 313
        
        optimisation_level:
            Enable quantum circuit optimisation in Qiskit when generating gates.
            
            Legal input values: 0, 1, 2, 3
            ... each defining a heightened level of optimisation.
        
        sample_cliffords_independently_for_all_lengths:
            Sets whether the Cliffords are independently sampled
            for all x-axis ticks.
            
            2023-04-20:
            https://support.minitab.com/es-mx/minitab/21/help-and-how-to/statistics/basic-statistics/supporting-topics/tests-of-means/what-are-independent-samples/
            "Independent samples are samples that are selected randomly so that
            its observations do not depend on the values other observations."
    '''
    
    # How many qubits are there?
    try:
        num_qubits = len(qubit_indices)
    except:
        raise ValueError("Error! Could not determine the number of qubits in the system. The provided argument was: "+str(qubit_indices))
    assert len(qubit_indices) > 0, "Error! Bad argument provided to the RB sequence generator for the qubit indices list. The provided argument was: "+str(qubit_indices)
    
    # User argument sanitisation.
    assert len(native_gate_set) > 0, "Error! Bad argument provided to the RB sequence generator for the native gate set. The provided argument was: "+str(native_gate_set)
    if optimisation_level > 3:
        print("Warning! The optimisation level for the Qiskit randomised banchmarking routines was changed from "+str(optimisation_level)+" to 3, which is the highest possible optimisation flag.")
        optimisation_level = 3
    elif optimisation_level < 0:
        print("Warning! The optimisation level for the Qiskit randomised banchmarking routines was changed from "+str(optimisation_level)+" to 0, which is the lowest possible optimisation flag.")
        optimisation_level = 0
    
    # Parse list_of_num_clifford_gates_per_x_axis_tick in case of bad user input.
    list_of_num_clifford_gates_per_x_axis_tick = \
        np.unique( np.sort(list_of_num_clifford_gates_per_x_axis_tick) )
    assert (len(list_of_num_clifford_gates_per_x_axis_tick) > 0), "Error! The Randomised Benchmarking routine was tasked to generate an RB sequence that was 0 long. The full list of tasked RB sequences was:\n"+str(list_of_num_clifford_gates_per_x_axis_tick)
    
    # Generate RB experiment.
    rb_experiment = StandardRB(
        qubits = qubit_indices,
        lengths = list_of_num_clifford_gates_per_x_axis_tick,
        num_samples = num_random_quantum_circuits_per_tick,
        seed = randomisation_seed,
        full_sampling = sample_cliffords_independently_for_all_lengths
    )
    
    # Transpile the circuit to our native gate set.
    transpiled_schema = transpile(
        circuits = rb_experiment.circuits(),
        basis_gates = native_gate_set,
        optimization_level = optimisation_level
    )
    
    # Return transpiled result.
    return transpiled_schema

def indecorum_parse(
    qk_sequence,
    gate_durations,
    qubit_indices
    ):
    ''' Routine that parses a Qiskit schema into a sequence compatible with
        Indecorum.
        
        Syntax of gate durations is:
        gate_durations = {
            "gate1": 25e-9,
            "gate2": 350e-9
        }
            ... where gate1 (gate2) is the name of the gate to map,
                and 25e-9 (350e-9) is the time in seconds that gate
                requires to execute.
    '''
    
    # TODO! The barrier handling in this parser needs to get better I think.
    # TODO! The connectivity map needs to be remade somewhat.
    
    # Create an empty list of barrier moments. Every quantum circuit moment
    # between two barriers is contained in a new entry of this list.
    barrier_moments = []
    curr_moment = []
    curr_moment_index = 0
    
    # Figure out what the current moment's time T is for every operation.
    # If the qubits involved for a given operation are more than one,
    # all qubit channels' "time counters" get set to the largest counter's
    # value. The length of the time_counters list matches
    # the total number of qubits involved.
    time_counters = [0] * len(qubit_indices)
    
    # Go through the quantum circuit.
    for i in range(len(qk_sequence)):
        
        # Get opcode.
        opcode_name = (qk_sequence[i])[0].name
        
        # If the operation was a barrier, close the curr_moment list
        # and move on to the next barrier moment.
        if not opcode_name == 'barrier':
            # Continue!
            
            # Get parameters of the opcode.
            opcode_param = (qk_sequence[i])[0].params
            if len(opcode_param) != 0:
                # TODO Ugly fetch of parameter content.
                opcode_param = [float(((str(opcode_param[0])).replace('ParameterExpression',''))[:-1])]
                
            # Figure out which qubits are involved.
            qubits = []
            for k in ((qk_sequence[i])[1]):
                # TODO There is an ugly string conversion here.
                qk_also_ugly = str(k)
                qk_also_ugly = (qk_also_ugly.replace('Qubit(QuantumRegister(',''))[:-1]
                qk_also_ugly = (qk_also_ugly.replace(' \'q\')','')).replace(' ','')
                qubits.append( qubit_indices[ int(qk_also_ugly.split(',,')[1]) ])
                # Note that the qk qubit number is used to look up
                # what provided qubit number it corresponds to.
            
            # Figure out at what time does the gate execute, and update the
            # time counters involved. If this is a multi-qubit gate,
            # push all qubits' counters to the highest value to match when
            # the multi-qubit gate operation occurs.
            operation_will_require_this_time = gate_durations[opcode_name]
            if len(qubits) > 0: # Optimisation! The speed(1x if-case) = faster!
                # Reset the largest_counter.
                largest_counter = 0
                
                # Find the largest time_counter involved with this
                # multi-qubit operation.
                for item in qubits:
                    # Get which counter belongs to that qubit.
                    item_index = qubit_indices.index(item)
                    
                    # Then, check which counter value was the largest one.
                    if time_counters[item_index] > largest_counter:
                        largest_counter = time_counters[item_index]
                
                # Assign the found value as the beginning value to be stored.
                begin_at_time = largest_counter
                
                # Now update all involved time_counters to
                # match that largest time_counter value.
                for item2 in qubits:
                    # Get which counter belongs to that qubit, again.
                    item_index = qubit_indices.index(item2)
                
                    # Now update the time counters to match the largest value,
                    # in case there is an n-qubit gate happening.
                    time_counters[item_index] = largest_counter + operation_will_require_this_time
                
            else:
                # There is only one qubit, set the item index to match it.
                item_index = qubit_indices.index(qubits[0])
                
                # Store the now-found "begin" time for the parsed list.
                begin_at_time = time_counters[item_index] 
                
                # Increment time_counters for this qubit.
                time_counters[item_index] += operation_will_require_this_time 
            
            # Append parsed operation at the current curr_moment_index.
            curr_moment.append( [opcode_name, opcode_param, qubits, begin_at_time] )
            
            # Was this a 'measure' operation? Could it be combined with
            # the previous 'measure' operation? (= multiplexed readout)
            if opcode_name == 'measure':
                
                # If we don't check whether the curr_moment_index != 0,
                # we might look at curr_moment[-1] -> would give a false
                # positive multiplexable operation, if a readout was done
                # after a barrier (because, the barrier resets the index).
                # Because, the current and the [-1] ( aka. LASt and ONLY )
                # operations in curr_moment would both be the very same
                # single 'measure' operation.
                if curr_moment_index != 0:
                    
                    # Was the previous operation also a 'measure' operation?
                    # Then these two could be done multiplexed.
                    if (curr_moment[curr_moment_index-1][0] == 'measure'):
                        # Let's combine readout operations!
                        # The qubits involved are covered at index [2].
                        curr_moment[curr_moment_index-1][2] = \
                            curr_moment[curr_moment_index-1][2] +\
                            curr_moment[curr_moment_index][2]
                        
                        # Finally, delete the last (now combined) entry of
                        # the curr_moment list. Remember to update the index.
                        curr_moment = curr_moment[:-1]
                        curr_moment_index -= 1
            
            # Increment the curr_moment_index for the next iteration.
            curr_moment_index += 1
            
        else:
            # Barrier detected. Close and move on.
            barrier_moments.append(curr_moment)
            curr_moment = []
            curr_moment_index = 0
    
    # Append the final curr_moment, in case there was no barrier generated
    # at the very end. In case there are barriers at the very end of
    # the quantum circuit, it will be caught and removed soon.
    barrier_moments.append(curr_moment)
    
    # Append final (total) duration to the end.
    barrier_moments.append( [['total_duration', time_counters, qubit_indices, np.max(time_counters)]] )
    
    # Remove empty barrier moments.
    barrier_moments = list(filter(None, barrier_moments))
    
    # Returned parsed sequence.
    return barrier_moments