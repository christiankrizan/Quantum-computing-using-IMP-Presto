#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import h5py
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def assemble_rb_data(
    list_of_rb_data_files_to_integrate,
    num_random_quantum_circuits_to_generate_for_one_sequence_length,
    list_of_num_clifford_gates_per_x_axis_tick,
    plot_title = '',
    delete_files_once_finished = False,
    force_override_safety_check_for_deleting_files = False
    ):
    ''' Following a randomised-benchmarking experiment,
        produce a curve showing the RB sequence fidelity.
        The list_of_rb_data_files_to_integrate is expected on the
        following format:
            Assume the RB X-axis shows RB sequence lengths 1, 313, 626.
                Aka. list_of_num_clifford_gates_per_x_axis_tick = [1, 313, 626]
            Assume the num_random_quantum_circuits_to_generate_for_one_sequence_length = 3
            
            Then, running RB would generate the following experiments:
              1 Clifford gate  generated -> run measurement -> make file 1.
            313 Clifford gates generated -> run measurement -> make file 2.
            626 Clifford gates generated -> run measurement -> make file 3.
              1 Clifford gate  generated -> run measurement -> make file 4.
            313 Clifford gates generated -> run measurement -> make file 5.
            626 Clifford gates generated -> run measurement -> make file 6.
              1 Clifford gate  generated -> run measurement -> make file 7.
            313 Clifford gates generated -> run measurement -> make file 8.
            626 Clifford gates generated -> run measurement -> make file 9.
            
        So on the X-axis tick for 1 Clifford gate, we put all P(|0..0>) dot
        entries for 1 Clifford executed, which we thus find in
        files 1, 4 and 7. So the three dots on the X-axis tick for 1 Clifford
        gate, correspond to P(|0..0> for file 1, P(|0..0> for file 4, and
        P(|0..0> for file 7.
        
        We then average these three and put a fourth dot on the X-axis tick for
        1 Clifford gate. And so on for the X-axis ticks for 313 and 626 gates.        
        Throughout, the Y-axis shows P(|0..0>).
        
        We finally fit the averaged dots to a decaying exponential.
        The fitted curve gives us the RB sequence fidelity.
    '''
    
    # User-input argument sanitisation.
    assert isinstance(list_of_rb_data_files_to_integrate, list), "Error! The routine which assembles randomised benchmarking data was not fed a list to process. Instead, the datatype of the received input argument was: "+str(type(list_of_rb_data_files_to_integrate))+"."
    
    # Identify whether the list of RB data files contain single-
    # or multi-qubit randomised benchmarking. And, how many qubits were
    # involved while doing the RB experiment.
    with h5py.File(list_of_rb_data_files_to_integrate[0], 'r') as h5f:
        
        # Scrape all involved states from one file. In case the user provided
        # an input argument that contains incompatible files, an error will be
        # thrown at a later point.
        log_dict_list = json.loads( h5f.attrs["log_dict_list"] )
        involved_states = []
        for ii in range(len(log_dict_list)):
            # List all involved states as strings in a list.
            involved_states.append(((log_dict_list[ii]["name"]).replace('Probability for state |','')).replace('⟩',''))
        
        # Now find the longest string of zeroes in the list.
        # This entry gives the state for which the RB should be concerned.
        qubits_in_rb = 0
        for item in involved_states:
            jj = 0
            break_early = False
            while (jj < len(item)) and (not break_early): # Check all characters.
                if item[jj] == '0':
                    # Increase counter.
                    jj += 1
                else:
                    # The character was something else altogether.
                    # Break early.
                    break_early = True
            if (not break_early):
                # If break_early remained false, then the string consists
                # of only zeroes. In that case, save this state as the ground
                # state with most qubits qubits in |0⟩.
                if len(item) > qubits_in_rb:
                    qubits_in_rb = len(item)
    
    # At this point, we should have established the "longest" ground state
    # present in the randomised benchmarking experiment. For instance, |000>.
    
    # See def introduction and make sure you understand the format
    # of the received data in list_of_rb_data_files_to_integrate.
    # Here, we assemble a matrix with every RB circuit of X gates along
    # the X-axis.    
    rb_plot_matrix = [[0.0] * len(list_of_num_clifford_gates_per_x_axis_tick)] * num_random_quantum_circuits_to_generate_for_one_sequence_length
    rb_plot_matrix = np.array(rb_plot_matrix)
    index = 0.0
    for row in range(num_random_quantum_circuits_to_generate_for_one_sequence_length):
        for col in range(len(list_of_num_clifford_gates_per_x_axis_tick)):
            # Here, I'd love to just store the string given in
            # str(list_of_rb_data_files_to_integrate[index]) for every
            # provided file path. But then we must first un-shit Python's
            # trousers by finding the longest string present in
            # list_of_rb_data_files_to_integrate, and allocate this amount
            # of memory, due to Python getting Creutzfeldt-Jakob's disease
            # and trying to be smart with implicit memory mapping.
            rb_plot_matrix[row][col] = index
            index += 1.0
    
    # Thus at this point, rb_plot_matrix only contains the index for
    # list_of_rb_data_files_to_integrate which is supposed to go at that
    # position in this matrix.
    
    # We know know which state (is supposed to be) the longest |0..0> ground
    # state. And, we have a matrix to fill up with ground state probabilities.
    # Let's do that.
    for row in range(len(rb_plot_matrix)):
        for col in range(len(rb_plot_matrix[row])):
            
            # Get which index we are looking at.
            index = rb_plot_matrix[row][col]
            
            # Get ground state probability with P(|0..0>) of the file at
            # list_of_rb_data_files_to_integrate[index], and store this number
            # in rb_plot_matrix[row][col]
            with h5py.File(list_of_rb_data_files_to_integrate[int(index)], 'r') as h5f:
                log_dict_list = json.loads( h5f.attrs["log_dict_list"] )
                processed_data = h5f["processed_data"][()]
            
            # Ensure that P(|0..0>) exists in the data.
            found = False
            ff = 0
            found_at_index = 0
            while (ff < len(log_dict_list)) and (not found):
                #if ('Probability for state |'+("0"*qubits_in_rb)+'⟩') in log_dict_list[ff]["name"]: TODO this one.
                if ('Probability for state |'+("1"*qubits_in_rb)+'⟩') in log_dict_list[ff]["name"]:
                    found = True
                    found_at_index = ff
                ff += 1
            assert found, "Error! The provided file \""+str(list_of_rb_data_files_to_integrate[int(index)])+"\" does not contain probability data for the expected ground state |"+("0"*qubits_in_rb)+"⟩"
            
            # TODO DEBUG REMOVE
            print("############### WARNING! ###############\nTreating data found in |11> as the data to be found in |00>, since the data acquired for code development has just 0.0% stored in |00> for all files. There is a code row labelled \"TODO this one\" above, change this line from \"1\" to \"0\" once finished.")
            
            # Here, we know that processed_data[found_at_index] contains
            # the probability that we are after. Let's fill up the matrix.
            rb_plot_matrix[row][col] = processed_data[found_at_index]
            
            # Clear out data.
            del log_dict_list
            del processed_data
    
    # At this point, rb_plot_matrix is completed.
    # Columns in rb_plot_matrix correspond to the number of Clifford gates
    # executed for the randomised benchmarking experiment. Rows in turn
    # correspond to individual "sequence shots," each and every one of them
    # will become gray dots in the diagram.
    fig = plt.figure(figsize = (5, 5))
    
    # Drawing dots.
    # row = x-tick with some RB sequence length.
    # col = entries for some x-tick. All entries in one col is places on
    #       the same, corresponding x-tick.
    for row in range(len(rb_plot_matrix)):
        for col in range(len(rb_plot_matrix[row])):
            plt.plot(list_of_num_clifford_gates_per_x_axis_tick[col], rb_plot_matrix[row][col], 'o', color="#acacac")
    
    # Axis modifications
    plt.xticks(rotation = 45)
    plt.ylim(0.0, 1.0)
    
    # Labels
    plt.xlabel("Number of Clifford gates")
    plt.ylabel("P( |"+'0'*qubits_in_rb+"⟩ )")
    if plot_title != '':
        plt.title( plot_title )
    else:
        if qubits_in_rb == 1:
            plt.title( "Single-qubit randomised benchmarking" ) # TODO
        else:
            word = str(qubits_in_rb)
            # No check for >1 needed.
            if qubits_in_rb > 2:
                if qubits_in_rb > 3:
                    if qubits_in_rb > 4:
                        if qubits_in_rb > 5:
                            if qubits_in_rb > 6:
                                if qubits_in_rb > 7:
                                    if qubits_in_rb > 8:
                                        if qubits_in_rb < 10:
                                            word = "nine"
                                        # else:
                                        #     # Then default to str(qubits_in_rb)
                                        #     pass
                                    else:
                                        word = "eight"
                                else:
                                    word = "seven"
                            else:
                                word = "six"
                        else:
                            word = "five"
                    else:
                        word = "four"
                else:
                    word = "three"
            else:
                word = "two"
            
            plt.title( "Randomised benchmarking on "+word+" qubits" ) # TODO
    
    # Plot!
    plt.show()
    
    # TODO DEBUG REMOVE
    raise NotImplementedError("Routine not finished, and RB plot not fitted.")
    
    # Clean up?
    if delete_files_once_finished:
        if (not force_override_safety_check_for_deleting_files):
            for file_to_delete in list_of_rb_data_files_to_integrate:
                # Safety check in case the user is as dumb as I am.
                assert 'Data output folder' in file_to_delete, "Error! The randomised benchmarking data assembler was provided a file path that was not located at an expected location. Consider setting the input argument flag force_override_safety_check_for_deleting_files = True, in case this action was deliberate. The provided file path that triggered the error was: \n"+str(file_to_delete)
        
        # If the above check passed, or if the safety override is in force,
        # then delete things.
        for file_to_delete in list_of_rb_data_files_to_integrate:
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)