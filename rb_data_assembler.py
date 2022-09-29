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
from scipy.optimize import curve_fit

def assemble_rb_data(
    list_of_rb_data_files_to_integrate,
    num_random_quantum_circuits_to_generate_for_one_sequence_length,
    list_of_num_clifford_gates_per_x_axis_tick,
    average_fidelity_guess_for_fit,
    plot_show_duration = 0.0,
    plot_title = '',
    verbose = False,
    delete_files_once_finished = False,
    force_override_safety_check_for_deleting_files = False,
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
            
            # Ensure that P(|0..0>) exists in the provided data.
            found = False
            ff = 0
            found_at_index = 0
            while (ff < len(log_dict_list)) and (not found):
                if ('Probability for state |'+("0"*qubits_in_rb)+'⟩') in log_dict_list[ff]["name"]:
                    found = True
                    found_at_index = ff
                ff += 1
            assert found, "Error! The provided file \""+str(list_of_rb_data_files_to_integrate[int(index)])+"\" does not contain probability data for the expected ground state |"+("0"*qubits_in_rb)+"⟩"
            
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
    fig = plt.figure(figsize = (6, 6))
    
    # Drawing dots.
    # row = x-tick with some RB sequence length.
    # col = entries for some x-tick. All entries in one col is places on
    #       the same, corresponding x-tick.
    for row in range(len(rb_plot_matrix)):
        for col in range(len(rb_plot_matrix[row])):
            plt.plot(list_of_num_clifford_gates_per_x_axis_tick[col], rb_plot_matrix[row][col], 'o', color="#acacac")
    
    # Drawing the average dots
    mean_rb_values = np.mean(rb_plot_matrix, axis=0)
    for col in range(len(rb_plot_matrix[row])):
        plt.plot(list_of_num_clifford_gates_per_x_axis_tick[col], mean_rb_values[col], 'o', color="#e69313")
    
    # Fitting a decaying exponential to the data
    difference_between_first_and_last_averaged_rb_datapoints = mean_rb_values[0] - mean_rb_values[-1]
    offset = mean_rb_values[-1]
    popt, pcov = curve_fit( \
        decaying_exponential_function, \
        list_of_num_clifford_gates_per_x_axis_tick, \
        mean_rb_values, \
        p0 = ( \
            difference_between_first_and_last_averaged_rb_datapoints, \
            offset, \
            average_fidelity_guess_for_fit \
        ) \
    )
    
    # Compute the one-sigma standard deviation error.
    # See matplotlib documentation, search for "np.sqrt(np.diag(pcov))"
    p_sigma = np.sqrt(np.diag(pcov))
    
    # Get the average error rate
    r_ref = extract_average_error_rate( \
        num_qubits = qubits_in_rb,
        reference_average_fidelity_p_ref = popt[-1], \
        error_bar_of_p_ref = p_sigma[-1] \
    )
    
    # Print things?
    if verbose:
        print("The fitted reference average fidelity is: p_ref = "+str(popt[-1])+" ±"+str(p_sigma[-1]))
        print("The average error rate of the Clifford gates is: r_ref = "+str(r_ref[0])+" ±"+str(r_ref[1]))
        print("SPAM characterisation parameters were: A = "+str(popt[0])+" ±"+str(p_sigma[0])+", B = "+str(popt[1])+" ±"+str(p_sigma[1]))
    
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
            # Make a title
            plt.title( "Single-qubit randomised benchmarking" )
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
            
            # Make a title
            plt.title( "Randomised benchmarking on "+word+" qubits" )
    
    # Plot?
    if plot_show_duration > 0.0:
        # Plot the fitted curve.
        plt.plot(list_of_num_clifford_gates_per_x_axis_tick, decaying_exponential_function(list_of_num_clifford_gates_per_x_axis_tick, *popt))
        
        # Show the plot
        plt.show(block=False)
        plt.pause(plot_show_duration)
        plt.close()
    
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
    
    # Finish.
    average_fidelity_per_clifford_gate = [popt[-1], p_sigma[-1]]
    average_single_qubit_gate_error = [r_ref[0], r_ref[1]]
    spam_characteristic_A = [popt[0], p_sigma[0]]
    spam_characteristic_B = [popt[1], p_sigma[1]]
    return [average_fidelity_per_clifford_gate, average_single_qubit_gate_error, spam_characteristic_A, spam_characteristic_B]


def decaying_exponential_function(m, A, B, p):
    ''' Function used by the fitting routine.
        A,B quantify state preparation and measurement errors,
            where A is the difference in probability from the very first
            (shortest) RB experiment, versus the very last (longest)
            RB experiment. Ie. "the height difference of the exponential curve"
        p is the average fidelity per Clifford gate.
        
        See https://dx.doi.org/10.1103/PhysRevLett.116.150505 for formula.
        
    '''
    return A*p**m + B


def extract_average_error_rate(
    num_qubits,
    reference_average_fidelity_p_ref,
    error_bar_of_p_ref,
    ):
    ''' This routine follows the methodology outlined here:
        https://web.physics.ucsb.edu/~martinisgroup/theses/Kelly2015.pdf
    '''
    
    # Average error per Clifford of the reference measurement,
    # r_ref ± r_ref_plusminus_error:
    ## Marcus' crew's Gatemon 2016 paper approach, commented for now.
    ## 1.875 is widely known in litterature, otherwise go ahead and count the
    ## gates listed in Appendix B here, and divide by the 24 different options:
    ## https://qudev.phys.ethz.ch/static/content/science/Documents/master/Samuel_Haberthur_Masterthesis.pdf
    ## physical_pulses_on_average_per_clifford_gate = 1.875
    ## r_ref = (1 - reference_average_fidelity_p_ref) / (2 * physical_pulses_on_average_per_clifford_gate)
    
    # Approach from Julian Kelly's thesis, likely first defined in
    # Magesan-Gambetta-Emerson 2012, seems to fit
    # better with the rest of the world's approach.
    # See https://arxiv.org/pdf/1109.6887.pdf for MGE2012
    
    # System dimension d
    d = 2**(num_qubits)
    
    # Average error rate
    # r = 1 - F_ave = 1 - ( p + (1-p)/d ) = (1-p)*(d-1)/d
    r_ref = (1 - reference_average_fidelity_p_ref) * ((d-1)/d)
    
    # Error bars on r_ref
    if error_bar_of_p_ref != np.inf:
        r_ref_error_high = (1 - reference_average_fidelity_p_ref + error_bar_of_p_ref) / (2 * physical_pulses_on_average_per_clifford_gate)
        r_ref_error_low  = (1 - reference_average_fidelity_p_ref - error_bar_of_p_ref) / (2 * physical_pulses_on_average_per_clifford_gate)
        r_ref_plusminus_error = np.abs( np.max([r_ref_error_low, r_ref_error_high]) - np.min([r_ref_error_low, r_ref_error_high]) )/2
    else:
        r_ref_plusminus_error = np.inf
    return r_ref, r_ref_plusminus_error
  