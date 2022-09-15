#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import h5py
import json

def assemble_rb_data(
    list_of_rb_data_files_to_integrate,
    delete_files_once_finished = False
    ):
    ''' Following a randomised-benchmarking experiment,
        produce a curve showing the RB sequence fidelity.
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
    
    # TODO DEBUG REMOVE
    raise NotImplementedError("Routine not finished. But "+str(qubits_in_rb)+" qubits were found to be involved.")
    
    # Clean up?
    if delete_files_once_finished:
        for file_to_delete in delete_files_once_finished:
            # TODO delete the file.
            pass