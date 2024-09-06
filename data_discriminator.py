#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import os
import sys
import time
import h5py
import json
import shutil
import random
import numpy as np
from numpy import hanning as von_hann
from math import isnan
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

def initiate_discriminator_settings_file(
    initial_resonator_qubit_pair = 0
    ):
    '''Assuming one resonator per qubit, make a discriminator settings file.'''
    
    # Get file path to the discriminator settings' JSON.
    full_path_to_discriminator_settings_json = \
        (get_file_path_of_discrimination_json())[0]
    
    # Make the JSON file if it does not exist.
    if not os.path.exists(full_path_to_discriminator_settings_json):
        open(full_path_to_discriminator_settings_json, 'a').close()
    
    # Build the JSON (dict template of dicts) that will be saved.
    json_dict = dict()
    json_dict.update(
        {'resonator_'+str(initial_resonator_qubit_pair): 
            {'qubit_states':
                dict( qubit_state_0_centre_real = 0, qubit_state_0_centre_imag = 0 )
            }
        }
    )
    
    # Build JSON file.
    with open(full_path_to_discriminator_settings_json, 'w') as json_file:
        json.dump(json_dict, json_file, indent = 4)

def calculate_population_centres(
    processed_data
    ):
    ''' From supplied data, calculate the centre for the populations held
        within.
        
        NOTE! For now, only a single resonator is analysed
        from the processed data. This is likely a TODO.
    '''
    centre = []
    for row in range(len( (processed_data[0])[:] )):
        # Run a complex mean on all complex values.
        curr_data  = (processed_data[0])[row,:]
        centre.append( np.mean(curr_data) )
    
    # Return list of centre coordinates.
    return centre
    
def calculate_readout_fidelity(
    data_to_process,
    state_centre_coordinates_list,
    prepared_qubit_states,
    ):
    ''' From the given data set and the state centre coordinates, calculate
        a readout fidelity.
        
        Expected format on state_centre_coordinates_list:
            [ complex coordinate of |0>,
              complex coordinate of |1>,
              complex coordinate of |2>
           ]                               etc. for n states.
        
        Readout fidelity F_RO = ( SUM(<m|m>) / N )  for N number of states |m>
        
        Example:    With N=3 states |0>, |1>, |2> you'd get:
                    F_RO = (<0|0> + <1|1> + <2|2>) / 3
    '''
    
    ## Currently, only one readout resonator is accounted for. Likely a TODO.
    
    # We will mainly return the readout fidelity.
    readout_fidelity = 0.0
    
    # However, we should also return the probability of measuring
    # each state, given the other prepared states.
    confusion_matrix = \
        [[]] * (np.max(prepared_qubit_states) + 1) # +1 accounts for state |0>
    
    # (data_to_process[0])[0] contains every shot prepared in |0>
    # (data_to_process[0])[1] contains every shot prepared in |1>
    # (data_to_process[0])[2] contains every shot prepared in |2>
    # ... or, some other states "temporarily mapped" to index 0 1 2 .. n
    #     in the user-supplied n-long list "prepared_qubit_states".
    
    # discretised_results will be the same matrix, but every item in the whole
    # matrix has been state discriminated, according to user-provided
    # coordinates.
    discretised_results = [[]] * len(data_to_process[0])
    for ii_all_prepared_states in range(len( data_to_process[0] )):
        
        # Current vector to be added in discretised_results.
        current_vector = np.zeros(len( (data_to_process[0])[0] ))
        
        # For all shots:
        for jj_all_shots_fired in range(len( (data_to_process[0])[0] )):
            
            # Get current item for some prepared readout state.
            current_item = ((data_to_process[0])[ii_all_prepared_states])[jj_all_shots_fired]
            
            # Discretise and add.
            assigned = np.argmin(np.abs(state_centre_coordinates_list - current_item))
            current_vector[jj_all_shots_fired] = prepared_qubit_states[assigned]
        
        # Add vector to discretised results
        discretised_results[ii_all_prepared_states] = current_vector.astype('int32')
        
    # discretised_results is now a matrix where every row corresponds
    # to some prepared state. The readout fidelity formula is given in the
    # function description above.
    sum_of_all_individual_readout_fidelities = 0.0
    number_of_fidelities_calculated = 0
    
    for kk_all_prepared_states in range(len( discretised_results )):
        
        # What state was prepared?
        the_prepared_state = prepared_qubit_states[kk_all_prepared_states]
        
        ## Here, I've already taken into account that np.bincount
        ## will be a vector that is *at most* as long as the highest
        ## state that was binned into. If there was no |2> for instance,
        ## then np.bincount will only return a len = 2 long vector.
        ## However, all values above |2> in the bincount would have
        ## been 0 anyways, since there were no counts for those N states.
        ## Thus, just zero-padding the np.bincount vector, is a good solution.
        
        # Grab the count of all outcomes. Example: [423, 1, 5] means that
        # for the prepared state "kk_all_prepared_states",
        # 423 hits were |0>, 1 hit was |1>, and 5 hits was |2>.
        binslice = np.bincount( discretised_results[kk_all_prepared_states] )
        binslice = np.pad(binslice, (0, np.max(prepared_qubit_states)+1 - len(binslice)))
        strikes  = binslice[the_prepared_state]
        
        # Store the full probability of mesasuring in P( n | m ).
        confusion_matrix[kk_all_prepared_states] = \
            binslice / np.sum( binslice )
    
    # Return calculated readout fidelity. Also, the full P( n | m ) matrix.
    #   Formula: P( n | n ) / no_states,
    #   where n ∈ all states considered for readout.
    readout_fidelity = np.trace(confusion_matrix) / len(np.diagonal(confusion_matrix))
    return readout_fidelity, np.array(confusion_matrix)

def update_discriminator_settings_with_value(
    path_to_data,
    resonator_transmon_pair = None,
    ):
    ''' Update the discrimination values based on supplied data.
        And, save the update to disk if you so please.
    '''
    
    # Load the file at path_to_data.
    with h5py.File(os.path.abspath(path_to_data), 'r') as h5f:
        processed_data = h5f["processed_data"][()]
        prepared_qubit_states = h5f["prepared_qubit_states"][()]
        
        # Automatically establish what resonator-transmon pair was used?
        if resonator_transmon_pair != None:
            # No: use the user-provided readout state.
            resonator_transmon_pair = int(resonator_transmon_pair)
        else:
            # Yes: then get current resonator-transmon-pair from the read file.
            try:
                resonator_transmon_pair = h5f.attrs["resonator_transmon_pair_id_number"]
            except KeyError:
                raise KeyError( \
                    "Fatal error! The function for updating the "            +\
                    "discriminator settings was called without specifying "  +\
                    "what resonator-transmon pair the data "                 +\
                    "in the supplied filepath belongs to. "                  +\
                    "But, the data file at the supplied filepath "           +\
                    "contains no explicit information regarding what "       +\
                    "resonator-transmon pair was used for taking the data. " +\
                    "Check the resonator frequency of this file, and "       +\
                    "then manually call this function again with the "       +\
                    "resonator-transmon pair argument correctly set.")
            
            # Assert that the datatype of the resonator_transmon_pair
            # is valid. This fact could depend on Labber bugs for instance.
            if (not isinstance(resonator_transmon_pair, int)):
                # The returned value is not an integer. Make it into one.
                resonator_transmon_pair = int(resonator_transmon_pair[0])
    
    # Calculate population centres from supplied data.
    centre = calculate_population_centres(processed_data)
    
    # Get JSON.
    loaded_json_data = load_discriminator_settings(resonator_transmon_pair)
    
    # Save this centre for this state X for resonator Y in the JSON.
    # Update missing keys if any.
    for jj in range(len(prepared_qubit_states)):
        # The key might be non-existant. Then we may do whatever we want.
        try:
            curr_dict = (loaded_json_data['resonator_'+str(resonator_transmon_pair)])['qubit_states']
            # The key existed. Then update. Remember that the key will update.
            # There is no need to re-save data into the loaded_json_data array.
            curr_dict.update( \
                { \
                    'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_real': centre[jj].real, \
                    'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_imag': centre[jj].imag  \
                }
            )
        except KeyError:
            ##top_dict = loaded_json_data['resonator_'+str(resonator_transmon_pair)]
            ##top_dict.update( \
            ##    {'readout_state_'+str(readout_state) : \
            top_dict = loaded_json_data
            top_dict.update(
                {'resonator_'+str(resonator_transmon_pair) : \
                    {'qubit_states' : \
                        {
                            'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_real' : centre[jj].real, \
                            'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_imag' : centre[jj].imag  \
                        }
                    }
                }
            )
    
    # Save settings.
    save_discriminator_settings(loaded_json_data)
    
def discriminate(
    states_that_will_be_checked,
    data_or_filepath_to_data,
    i_provided_a_filepath = False,
    ordered_resonator_ids_in_readout_data = []
    ):
    ''' Takes a data set and discriminates the data held within.
    '''
    
    # Input sanitisation.
    assert (len(ordered_resonator_ids_in_readout_data) > 0), "Error: state discrimination failed, no readout resonator ID provided."
    
    # Load discrimination settings. Each new row in
    # states_present_for_resonators corresponds to a new resonator ID.
    states_present_for_resonators = [[]] * len(ordered_resonator_ids_in_readout_data)
    curr_ii = 0
    for ff in ordered_resonator_ids_in_readout_data:
        
        # Cut out what states are available to discriminate to at resonator ff:
        looked_at_res_dict = (load_discriminator_settings(ff))['resonator_'+str(ff)]
        curr_looked_at_res_dict = (looked_at_res_dict)['qubit_states']
        list_of_qubit_states = list(curr_looked_at_res_dict.keys())
        new_list = []
        for tt in range(0,len(list_of_qubit_states),2):
            new_list.append(int((list_of_qubit_states[tt].replace('qubit_state_','')).replace('_centre_real','')))
        list_of_qubit_states = new_list
        
        # Now get the coordinates (real, imaginary) for all of the present states.
        coordinate_list = []
        for cc in list_of_qubit_states:
            real_coord = curr_looked_at_res_dict['qubit_state_'+str(cc)+'_centre_real']
            imag_coord = curr_looked_at_res_dict['qubit_state_'+str(cc)+'_centre_imag']
            coordinate_list.append(real_coord + 1j*imag_coord)
        
        # Note:
        # coordinate_list[ff] = centre coordinates for list_of_qubit_states[ff]
        states_present_for_resonators[curr_ii] = [list_of_qubit_states, coordinate_list]
        curr_ii += 1
        
    # Assert that the user is not trying to look for states that are
    # unavailable for a particular resonator.
    for curr_state_string in states_that_will_be_checked:
        for pp in range(len(curr_state_string)):
            # Check that this sought-for state is available in said resonator.
            for cc in range(len(states_present_for_resonators)):
                cur_res = states_present_for_resonators[cc]
                assert (int(curr_state_string[pp]) in cur_res[0]), \
                    "Error: the user is trying to state discriminate to "    +\
                    "state "+str(curr_state_string[pp])+" in resonator with" +\
                    " ID "+str(ordered_resonator_ids_in_readout_data[cc])+"."+\
                    " But, there is no information about this state for this"+\
                    " resonator stored. Please configure state "             +\
                    "discrimination for this state on this resonator "       +\
                    "to continue."
    
    # Get data.
    if i_provided_a_filepath:
        # The user provided a filepath to data.
        assert isinstance(data_or_filepath_to_data, str), "Error: the discriminator was provided a non-string datatype. Expected string (filepath to data)."
        with h5py.File(os.path.abspath(data_or_filepath_to_data), 'r') as h5f:
            extracted_data = h5f["processed_data"][()]
    else:
        # Then the user provided the data raw.
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the discriminator was provided a string type. Expected raw data. The provided variable was: "+str(data_or_filepath_to_data)
        extracted_data = data_or_filepath_to_data
    
    # Prepare the highest_state_in_system variable.
    # As we will soon look at all states available in the system,
    # we might as well keep track of the number of states in the system
    # in the for loop below. num_states_in_system = highest_state_in_system +1
    # Because |0> counts too.
    highest_state_in_system = 0
    
    # We now have the discretisation information needed, and the
    # processed data that will be discretised using the known state centres.
    for current_res_idx in range(len(extracted_data)):
        res_content = extracted_data[current_res_idx]
        states = (states_present_for_resonators[current_res_idx])[0]
        state_centres = np.array((states_present_for_resonators[current_res_idx])[1])
        
        # For all complex-valued indices from the readout on this resonator,
        # figure out what state centre a_shot was closest to.
        for bb in range(len(res_content)):
            assigned = np.argmin(np.abs(state_centres-res_content[bb]))
            res_content[bb] = states[assigned]
        
        # Save and move on to the next resonator.
        extracted_data[current_res_idx] = res_content.astype(int)
        
        # For future reference (the return of this funtion), store the
        # largest represented state in the system.
        if np.max(states) > highest_state_in_system:
            highest_state_in_system = np.max(states)
    
    # The number of states in the system is the highest represented
    # state in the system +1, because |0> counts.
    num_states_in_system = highest_state_in_system + 1
    
    return extracted_data, num_states_in_system

def calculate_area_mean_perimeter_fidelity(
    path_to_data,
    update_discriminator_settings_json = False,
    resonator_transmon_pair = None,
    ):
    ''' Given a file containing readout state data, calculate
        the area and perimeter spanned by the readout states,
        as well as readout fidelities.
    '''
    
    # For all qubit states in the readout, gather spanned area, perimeter,
    # mean distance between all states, and readout fidelity where possible.
    area_spanned_by_qubit_states = []
    mean_distance_between_all_states = []
    hamiltonian_path_perimeter = []
    readout_fidelity = []
    
    # Load the file at path_to_data.
    with h5py.File(os.path.abspath(path_to_data), 'r') as h5f:
        processed_data = h5f["processed_data"][()]
        prepared_qubit_states = h5f["prepared_qubit_states"][()]
        
        # Automatically establish what resonator-transmon pair was used?
        if resonator_transmon_pair != None:
            # No: use the user-provided readout state.
            resonator_transmon_pair = int(resonator_transmon_pair)
        else:
            # Yes: then get current resonator-transmon-pair from the read file.
            try:
                resonator_transmon_pair = h5f.attrs["resonator_transmon_pair_id_number"]
            except KeyError:
                raise KeyError( \
                    "Fatal error! The function for updating the "            +\
                    "discriminator settings was called without specifying "  +\
                    "what resonator-transmon pair the data "                 +\
                    "in the supplied filepath belongs to. "                  +\
                    "But, the data file at the supplied filepath "           +\
                    "contains no explicit information regarding what "       +\
                    "resonator-transmon pair was used for taking the data. " +\
                    "Check the resonator frequency of this file, and "       +\
                    "then manually call this function again with the "       +\
                    "resonator-transmon pair argument correctly set.")
            
            # Assert that the datatype of the resonator_transmon_pair
            # is valid. This fact could depend on Labber bugs for instance.
            if (not isinstance(resonator_transmon_pair, int)):
                # The returned value is not an integer. Make it into one.
                resonator_transmon_pair = int(resonator_transmon_pair[0])
    
    # Calculate population centres from supplied data.
    centre = calculate_population_centres(processed_data)
    no_qubit_states = len(centre)
    
    # Calculate the readout fidelity from supplied data.
    readout_fidelity, confusion_matrix = \
        calculate_readout_fidelity(
            data_to_process = processed_data,
            state_centre_coordinates_list = centre,
            prepared_qubit_states = prepared_qubit_states,
        )
    
    # Check whether the program can find and area and a mean distance
    # between all states given the acquired data.
    curr_area_spanned_by_qubit_states = None
    curr_mean_distance_between_all_states = None
    curr_hamiltonian_path_perimeter = None
    ##curr_readout_fidelity = None # Taken care of earlier up the code.
    if no_qubit_states <= 2:
        if no_qubit_states <= 1:
            # If there are <= 1 states to look at, then one can not span
            # an area for the readout done in |n>. Nor a perimeter.
            print( \
                "Warning: the file at "+str(path_to_data)+" does not contain"+\
                " a sufficient number of qubit states for calculating "   +\
                "area, distance between states, nor a readout fidelity."+\
                " (№ states = "+str(no_qubit_states)+"). Returning several "+\
                "\"None\" values.")
        else:
            # If there are 2 states to look at, then at least a vector
            # can be acquired giving the length between the two states.
            # And, a readout fidelity.
            print( \
                "Warning: the file at "+str(path_to_data)+" only contains "  +\
                "a sufficient number of states for calculating a readout "   +\
                "fidelity, and a mean distance between two states (№ states "+\
                "= "+str(no_qubit_states)+"). The returned area will be "    +\
                "\"None\".")
            curr_mean_distance_between_all_states = -1
            ##curr_readout_fidelity = -1 # Taken care of earlier up the code.
    else:
        # We have a sufficient number of states acquired. Remove the None
        # status to signal down below that the area and mean state distance
        # will be calculated.
        curr_area_spanned_by_qubit_states = -1
        curr_mean_distance_between_all_states = -1
        ##curr_readout_fidelity = -1 # Taken care of earlier up the code.
    
    # TODO: Currently, this script does not calculate the Hamiltonian path
    # for all states, other than for the simplest possible scenario
    # (n points along the hamiltonian path = 3)
    if no_qubit_states == 3:
        curr_hamiltonian_path_perimeter = -1
    elif no_qubit_states > 0:
        print("WARNING: Hamiltonian path algorithm not implemented. Will only return a perimeter if № states = 3.") # TODO
    
    # Get all vertices of the polygon spanned in
    # the real-imaginary plane.
    vertices_x_arr = []
    vertices_y_arr = []
    vertices = np.array([])
    for pp in range( no_qubit_states ):
        ##vertices_x_arr.append(curr_investigated_dict['qubit_state_'+str(pp)+'_centre_real'])
        ##vertices_y_arr.append(curr_investigated_dict['qubit_state_'+str(pp)+'_centre_imag'])
        ##vertices = np.append( vertices, (vertices_x_arr[pp] + vertices_y_arr[pp]*1j) )
        vertices_x_arr.append((centre[pp]).real)
        vertices_y_arr.append((centre[pp]).imag)
        vertices = np.append( vertices, centre[pp] )
    
    # Should a spanned area be calculated?
    if curr_area_spanned_by_qubit_states != None:
        
        # Implement jacobian trapezoidal formula to get the area.
        def jacobian_trapezoidal_area(x_list,y_list):
            a1,a2 = 0,0
            x_list.append(x_list[0])
            y_list.append(y_list[0])
            for j in range(len(x_list)-1):
                a1 += x_list[j]*y_list[j+1]
                a2 += y_list[j]*x_list[j+1]
            l = abs(a1-a2)/2
            return l
        curr_area_spanned_by_qubit_states = \
            jacobian_trapezoidal_area(vertices_x_arr, vertices_y_arr)
    
    # Should a mean distance between all readout states be calculated?
    if mean_distance_between_all_states != None:
        # Set up running mean. All state distances are summed together
        # exactly once, and the total sum is divided with the
        # number of considered states.
        uu = len(vertices)
        if uu > 1:
            mean_sum     = 0
            mean_divisor = 0
            while (uu-1) > 0:
                for rr in range(uu-1):
                    mean_sum     += np.abs(vertices[uu-1]-vertices[rr])
                    mean_divisor += 1
                uu -= 1 # Subtract reciprocal counter
            # Perform final mean divison.
            curr_mean_distance_between_all_states = mean_sum / mean_divisor
        else:
            # There are less than two states considered,
            # we may thus not calculate a distance.
            curr_mean_distance_between_all_states = None
    
    # Should a Hamiltonian path be drawn between all vertices
    # to calculate the shortest possible perimeter that spans all states?
    if curr_hamiltonian_path_perimeter != None:
        # TODO! Not the Hamiltonian path algorithm
        curr_hamiltonian_path_perimeter = \
            np.abs(vertices[2] - vertices[1]) +\
            np.abs(vertices[1] - vertices[0]) +\
            np.abs(vertices[0] - vertices[2])
    
    # Update the return values.
    area_spanned_by_qubit_states = curr_area_spanned_by_qubit_states
    mean_distance_between_all_states = curr_mean_distance_between_all_states
    hamiltonian_path_perimeter = curr_hamiltonian_path_perimeter
    
    # Update the local JSON that will be saved to file?
    if update_discriminator_settings_json:
    
        # Load JSON.
        loaded_json_data = load_discriminator_settings(resonator_transmon_pair)
        resonator_dict = loaded_json_data['resonator_'+str(resonator_transmon_pair)]
        
        try:
            analysis_dict = resonator_dict['analysis']
        except KeyError:
            resonator_dict.update( \
                {'analysis': {} }  \
            )
            analysis_dict = resonator_dict['analysis']
        if curr_area_spanned_by_qubit_states != None:
            analysis_dict.update( \
                {'area_spanned_by_qubit_states': curr_area_spanned_by_qubit_states} \
            )
        else:
            try:
                del analysis_dict['area_spanned_by_qubit_states']
            except KeyError:
                # The key did not exist, this is OK.
                pass
        if curr_mean_distance_between_all_states != None:
            analysis_dict.update( \
                {'mean_distance_between_all_states': curr_mean_distance_between_all_states} \
            )
        else:
            try:
                del analysis_dict['mean_distance_between_all_states']
            except KeyError:
                # The key did not exist, this is OK.
                pass
        if curr_hamiltonian_path_perimeter != None:
            analysis_dict.update( \
                {'hamiltonian_path_perimeter': curr_hamiltonian_path_perimeter} \
            )
        else:
            try:
                del analysis_dict['hamiltonian_path_perimeter']
            except KeyError:
                # The key did not exist, this is OK.
                pass
        
        # Save calculated settings
        save_discriminator_settings(loaded_json_data)
    
    return area_spanned_by_qubit_states, mean_distance_between_all_states, \
        hamiltonian_path_perimeter, readout_fidelity, \
        confusion_matrix

def construct_state_from_quadrature_voltage(
    ):
    ''' Readout voltage quadratures can be used to reconstruct qubit states.
        This has been shown here:
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.80.043840
        
        You gather a **heavily averaged set of data**, note again: averaged.
        Even before FFT.
        
        From this data set, we can create a mapping that maps quadrature
        voltages into qubit states. A linearising map.
    '''
    raise NotImplementedError("Halted! This function is not finished.")
    
def post_process_data_given_confusion_matrix(
    filepaths_to_plot,
    confusion_matrix,
    plot_output_export_path = '',
    figure_size_tuple = (2.654, 1.887),
    title = '',
    plot_output = False,
    hdf5_export_path = '',
    maximum_number_of_qubits_contained_in_file = 5, # Number selected arbitrarily!
    select_file_entry_to_extract = [],
    attempt_to_fix_string_input_argument = True,
    sum_states_of_vectors_in_order_to_select_single_qubit_probabilities = [],
    captivorum_colore_utere = False,
    ):
    ''' Given a known confusion matrix, perform least-square minimisation
        between some assumed input and the measured data given the
        density matrix.
        
        The motivation of accounting for the confusion matrix like so,
        is to avoid matrix inversion, which makes off-diagonal elements
        very large (since these are typically just a few percent in size
        in the confusion matrix).
        
        Method: fit for this thing
        |  [y0,y1] = [conf00, conf01, conf10, conf11] · [x0,x1]  |^2 = 0
        
        ... and return [x0, x1]. This vector is the vector in which
        the difference between the measured data and the guessed original
        is the smallest.
        
        confusion_matrix should be given as a numpy array. If measurement
        error mitigation is performed on two states, and two qubits, then
        the confusion matrix must by definition be a (states^qubits) large
        matrix. 3 states, 2 qubits, would require a 3^2 = 9x9 matrix.
        Its axes would be Measured 2q state |00>, |01> ... |21>, |22> on the
        x-axis, and Prepared 2q state |00>, |01> ... |21>, |22> on the y-axis.
        
        select_file_entry_to_extract: array-like object showing indices
        of the traces that are to be extracted (selected).
        
        sum_states_of_vectors_in_order_to_select_single_qubit_probabilities
        allows the user to check the single-qubit probability, by supplying
        the two-qubit states that when taken together adds up to the
        probability of the single qubit.
        Example: if the user wants to see the probability of qubit 1 being
        in |1>, while the state probability of the other qubit(s) is
        irrelevant, then the user can provide here ['10','11','12']
        for instance. Whether qubit 2 was in |0>, |1> or |2> is irrelevant,
        the user currently only wants to know the state probability of Q1 = |1>
        
        captivorum_colore_utere:
        If True, then plot in cyan/purple for ±π radian x-axis plots,
        like CZ conditional-Ramsey plots, and red/green for ±2π rad x-axis
        plots, like iSWAP and SWAP cross-Ramsey plots.
        Used for the 2024 iSWAP, SWAP, CZ paper by Christian Križan et al.
    '''
    
    ## Input checking!
    
    # Check the density matrix.
    try:
        assert (confusion_matrix.shape[0] == confusion_matrix.shape[1]), "Error! The confusion matrix must be symmetric, by definition."
    except AttributeError:
        raise AttributeError("The user-provided confusion matrix did not match expectations. Did you provide a numpy array?")
    except IndexError:
        raise IndexError("The user-provided confusion matrix seems to not be a matrix. Check your arguments.")
    for curr in range(len(confusion_matrix)):
        row = confusion_matrix[curr]
        row_sum = 0
        for col in row:
            row_sum += col
        assert ((row_sum >= 0.99999999) and (row_sum <= 1.00000001)), "Error! Confusion matrix row "+str(curr)+" (counting from 0) sums to "+str(row_sum)+", which is not 100 % probability. Not possible."
    
    ## Extract the data ##
    
    # We require some API tools from the Labber API.
    import Labber
    
    def get_stepchannelvalue(StepChannels, name):
        for i, item in enumerate(StepChannels):
            if item['name'] == name:
                return item['values']
        return np.nan
    
    # User syntax repair
    if attempt_to_fix_string_input_argument:
        if isinstance(filepaths_to_plot, str):
            filepaths_to_plot = os.path.abspath(filepaths_to_plot)
            filepaths_to_plot = [filepaths_to_plot]
    
    # For all entries!
    output_vector = []
    for item in filepaths_to_plot:
        
        # Get current path
        filepath_to_file_with_probability_data_to_plot = item
        
        # Ensure that the filepath received can be processed.
        if type(filepath_to_file_with_probability_data_to_plot) == list:
            filepath_to_file_with_probability_data_to_plot = "".join(filepath_to_file_with_probability_data_to_plot)
        
        # Get trace data.
        with h5py.File(filepath_to_file_with_probability_data_to_plot, 'r') as h5f:
            
            # Gather data to plot.
            hfile = Labber.LogFile(filepath_to_file_with_probability_data_to_plot)
            
            # Figure out how many qubits are present in the data.
            ## This is done by looking for the longst-available ground state
            ## vector.
            bit_vector = ''
            number_of_qubits = 0
            for i in range(maximum_number_of_qubits_contained_in_file):
                bit_vector += '0'
                try:
                    data = hfile.getData(name = 'State Discriminator - Average 2Qstate P'+bit_vector)
                    number_of_qubits = len(bit_vector)
                    break
                except: # Eww, Labber is not well-written IMO.
                    # This trace didn't exist, it seems.
                    # But how would we know? Labber doesn't tell the difference.
                    pass
            assert number_of_qubits > 0, "Halted! The input data doesn't contain readout information for an all-ground-state vector (like P( |000> )). Or, the analysed data contains information for more than "+str(maximum_number_of_qubits_contained_in_file)+" qubits. In that case, increase the argument \"maximum_number_of_qubits_contained_in_file\"."
            if number_of_qubits == 1:
                print("Detected that the measurement executed on 1 qubit.")
            else:
                print("Detected that the measurement executed on "+str(number_of_qubits)+" qubits.")
            
            # Now, we know the number of qubits read out.
            # How many states are treated in the measurement?
            ## Check the size of the confusion matrix.
            ## We already checked that the confusion matrix is legal, see above.
            number_of_states = round(len(confusion_matrix) ** (1/number_of_qubits))
            
            # We now know all state vectors that we need to get from
            # the input file. Get them.
            big_ass_y_vector = []
            
            # We'll need the x-axis for later, let's get it here.
            x_axis = (hfile.getStepChannels())[0]['values']
            
            def basis_conversion(number, basis):
                signum = '-' if (number < 0) else ''
                number = abs(number)
                if number < basis:
                    return str(number)
                s = ''
                while number != 0:
                    s = str(number % basis) + s
                    number = number // basis
                return (signum + s)
            
            # Extract.
            all_investigated_states = []
            for row in range(len(confusion_matrix)):
                
                # Get the current to-be-checked state, as an int.
                state_in_number = int(basis_conversion(row, number_of_states))
                
                # state_in_number must be zero-padded.
                s_vector = str(state_in_number).zfill(number_of_qubits)
                
                ## Save the s_vector for later, it is used when plotting.
                all_investigated_states += [s_vector]
                
                # Get data.
                data = hfile.getData(name = 'State Discriminator - Average 2Qstate P'+str(s_vector))
                
                # Select specific row entries in the data?
                if not select_file_entry_to_extract == []:
                    # Go through the array-like object and save the sought-for data
                    sought_for_data = []
                    for item_to_keep in select_file_entry_to_extract:
                        sought_for_data += [data[item_to_keep]]
                    big_ass_y_vector += [np.array(sought_for_data)]
                else:
                    # Then store every entry.
                    
                    # Add to big_ass_y_vector
                    ## This will, for instance, add the |00> entry for
                    ## every single trace in the data.
                    big_ass_y_vector += [data]
            
            ## The data structure of big_ass_y_vector right now is: (example)
            ## [P00, P01 ... P21, P22] ... where P00 is a matrix list of:
            ##     [row0, row1 ... row40] ... where rowX is a trace of P00.
            ##        In total there will be Y number of rows, where Y is
            ##        the combined number of all measurement iteration
            ##        combinations. Like, [CZ_off, CZ_on] * [Dummy0 ... Dummy7]
            
            # Convert to multi-dimensional numpy array.
            big_ass_y_vector = np.array(big_ass_y_vector)
        
        ## Fit the data ##
        
        # For this, we need to define the output.
        big_ass_x_vector = np.zeros_like(big_ass_y_vector)
        
        # 1. The {P00, P01 .. P21, P22}-side, trace-side, and point-side,
        #       together form a cube in parameter space. The y-vectors
        #       that will be fitted, are the x-axes of the
        #       {trace,point}-coordinate in parameter space.
        # 2. We select a new trace, and begin collecting P00..P22 data.
        #       this data is the y-vector for that particular point of
        #       every single trace.
        # 3. Perform the fit that minimises
        #       | "some unknown input" · [Confusion matrix] | ^2 = 0,
        #       which gets you the x vector that was "the closest" to
        #       giving that y you had, based in the confusion matrix.
        # 4. Go to the next point. Repeat until every single datapoint has
        #       been massaged and stuffed into the x-vector.
        
        ## For all traces, for all points...
        ## IMPORTANT: doing big_ass_y_vector[:][0][0] -> returns all individual
        ##            points of the selected trace for some darn reason.
        ##            But individual index addressing does not!
        ##            Hence this hoop jumping here with the indices.
        number_of_two_qubit_states = len(big_ass_y_vector) ##  <--  Hoop.
        for all_traces in range(len(big_ass_y_vector[0][:])):
            for all_points in range(len(big_ass_y_vector[0][0][:])):
                
                # Construct y-vector, i.e. P00..P22 for this datapoint.
                y_vector = []
                for iii in range(number_of_two_qubit_states):
                    y_vector.append(big_ass_y_vector[iii][all_traces][all_points])
                y_vector = np.array(y_vector)
                
                # Use the confusion matrix to fit the data! Thus, get x.
                x_vector = fit_to_confusion_matrix(y_vector, confusion_matrix)
                
                # At this point, we have the confusion-adjusted vector x.
                # This data can go into our big_ass_x_vector cube.
                for jjj in range(number_of_two_qubit_states):
                    big_ass_x_vector[jjj][all_traces][all_points] = x_vector[jjj]
        
        # Is the user looking for a single qubit's state?
        if sum_states_of_vectors_in_order_to_select_single_qubit_probabilities != []:
            # The user is looking for a single qubit's state.
            
            # Check which indices of the big_ass_x_vector correspond to
            # the user-selected states.
            indices_to_sum = []
            for kk in range(len(sum_states_of_vectors_in_order_to_select_single_qubit_probabilities)):
                for ll in range(len(all_investigated_states)):
                    if (all_investigated_states[ll] == sum_states_of_vectors_in_order_to_select_single_qubit_probabilities[kk]):
                        indices_to_sum += [ll]
            
            # We now know which indices in the big_ass_x_vector corresponds
            # to the user-selected states. These indices are stored in
            # indices_to_sum. We'll define a separate vector for single-qubit
            # states.
            not_equally_big_ass_single_qubit_vector = np.zeros_like( big_ass_x_vector[0] )
            
            # This vector must now be filled up.
            for idx in indices_to_sum:
                not_equally_big_ass_single_qubit_vector = not_equally_big_ass_single_qubit_vector + big_ass_x_vector[idx]
            
            # The data structure not_equally_big_ass_single_qubit_vector
            # now contains the single-qubit information, as was selected
            # by the user.
        
        ## Output result ##
        
        # Plot?
        if plot_output or plot_output_export_path:
            
            # For all traces to be plotted:
            for trace_number in range(len(big_ass_x_vector[0])):
                
                # For all states investigated:
                for probability_investigated in range(len(big_ass_x_vector)):
                    
                    # What state is being checked right now?
                    state_investigated = all_investigated_states[probability_investigated]
                
                    # Get current trace.
                    trace = big_ass_x_vector[probability_investigated][trace_number]
                    
                    # Set figure size.
                    plt.figure(figsize = figure_size_tuple, dpi=600.0)
                    
                    # Define axis limits.
                    ax = plt.gca()
                    ax.set_xlim([x_axis[0], x_axis[-1]])
                    ax.set_ylim([0.0, 1.0])
                    
                    # Title?
                    if title != None:
                        if title == '':
                            title_to_plot = "P( |"+str(state_investigated)+"⟩ )"
                            use_this_title = ''
                        else:
                            title_to_plot  = title
                            use_this_title = title
                        plt.title(title_to_plot)
                    else:
                        # Then, no title for now.
                        # It will be added to the save file, if relevant.
                        use_this_title = ''
                    
                    # Choose colour for the plot.
                    if captivorum_colore_utere:
                        if abs(x_axis[0]) < (np.pi+0.1):
                            if ((trace_number % 2) == 0):
                                use_this_colour = "#34d2d6"
                            else:
                                use_this_colour = "#8934d6"
                        else:
                            figure_size_tuple = (2.654 * 2, 1.887)
                            if ((trace_number % 2) == 0):
                                use_this_colour = "#d63834"
                            else:
                                use_this_colour = "#81d634"
                    else:
                        # Select random colour.
                        use_this_colour = '#'+((hex(random.randint(0,256*256*256-1))).replace('0x','')).zfill(6)
                    
                    # Disable axis ticks.
                    plt.axis('off')
                    
                    # Enter data into the plot!
                    ##plt.plot(x_axis, trace, ':', color=use_this_colour)
                    ##plt.grid(visible=True, axis='both', color="#626262", linestyle='-', linewidth=5)
                    plt.scatter(x_axis, trace, s=10, color=use_this_colour)
                    
                    # Explicitly print the plotted data?
                    if captivorum_colore_utere:
                        print("PLOTTING THE FOLLOWING DATA:")
                        print(str(trace))
                    
                    # Save plot?
                    if plot_output_export_path != '':
                        use_this_title += 'trace'+str(trace_number)+'_P'+str(state_investigated)
                        print("Saving plot "+use_this_title+".png")
                        if plot_output_export_path.endswith("Desktop"):
                            print("You tried to name the export plot \"Desktop\", this is probably an error. Attempting to correct.")
                            plot_output_export_path = plot_output_export_path + "\\"
                        plt.savefig(plot_output_export_path+use_this_title+".png", bbox_inches="tight", pad_inches = 0, transparent=True)
                    
                    # Plot?
                    if plot_output:
                        plt.show()
                
                ## As a bonus, let's check if the user wants single-qubit data too.
                ## Remember that the number of traces will be the same.
                ## Thus we can check the variable trace_number here too.
                if sum_states_of_vectors_in_order_to_select_single_qubit_probabilities != []:
                    # The user wants to see the single-qubit data too.
                    trace = not_equally_big_ass_single_qubit_vector[trace_number]
                    
                    # Set figure size.
                    plt.figure(figsize = figure_size_tuple, dpi=600.0)
                    
                    # Define axis limits.
                    ax = plt.gca()
                    ax.set_xlim([x_axis[0], x_axis[-1]])
                    ax.set_ylim([0.0, 1.0])
                    
                    ## Here, we must figure out which qubit and what
                    ## probability the user is looking for.
                    state_character_matrix = []
                    for item in sum_states_of_vectors_in_order_to_select_single_qubit_probabilities:
                        state_character_matrix += [([*item])]
                    for all_rows in range(len(state_character_matrix)):
                        for all_cols in range(len(state_character_matrix[0])):
                            state_character_matrix[all_rows][all_cols] = int(state_character_matrix[all_rows][all_cols])
                    # Cast to numpy array. Check which column only has the same number (i.e. state)
                    state_character_matrix = np.array(state_character_matrix)
                    comparison_table = (state_character_matrix == state_character_matrix[0,:])
                    column_found = False
                    single_qubit_checked = 0
                    single_qubit_state = 0
                    for column_checked in range(len(comparison_table[0,:])):
                        if np.all(comparison_table[:,column_checked] == True):
                            # Done, this column shows the qubit and the
                            # state which is permanent throughout.
                            if not column_found:
                                column_found = True
                                single_qubit_checked = column_checked+1
                                single_qubit_state = state_character_matrix[0,column_checked]
                            else:
                                raise AttributeError("Error! The user provided some combination of multi-qubit states, that when put together, does not equal a single qubit's state probability.")
                    if not column_found:
                        raise AttributeError("Error! The user provided some combination of multi-qubit states, that when put together, does not equal a single qubit's state probability.")
                    ## At this point, we know which qubit the user selected,
                    ## And which state's probability is being checked on it.
                    
                    # Title?
                    if title != None:
                        if title == '':
                            title_to_plot = "P( Q"+str(single_qubit_checked)+" = |"+str(single_qubit_state)+"⟩ )"
                            use_this_title = ''
                        else:
                            title_to_plot  = title
                            use_this_title = title
                        plt.title(title_to_plot)
                    else:
                        # Then, no title for now.
                        # It will be added to the save file, if relevant.
                        use_this_title = ''
                    
                    # Choose colour for the plot.
                    if captivorum_colore_utere:
                        if abs(x_axis[0]) < (np.pi+0.1):
                            if ((trace_number % 2) == 0):
                                use_this_colour = "#34d2d6"
                            else:
                                use_this_colour = "#8934d6"
                        else:
                            if ((trace_number % 2) == 0):
                                use_this_colour = "#d63834"
                            else:
                                use_this_colour = "#81d634"
                    else:
                        # Select random colour.
                        use_this_colour = '#'+((hex(random.randint(0,256*256*256-1))).replace('0x','')).zfill(6)
                    
                    # Disable axis ticks.
                    plt.axis('off')
                    
                    # Enter data into the plot!
                    plt.plot(x_axis, trace, color=use_this_colour)
                    
                    # Save plot?
                    if plot_output_export_path != '':
                        use_this_title += 'trace'+str(trace_number)+'_Q'+str(single_qubit_checked)+'_P'+str(single_qubit_state)
                        print("Saving plot "+use_this_title+".png")
                        if plot_output_export_path.endswith("Desktop"):
                            print("You tried to name the export plot \"Desktop\", this is probably an error. Attempting to correct.")
                            plot_output_export_path = plot_output_export_path + "\\"
                        plt.savefig(plot_output_export_path+use_this_title+".png", bbox_inches="tight", pad_inches = 0, transparent=True)
                    
                    # Plot?
                    if plot_output:
                        plt.show()
            
            
            
        # Save into new .hdf5 file?
        if hdf5_export_path != '':
            raise NotImplementedError("Halted! Exporting the data into a Log Browser compatible .hdf5 file has not been implemented.")
        
        # Store in larger multi-dimensional matrix for outputting.
        output_vector += [big_ass_x_vector]
    
    # Return
    return output_vector

def fit_to_confusion_matrix(y_vector, confusion_matrix):
    ''' Given some vector of P( |00> ) ... P( |11> ) data,
        use a confusion matrix to extract the most likely
        actual, non-confused output x_vector.
    '''
    
    # Let's provide an initial guess for the x_vector.
    x_vector = y_vector
    
    # Fit! Using least-square linear fitting.
    '''  This fitting attempts to find x for 0.5 * |Ax - b|^2 = "minimum"
    '''
    success = lsq_linear(
        A = confusion_matrix,
        b = y_vector,
    )
    
    # Extract x-vector and return.
    return success['x']




def get_file_path_of_discrimination_json():
    ''' Get name, file path for discriminator script.
    '''
    # Go!
    path_to_discriminator = __file__
    if not ('QPU interfaces' in path_to_discriminator):
        raise OSError("The provided path to the discriminator settings file did not include a 'Core parameters' folder. A discriminator settings file was not created.")
    call_root = path_to_discriminator.split('QPU interfaces',1)[0]
    not_call_root = path_to_discriminator.split('QPU interfaces',1)[1]
    name_of_interface = not_call_root.split(os.sep)[1] # Note: the first entry will be a '' in the split-list.
    
    # Make path to the discriminator JSON, make a discriminator_settings.json file if missing.
    discriminator_json_folder_path = os.path.join(call_root, 'Core parameters')
    discriminator_json_folder_path = os.path.join(discriminator_json_folder_path, name_of_interface)
    full_path_to_discriminator_settings_json = os.path.join(discriminator_json_folder_path,'discriminator_settings.json')
    
    assert ( not (".py" in name_of_interface)), \
        "Critical error: could not establish name of QPU interface. "        +\
        "You have likely not followed the proper folder structure. Make "    +\
        "sure the function scripts are placed within a folder (with some "   +\
        "name corresponding to your chosen interface), and that this folder "+\
        "is placed in a folder called \"QPU interfaces\"."
    
    return [full_path_to_discriminator_settings_json, call_root, name_of_interface]
    
def save_discriminator_settings(json_data = [], verbose = False):
    ''' Update the discriminator_settings.json file.
    '''
    
    # Get file path of the discriminator_settings.json file.
    full_path = (get_file_path_of_discrimination_json())[0]
    
    if verbose:
        print("Will start dumping JSON data to file \""+str(full_path)+"\"")
    with open(full_path, "w") as fjson:
        json.dump(json_data, fjson, indent=4)
        fjson.close()
    if verbose:
        print("Finished dumping JSON data to file.")
    
def load_discriminator_settings(resonator_transmon_pair = None, verbose = False):
    ''' Get the content of the discriminator_settings.json file.
    '''
    
    # Get file path of the discriminator_settings.json file.
    full_path = (get_file_path_of_discrimination_json())[0]
    
    try:
        with open(full_path, "r") as fjson:
            loaded_json_data = json.load(fjson)
            '''          .,,*/(#########%%%%%%%%%%((/***,,,*,,*,,,,,,,,,,,
                         .#####################%#%%%%/*,,...,,,,,,,,,,,,,,
                       ,######################%%%%%%%#%#(*,.......     .. 
                     (###########################%%%%%%####(.             
                   ,##(##%##########%###%########(//(########*.           
                  *#########(((((((#**/*((, ..,       (#######(.          
                 .#####(((***/. .,,     .*              (###%##(.         
                 (((/*.*. ,,                             .(##%%#/,..   ...
            ....,#((/*   *                                ./(###/,..     .
            ,***/(((/,                           ,*///*    .(##%(,,..     
            ***//((/*.        .****,                :       *(##/,,..    .
            (((((((//,          ::            .,, */,        ((  ....  ...
            ((((/((//*               ..         .            /      ......
            ((((/,   ,.                                      .     ..,,,,,
            ((((/.                                                ..,,,,,,
            ((((/*                             **                .,,****,,
            ((((//*  ..                                         ,,********
            ####((//.   ..             .,,..,,...****      .,, ,,,***,****
            #######((/.  ,,         ..               **   ,*,.          ..
            /(((((((//////**,       ,  (%%%%%%%%%%%%(,*/**////.           
            ((((/(///****//////*.  ,,.%%%%%%%%%%%%%#*,*/((((//,,.....     
            ///(((((///////((((/((((/  *%%(((/(/*(/ . *((((((/**,,,,,*****
            ..,,****,,,,,,,*/(((((((/,    ....,,.    ,/((###*%%%#,,,,,,**/
            ..,,,,,,,,....,,,./(##(//*.          ..,**/(####.%%%%%#*,....,
            ..,,,,,,,*****/#%. ./(((///*    ,/(((/,.,/(##(/*.(%%%%%#(,,,.,
            ,,,,,,***//(%%%##*     /(//*,.     .,,**/((((/*, .&&%%%%%(#***
            ********/(#%%%%%%*        #(***.  ..**/(((,....    &%%%%%%#%#(
            ////////(%%%%#%%%             (((#(/*//*...         %%%%%%%%%%
            //////(#%%%%%#%%/                                   %%%%%%%%%%
            ///((#%%%%%%%%%%#                                  %%%%%%%%%%%
            ((#%%%%%%%%%%%%%%                JASON!           *%%%%%%%%'''
        
    except FileNotFoundError:
        # The discriminator settings JSON file has likely not been
        # initiated. Let's do that. Then reload the data.
        if verbose:
            print("The JSON settings file did not exist. It was thus initiated (created).")
        if resonator_transmon_pair != None:
            initiate_discriminator_settings_file(resonator_transmon_pair)
        else:
            initiate_discriminator_settings_file(0)
        with open(full_path, "r") as fjson:
            loaded_json_data = json.load(fjson)
    
    if verbose:
        print("Loaded JSON data from file \""+str(full_path)+"\"")
    
    return loaded_json_data
        
        
