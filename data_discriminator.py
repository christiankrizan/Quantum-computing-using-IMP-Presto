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
import numpy as np
from numpy import hanning as von_hann
from datetime import datetime

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


def calculate_and_update_resonator_value(
    path_to_data,
    resonator_transmon_pair = None,
    do_not_update_discriminator_settings = False,
    ):
    ''' Update the discrimination.json file based on supplied data.
    '''
    
    # Load the file at path_to_data.
    with h5py.File(os.path.abspath(path_to_data), 'r') as h5f:
        processed_data = h5f["processed_data"][()]
        prepared_qubit_states = h5f["prepared_qubit_states"][()]
        
        ## # Automatically establish what state was gathered?
        ## if readout_state != None:
        ##     # No: use the user-provided readout state.
        ##     readout_state = int(readout_state)
        ## else:
        ##     # Yes: then find out the state from the read file.
        ##     readout_state_found = False
        ##     readout_state_numerical = 0
        ##     for state_char in ['g','e','f','h']:
        ##         if not readout_state_found:
        ##             try:
        ##                 readout_state_frequency = h5f.attrs["readout_freq_"+state_char+"_state"]
        ##                 readout_state_found = True
        ##             except KeyError:
        ##                 readout_state_numerical += 1
        ##     assert readout_state_found, "Error: could not establish what state the readout was done in. States |g⟩, |e⟩, |f⟩, |h⟩ were considered."
        ##     readout_state = int(readout_state_numerical)
        ''' TODO: Make sure that when collecting IQ plane data,
        to what extent does it matter that the IQ data for the individual
        state centres have been collected using single-resonator readout?
        Should there be a step here where values for several resonators
        are picked out?'''
        
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
    
    # Analyse. NOTE! For now, only a single resonator is analysed
    # from the processed data. This is likely a TODO.
    centre = []
    for row in range(len( (processed_data[0])[:] )):
        # Run a complex mean on all complex values.
        curr_data  = (processed_data[0])[row,:]
        centre.append( np.mean(curr_data) )
    
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
                    'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_real' : centre[jj].real, \
                    'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_imag' : centre[jj].imag  \
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
    
    # Save updated JSON?
    if (not do_not_update_discriminator_settings):
        save_discriminator_settings(loaded_json_data)
    
    # Also get the area (and more) spanned by the qubit states for
    # this resonator. This value can also be returned.
    return find_area_and_mean_distance_between_all_qubit_states(resonator_transmon_pair)
    

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
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the discriminator was provided a string type. Expected raw data. The provided variable was: "+str(extracted_data)
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
    
    
def find_area_and_mean_distance_between_all_qubit_states(
    resonator_transmon_pair
    ):
    ''' Acquire the area and perimeter spanned by the readout states
        for a given resonator. Also, update the JSON settings file.
        Area * perimeter will typically have the unit cubic volt [V^3].
    '''
    
    # Get JSON.
    loaded_json_data = load_discriminator_settings(resonator_transmon_pair)
    
    # For all qubit states in the readout, gather area and perimeter
    # where possible.
    area_spanned_by_qubit_states = []
    mean_distance_between_all_states = []
    hamiltonian_path_perimeter = []
    
    # Check the number of complex coordinates provided.
    resonator_dict = loaded_json_data['resonator_'+str(resonator_transmon_pair)]
    curr_investigated_dict = (resonator_dict)['qubit_states']
    no_qubit_states = len(curr_investigated_dict) // 2
    
    # Check whether the program can find and area and a mean distance
    # between all states given the acquired data.
    curr_area_spanned_by_qubit_states = None
    curr_mean_distance_between_all_states = None
    curr_hamiltonian_path_perimeter = None
    if no_qubit_states <= 2:
        if no_qubit_states <= 1:
            # If there are <= 1 states to look at, then one can not span
            # an area for the readout done in |n>. Nor a perimeter.
            print( \
                "Warning: the discriminator_settings.json file for resonator "   +\
                str(resonator_transmon_pair)+" does not contain a sufficient "   +\
                "number of qubit states for calculating an area nor a mean "     +\
                "distance between all states (№ states = "                       +\
                str(no_qubit_states) + "). Returning \"None\" and \"None\".")
        else:
            # If there are 2 states to look at, then at least a vector
            # can be acquired giving the length between the two states.
            print( \
                "Warning: the discriminator_settings.json file for resonator "   +\
                str(resonator_transmon_pair)+" only contains a sufficient "      +\
                "number of states for calculating a mean distance between "      +\
                "two states (№ states = "+str(no_qubit_states)+"). The "         +\
                "returned area will be \"None\".")
            curr_mean_distance_between_all_states = -1
    else:
        # We have a sufficient number of states acquired. Remove the None
        # status to signal down below that the area and mean state distance
        # will be calculated.
        curr_area_spanned_by_qubit_states = -1
        curr_mean_distance_between_all_states = -1
    
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
        vertices_x_arr.append(curr_investigated_dict['qubit_state_'+str(pp)+'_centre_real'])
        vertices_y_arr.append(curr_investigated_dict['qubit_state_'+str(pp)+'_centre_imag'])
        vertices = np.append( vertices, (vertices_x_arr[pp] + vertices_y_arr[pp]*1j) )
    
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
    
    # Update the local JSON that will be saved to file.
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
    
    return area_spanned_by_qubit_states, mean_distance_between_all_states, hamiltonian_path_perimeter
    
    
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
        
        
