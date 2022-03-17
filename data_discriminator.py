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
from datetime import datetime

def initiate_discriminator_settings_file(
    no_resonators_qubit_pairs = 1
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
    for ii in range(no_resonators_qubit_pairs):
        json_dict.update(
            {'resonator_'+str(ii): 
                {'readout_state_0':
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
    readout_state = None,
    ):
    ''' Update the discrimination.json file based on supplied data.
    '''
    
    # Get file paths.
    full_path_to_discriminator_settings_json = \
        (get_file_path_of_discrimination_json())[0]
    
    # Load the file at path_to_data.
    with h5py.File(os.path.abspath(path_to_data), 'r') as h5f:
        processing_volume = h5f["processed_data"][()]
        prepared_qubit_states = h5f["prepared_qubit_states"][()]
        
        # Automatically establish what state was gathered?
        if readout_state != None:
            # No: use the user-provided readout state.
            readout_state = int(readout_state)
        else:
            # Yes: then find out the state from the read file.
            readout_state_found = False
            readout_state_numerical = 0
            for state_char in ['g','e','f','h']:
                if not readout_state_found:
                    try:
                        readout_state_frequency = h5f.attrs["readout_freq_"+state_char+"_state"]
                        readout_state_found = True
                    except KeyError:
                        readout_state_numerical += 1
            assert readout_state_found, "Error: could not establish what state the readout was done in. States |g⟩, |e⟩, |f⟩, |h⟩ were considered."
            readout_state = int(readout_state_numerical)
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
    
    # Analyse. NOTE! For now, only a single resonator is analysed
    # from the processing volume. This is likely a TODO.
    centre = []
    for row in range(len( (processing_volume[0])[:] )):
        # Run a complex mean on all complex values.
        curr_data  = (processing_volume[0])[row,:]
        centre.append( np.mean(curr_data) )
    
    # Get JSON.
    with open(full_path_to_discriminator_settings_json, "r") as fjson:
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
    
    # Save this centre for this state X for resonator Y in the JSON.
    # Update missing keys if any.
    for jj in range(len(prepared_qubit_states)):
        # The key might be non-existant. Then do whatever we want.
        try:
            curr_dict = (loaded_json_data['resonator_'+str(resonator_transmon_pair)])['readout_state_'+str(readout_state)]
            # The key existed. Then update. Remember that the key will update.
            # There is no need to re-save data into the loaded_json_data array.
            curr_dict.update( \
                { \
                    'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_real' : centre[jj].real, \
                    'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_imag' : centre[jj].imag  \
                }
            )
        except KeyError:
            top_dict = loaded_json_data['resonator_'+str(resonator_transmon_pair)]
            top_dict.update( \
                {'readout_state_'+str(readout_state) : \
                    {
                        'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_real' : centre[jj].real, \
                        'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_imag' : centre[jj].imag  \
                    }
                }
            )
    
    # Save updated JSON.
    with open(full_path_to_discriminator_settings_json, "w") as fjson:
        json.dump(loaded_json_data, fjson, indent=4)
    
    
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
    
    return [full_path_to_discriminator_settings_json, call_root, name_of_interface]