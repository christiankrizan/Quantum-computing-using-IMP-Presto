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
            {'resonator_'+str(ii): dict( qubit_state_0_centre_real = 0, qubit_state_0_centre_imag = 0 )}
        )
    
    # Build JSON file.
    with open(full_path_to_discriminator_settings_json, 'w') as json_file:
        json.dump(json_dict, json_file, indent = 4)


def calculate_and_update_resonator_value(
    path_to_data,
    resonator_transmon_pair,
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
        ##    '''repetition_delay = h5f.attrs["repetition_delay"] TODO STRING SAMPLE'''
        ''' TODO: Make sure that when collecting IQ plane data,
        to what extent does it matter that the IQ data for the individual
        state centres have been collected using single-resonator readout?
        Should there be a step here where values for several resonators
        are picked out?'''
    
    # Analyse. NOTE! For now, only a single resonator is analysed
    # from the processing volume. This is likely a TODO.
    centre = []
    for row in range((processing_volume[0])[:]):
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
        ((#%%%%%%%%%%%%%%                                 *%%%%%%%%'''
        
        ''' JASON! '''
    
    # Save this centre for this state X for resonator Y in the JSON.
    for jj in range(prepared_qubit_states):
        ## TODO: Here, I am not appending dicts properly for a given resonator.
        ##loaded_json_data['resonator_'+str(resonator_transmon_pair)] = \
        ##    { \
        ##        'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_real' : centre[jj].real, \
        ##        'qubit_state_'+str(prepared_qubit_states[jj])+'_centre_imag' : centre[jj].imag  \
        ##    }
    
    # Save updated JSON.
    with open(full_path_to_discriminator_settings_json, "w") as fjson:
        json.dump(loaded_json_data, fjson)
    
    
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