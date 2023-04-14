#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import os
import numpy as np

def verify_supported_input_argument_to_fit(raw_data_or_path_to_data):
    ''' This function verifies whether the user is providing some kind
        of legible, supported, input argument for the fitting routines.
        
        The user may provide several types of input argument types for
        raw_data_or_path_to_data. Let's figure out what was provided.
            > Folder
            > Filepath
            > Raw data
            > List of filepaths
            > List of raw data arrays
        
        The function returns a list containing five booleans.
        At least one boolean will be true. That boolean,
        shows what kind of supported input, that the user provided.
        
        Essentially,
        return [
            the_user_provided_a_folder,
            the_user_provided_a_file,
            the_user_provided_raw_data,
            the_user_provided_a_list_of_files,
            the_user_provided_a_list_of_raw_data
        ]
    '''
    
    # Did the user provide a string? Then, it might be a folder,
    # or a filepath.
    the_user_provided_a_folder  = False
    the_user_provided_a_file    = False
    the_user_provided_raw_data  = False
    the_user_provided_a_list_of_files    = False
    the_user_provided_a_list_of_raw_data = False
    if isinstance(raw_data_or_path_to_data, str):
        # The user did provide a string. Folder, filepath, or other?
        if os.path.isdir(raw_data_or_path_to_data):
            # Folder!
            the_user_provided_a_folder = True
        elif os.path.isfile(raw_data_or_path_to_data):
            # File!
            the_user_provided_a_file = True
        else:
            # Just a string, dafuq?
            raise TypeError("Error! A fitting function was provided with an input argument string, but neither a folder nor a file could be found following that string.")
    elif (isinstance(raw_data_or_path_to_data, list)) or (type(raw_data_or_path_to_data) == np.ndarray): # Ok, did the user provide a list of things then?
        
        # Check if it's just raw data. Check the first resonator in that data.
        try:
            for uu in range(len(raw_data_or_path_to_data[0])):
                discard = raw_data_or_path_to_data[uu] + 313.0
            # If we made it this far, we're good.
            the_user_provided_raw_data = True
        except TypeError:
            # Not a supported datatype.
            pass
        
        # Did the check fail? Then the list might contain file paths.
        if (not the_user_provided_raw_data):
            # The user is providing a list!
            # This list could be a list of data files to fit.
            all_entries_of_input_list_were_files = True
            all_entries_of_input_list_were_raw_data_lists = True
            for entry in raw_data_or_path_to_data:
                
                # Slight optimisation.
                if all_entries_of_input_list_were_files:
                    if (not os.path.isfile( entry )):
                        all_entries_of_input_list_were_files = False
                
                # Slight optimisation.
                if all_entries_of_input_list_were_raw_data_lists:
                    if ( isinstance(entry, list) or isinstance(entry, np.ndarray) ):
                        # The entry is a list! Does it contain data?
                        # Attempt operation on the first resonator in the data.
                        try:
                            discard = (entry[0])[0] + 313.0
                        except TypeError:
                            # Not a supported datatype.
                            all_entries_of_input_list_were_raw_data_lists = False
                    else:
                        # Not a list (or numpy array), so not raw data.
                        all_entries_of_input_list_were_raw_data_lists = False
            
            if (all_entries_of_input_list_were_files) and (all_entries_of_input_list_were_raw_data_lists):
                raise RuntimeError("Error! Somehow, all entries of the user-supported list were identified to be both filepaths, and numeric raw data. This fact does not compute.")
            
            elif all_entries_of_input_list_were_files:
                # The user provided a list of files.
                the_user_provided_a_list_of_files = True
            
            elif all_entries_of_input_list_were_raw_data_lists:
                # The user provided a list of raw data!
                the_user_provided_a_list_of_raw_data = True
    
    # At this point, let's check whether the fitter was provided with at
    # least something legible.
    legible = [ \
        the_user_provided_a_folder           , \
        the_user_provided_a_file             , \
        the_user_provided_raw_data           , \
        the_user_provided_a_list_of_files    , \
        the_user_provided_a_list_of_raw_data ]
    assert sum(legible) == 1, \
        "Error! The fitter was provided with incomprehensible input. The "   +\
        "input value(s) in raw_data_or_path_to_data, cannot uniquely be "    +\
        "identified to be *only* one of the following supported types: "     +\
        "folder; filepath; raw data array; a list of filepaths; a list of "  +\
        "raw data arrays."
    
    # Now, we know what type of input that the user provided.
    return legible