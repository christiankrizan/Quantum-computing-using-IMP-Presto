#####################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/upload/main
#####################################################################################

import os
import sys
import time
import h5py
import Labber
import shutil
import numpy as np
from datetime import datetime
from presto.utils import rotate_opt

def save(
    ext_keys,
    log_dict_list,
    
    time_matrix,
    fetched_data_arr,
    integration_window_start,
    integration_window_stop,
    axes,
    
    save_complex_data,
    path_to_script,
    use_log_browser_database,
    
    inner_loop_size,
    outer_loop_size,
    ):
    ''' Function for saving an IMP Presto measurement in an HDF5-format
        that is compatible with Labber's Log Browser.
    '''
    
    # Get name, file path and time for logfile.
    current_dir, name_of_running_script = os.path.split(path_to_script)
    year_num    = (datetime.now()).strftime("%Y")
    month_num   = (datetime.now()).strftime("%m")
    day_num     = (datetime.now()).strftime("%d")
    
    # Prepare path names needed for creating the folder tree.
    path1 = os.path.join(current_dir, year_num)
    path2 = os.path.join(current_dir, year_num, month_num)
    path3 = os.path.join(current_dir, year_num, month_num, 'Data_' + day_num)
    
    # Make file name and its timestamp.
    script_filename = os.path.splitext(name_of_running_script)[0]  # Name of current script
    timestamp = (datetime.now()).strftime("%d-%b-%Y_(%H_%M_%S)") # Date and time
    
    # Create the log file. Note that the Log Browser API is bugged,
    # and adds a duplicate '.hdf5' file ending when using the database.
    if use_log_browser_database:
        savefile_string = script_filename + '_' + timestamp
    else:
        savefile_string = script_filename + '_' + timestamp + '.hdf5'
    f = Labber.createLogFile_ForData(
        savefile_string,
        log_dict_list,
        step_channels = ext_keys,
        use_database  = use_log_browser_database
    )
    
    # Set project name, tag, and user in logfile.
    f.setProject(script_filename)
    f.setTags('krizan')
    f.setUser('Christian Križan')
    
    # Split fetched_data_arr into repeats:
    # fetched_data_arr SHAPE: num_stores * repeat_count, num_ports, smpls_per_store
    t_span = integration_window_stop - integration_window_start
    
    # Get index corresponding to integration_window_start and integration_window_stop respectively
    integration_start_index = np.argmin(np.abs(time_matrix - integration_window_start))
    integration_stop_index = np.argmin(np.abs(time_matrix - integration_window_stop))
    integr_indices = np.arange(integration_start_index, integration_stop_index)
    
    # Construct a matrix, where every row is an integrated sampling
    # sequence corresponding to exactly one bias point.
    
    if(save_complex_data):
        angles = np.angle(fetched_data_arr[:, 0, integr_indices], deg=False)
        rows, cols = np.shape(angles)
        for row in range(rows):
            for col in range(cols):
                #if angles[row][col] < 0.0:
                angles[row][col] += (2.0 * np.pi)
        angles_mean = np.mean(angles, axis=-1)
        mean_len = len(angles_mean)
        for row in range(mean_len):
            angles_mean[row] -= (2.0 * np.pi)
        processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * angles_mean)
        ##processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(angles, axis=-1))
        ## processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(np.angle(fetched_data_arr[:, 0, integr_indices]), axis=-1))
    else:
        processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1)
    
    # Reshape depending on the repeat variable, as well as the inner loop
    # of the sequencer program.
    processing_arr.shape = (outer_loop_size, inner_loop_size)
    
    # For every array stored as log traces:
    for ts in range(0,len(log_dict_list)):
    
        # For every row in processing arr:
        for i in range(len(processing_arr[:])):
        
            # Add an entry in the log browser file.
            f.addEntry( {(log_dict_list[ts])['name']: processing_arr[i,:]} )
    
    # Check if the hdf5 file was created in the local directory.
    # This would happen if you change use_data to False in the
    # Labber.createLogFile_ForData call. If so, move it to an appropriate
    # directory. Make directories where necessary.
    success_message = " in the Log Browser directory!"
    if os.path.isfile(os.path.join(current_dir, savefile_string)):
        for lb_path_name in [path1, path2, path3]:
            if not os.path.exists(lb_path_name):
                os.makedirs(lb_path_name)
        save_path = os.path.join(path3, savefile_string)  # Full save path
        shutil.move( os.path.join(current_dir, savefile_string) , save_path)
        success_message = ", see " + save_path
    
    # Print final success message.
    print("Data saved" + success_message)
    