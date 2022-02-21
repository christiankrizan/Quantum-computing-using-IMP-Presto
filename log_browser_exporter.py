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
import shutil
import numpy as np
from datetime import datetime
from presto.utils import rotate_opt

def save(
    timestamp,
    ext_keys,
    log_dict_list,
    
    time_matrix,
    fetched_data_arr,
    resonator_freq_if_arrays_to_fft,
    integration_window_start,
    integration_window_stop,
    
    path_to_script,
    use_log_browser_database,
    
    inner_loop_size,
    outer_loop_size,
    
    save_complex_data = True,
    source_code_of_executing_file = '',
    append_to_log_name_before_timestamp = '',
    append_to_log_name_after_timestamp  = '',
    select_resonator_for_single_log_export = '',
    ):
    ''' Function for saving an IMP Presto measurement in an HDF5-format
        that is compatible with Labber's Log Browser.
        
        Please note that the timestamp *should* really be fed into this routine
        as an argument, because there is an option to select resonators
        for single log exports. If the timestamp is fetched inside this
        very save routine in this very file, then the exported files
        will bear different timestamps, and be sorted weirdly
        when viewed in folders. This has happened. So don't change how
        the timestamp is fed into this routine.
    '''
    
    # Get name, file path and time for logfile.
    current_dir, name_of_running_script = os.path.split(path_to_script)
    year_num    = (datetime.now()).strftime("%Y")
    month_num   = (datetime.now()).strftime("%m")
    day_num     = (datetime.now()).strftime("%d")
    
    # Prepare path names needed for creating the folder tree.
    # First, find the root directory of whatever is executing the calling script.
    if not ('QPU interfaces' in current_dir):
        raise OSError("The log browser export was called from a script not residing within a QPU interfaces folder. The save action was halted before finishing.")    
    call_root = current_dir.split('QPU interfaces',1)[0]
    data_folder_path = os.path.join(call_root, 'Data output folder')
    path1 = os.path.join(data_folder_path, year_num)
    path2 = os.path.join(data_folder_path, year_num, month_num)
    path3 = os.path.join(data_folder_path, year_num, month_num, 'Data_' + day_num)
    for lb_path_name in [path1, path2, path3]:
        if not os.path.exists(lb_path_name):
            os.makedirs(lb_path_name)
    
    # Get file name of calling script.
    script_filename = os.path.splitext(name_of_running_script)[0]
    
    # Touch up on user-input strings in the calling script.
    if (not append_to_log_name_after_timestamp.startswith('_')) and (append_to_log_name_after_timestamp != ''):
        append_to_log_name_after_timestamp = '_' + append_to_log_name_after_timestamp
    if (not append_to_log_name_before_timestamp.startswith('_')) and (append_to_log_name_before_timestamp != ''):
        append_to_log_name_before_timestamp = '_' + append_to_log_name_before_timestamp
    if (not timestamp.startswith('_')) and (timestamp != ''):
        timestamp = '_' + timestamp
    
    # Get a preliminary name for the H5PY savefile-string.
    # Depending on the state of the Log Browser export, this string may change.
    savefile_string_h5py = script_filename + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + '.h5'
    
    # Has the user set up the calling script so that the X and Z axes are
    # reversed? I.e. "the graph is rotated -90° in the Log Browser."
    if (len(ext_keys) > 1) and (inner_loop_size != outer_loop_size):
        first_dict  = ext_keys[0]
        second_dict = ext_keys[1]
        if (len(first_dict.get('values')) == outer_loop_size) and (len(second_dict.get('values')) == inner_loop_size):
            print("Detected external key reversal. Will flip axes "+first_dict.get('name')+" and "+second_dict.get('name')+".")
            tempflip = inner_loop_size
            inner_loop_size = outer_loop_size
            outer_loop_size = tempflip
    
    # Get index corresponding to integration_window_start and integration_window_stop respectively
    t_span = integration_window_stop - integration_window_start
    integration_start_index = np.argmin(np.abs(time_matrix - integration_window_start))
    integration_stop_index = np.argmin(np.abs(time_matrix - integration_window_stop))
    integr_indices = np.arange(integration_start_index, integration_stop_index)
    
    # Attempt an export to Labber's Log Browser!
    labber_import_worked = False
    try:
        import Labber
        labber_import_worked = True
    except:
        print("Could not import the Labber library; " + \
              "no data was saved in the Log Browser compatible format")
    if labber_import_worked:
        # Create the log file. Note that the Log Browser API is bugged,
        # and adds a duplicate '.hdf5' file ending when using the database.
        if use_log_browser_database:
            savefile_string = script_filename + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp
            savefile_string_h5py = savefile_string + '.h5'
        else:
            savefile_string = script_filename + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + '.hdf5'
            savefile_string_h5py = savefile_string.replace('.hdf5','.h5')
        print("... building Log Browser-compatible .HDF5 log file: " + savefile_string)
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
        
        # Get index corresponding to integration_window_start and integration_window_stop respectively
        #t_span = integration_window_stop - integration_window_start
        #integration_start_index = np.argmin(np.abs(time_matrix - integration_window_start))
        #integration_stop_index = np.argmin(np.abs(time_matrix - integration_window_stop))
        #integr_indices = np.arange(integration_start_index, integration_stop_index)
        
        # Is the readout non-multiplexed? (Will there be no FFT involved?)
        ''' TODO:   The complex data storage routine
                    should be made compatible for multiplexed readout!
                    Aka. move it out of this if case, and/or simply
                    omit the if case and always make the analysis FFT-based.'''
        processing_volume = [] # Declare the processing volume (tensor)
        if len(resonator_freq_if_arrays_to_fft) < 2:
        
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
                ## processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(angles, axis=-1))
                ## processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1) * np.exp(1j * np.mean(np.angle(fetched_data_arr[:, 0, integr_indices]), axis=-1))
            else:
                processing_arr = np.mean(np.abs(fetched_data_arr[:, 0, integr_indices]), axis=-1)
            
            # Reshape depending on the repeat variable, as well as the inner loop
            # of the sequencer program.
            processing_arr.shape = (outer_loop_size, inner_loop_size)
            
            # Put the array into the processing list.
            processing_volume.append( processing_arr )
        
        else:
            # Multiplexed readout.
            # Acquire time step needed for returning the DFT sample frequencies.
            dt = time_matrix[1] - time_matrix[0]
            nr_samples = len(integr_indices)
            freq_arr = np.fft.fftfreq(nr_samples, dt)
            
            # Execute complex FFT.
            resp_fft = np.fft.fft(fetched_data_arr[:, 0, integr_indices], axis=-1) / len(integr_indices)
            
            # Get new indices for the new processing_arr arrays.
            integr_indices_list = []
            for _ro_freq_if in resonator_freq_if_arrays_to_fft:
                integr_indices_list.append( np.argmin(np.abs(freq_arr - _ro_freq_if)) )
            
            # Build new processing_arr arrays.
            for _item in integr_indices_list:
                processing_volume.append( 2 * resp_fft[:, _item] )
            
            # Reshape the data to account for repeats.
            for mm in range(len(processing_volume[:])):
                fetch = processing_volume[mm]
                fetch.shape = (outer_loop_size, inner_loop_size)
                # TODO: perhaps this absolute-value step should be remade?
                processing_volume[mm] = np.abs(fetch)
        
        # For every row in processing volume:
        print("... storing processed data into the .HDF5 file.")
        
        # TODO! This part should cover an arbitrary number of fetched_data_arr arrays!
        if select_resonator_for_single_log_export == '':
            # Then store multiplexed!
            # For every loop entry that is to be stored in this log:
            for outer_loop_i in range(outer_loop_size):
                f.addEntry({
                    (log_dict_list[0])['name']: (processing_volume[0])[outer_loop_i, :],
                    (log_dict_list[1])['name']: (processing_volume[1])[outer_loop_i, :]
                })
        else:
            # TODO  I think there is no usage case where this for-loop should be here.
            #       It should be removed.
            for log_i in range(len(log_dict_list[:])):
            
                # ... and for every loop entry that is to be stored in this log:
                for outer_loop_i in range(outer_loop_size):
                    f.addEntry({
                        (log_dict_list[log_i])['name']: (processing_volume[int(select_resonator_for_single_log_export)])[outer_loop_i, :]
                    })

        
        # Check if the hdf5 file was created in the local directory.
        # This would happen if you change use_data to False in the
        # Labber.createLogFile_ForData call. If so, move it to an appropriate
        # directory. Make directories where necessary.
        success_message = " in the Log Browser directory!"
        save_path = os.path.join(path3, savefile_string)  # Full save path
        if os.path.isfile(os.path.join(current_dir, savefile_string)):
            shutil.move( os.path.join(current_dir, savefile_string) , save_path)
            success_message = ", see " + save_path
        
        # Print success message.
        print("Data saved" + success_message)
    
    
    
    