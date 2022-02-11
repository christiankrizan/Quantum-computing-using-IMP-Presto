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
    path1 = os.path.join(current_dir, year_num)
    path2 = os.path.join(current_dir, year_num, month_num)
    path3 = os.path.join(current_dir, year_num, month_num, 'Data_' + day_num)
    
    # Make file name.
    script_filename = os.path.splitext(name_of_running_script)[0]  # Name of current script
    
    # Create the log file. Note that the Log Browser API is bugged,
    # and adds a duplicate '.hdf5' file ending when using the database.
    if (not append_to_log_name_after_timestamp.startswith('_')) and (append_to_log_name_after_timestamp != ''):
        append_to_log_name_after_timestamp = '_' + append_to_log_name_after_timestamp
    if use_log_browser_database:
        savefile_string = script_filename + '_' + append_to_log_name_before_timestamp + '_' + timestamp + append_to_log_name_after_timestamp
    else:
        savefile_string = script_filename + '_' + append_to_log_name_before_timestamp + '_' + timestamp + append_to_log_name_after_timestamp + '.hdf5'
    print("... building HDF5 log file: " + savefile_string)
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
    print("... storing processed data into the HDF5 file.")
    for log_i in range(len(log_dict_list[:])):
        
        # ... and for every loop entry that is to be stored in this log:
        for outer_loop_i in range(outer_loop_size):
        
            # Add an entry in the log browser file, unless the calling
            # script wants a specific entry exported into a single file.
            if select_resonator_for_single_log_export == '':
                f.addEntry( {(log_dict_list[log_i])['name']: (processing_volume[log_i])[outer_loop_i, :]} )
            else:
                f.addEntry( {(log_dict_list[log_i])['name']: (processing_volume[int(select_resonator_for_single_log_export)])[outer_loop_i, :]} )

    
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
    