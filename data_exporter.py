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
import shutil
import numpy as np
from numpy import hanning as von_hann
from datetime import datetime
from data_discriminator import discriminate


def ensure_all_keyed_elements_even(hdf5_steps, hdf5_singles, hdf5_logs):
    ''' Assert that the received keys bear (an even number of) entries,
        implying whether a unit is missing for some entry.
    '''
    number_of_keyed_elements_is_even = \
        ((len(hdf5_steps) % 2) == 0) and \
        ((len(hdf5_singles) % 2) == 0) and \
        ((len(hdf5_logs) % 2) == 0)
    return number_of_keyed_elements_is_even

def stylise_axes(axes):
    ''' Stylistically rework underscored characters in the axes dict.
    '''
    for axis in ['x_name','x_unit','y_name','y_unit','z_name','z_unit']:
        axes[axis] = axes[axis].replace('/2','/₂')
        axes[axis] = axes[axis].replace('/3','/₃')
        axes[axis] = axes[axis].replace('_01','₀₁')
        axes[axis] = axes[axis].replace('_02','₀₂')
        axes[axis] = axes[axis].replace('_03','₀₃')
        axes[axis] = axes[axis].replace('_12','₁₂')
        axes[axis] = axes[axis].replace('_13','₁₃')
        axes[axis] = axes[axis].replace('_20','₂₀')
        axes[axis] = axes[axis].replace('_23','₂₃')
        axes[axis] = axes[axis].replace('_0','₀')
        axes[axis] = axes[axis].replace('_1','₁')
        axes[axis] = axes[axis].replace('_2','₂')
        axes[axis] = axes[axis].replace('_3','₃')
        axes[axis] = axes[axis].replace('lambda','λ')
        axes[axis] = axes[axis].replace('Lambda','Λ')
    return axes

def get_timestamp_string():
    ''' Return an appropriate timestamp string.
    '''
    return (datetime.now()).strftime("%d-%b-%Y_(%H_%M_%S)")

def get_dict_for_step_list(
    step_entry_name,
    step_entry_object,
    step_entry_unit = '',
    axes = [],
    axis_parameter = ''
    ):
    ''' Return a formatted dict for the step list entries.
    '''
    if axis_parameter.lower() == 'x':
        if (axes['x_name']).lower() != 'default':
            # Replace the x-axis name
            step_entry_name = axes['x_name']
        if axes['x_scaler'] != 1.0:
            # Re-scale the x-axis
            step_entry_object *= axes['x_scaler']
        if (axes['x_unit']).lower() != 'default':
            # Change the unit on the x-axis
            step_entry_unit = axes['x_unit']
    elif axis_parameter.lower() == 'z':
        if (axes['z_name']).lower() != 'default':
            # Replace the z-axis name
            step_entry_name = axes['z_name']
        if axes['z_scaler'] != 1.0:
            # Re-scale the z-axis
            step_entry_object *= axes['z_scaler']
        if (axes['z_unit']).lower() != 'default':
            # Change the unit on the z-axis
            step_entry_unit = axes['z_unit']
    return dict(name=step_entry_name, unit=step_entry_unit, values=step_entry_object)

def get_dict_for_log_list(
    log_entry_name,
    unit,
    axes,
    log_is_complex = True,
    ):
    ''' Return a formatted dict for the log entries.
    '''
    if (axes['y_unit']).lower() != 'default':
        unit = axes['y_unit']
    if (axes['y_name']).lower() != 'default':
        log_entry_name = axes['y_name']
    return dict(name=log_entry_name, unit=unit, vector=False, complex=log_is_complex)

def make_and_get_save_folder(
    filepath_of_calling_script
    ):
    ''' Figure out the full filepath where data should be saved,
        make the necessary directories needed (if needed). And,
        get the folder path of the script that was attempting the save,
        and thus the name of the running measurement.
    '''
    
    # Get name, file path and time for naming the folders that will contain
    # the exported log file.
    folder_path_to_calling_script_attempting_to_save, name_of_script_trying_to_save_data = os.path.split(filepath_of_calling_script)
    name_of_measurement_that_ran = os.path.splitext(name_of_script_trying_to_save_data)[0]
    year_num    = (datetime.now()).strftime("%Y")
    month_num   = (datetime.now()).strftime("%m")
    day_num     = (datetime.now()).strftime("%d")
    
    # Prepare path names needed for creating the folder tree.
    # First, find the root directory of whatever is executing the calling script.
    if not ('QPU interfaces' in folder_path_to_calling_script_attempting_to_save):
        raise OSError("The log browser export was called from a script not residing within a QPU interfaces folder. The save action was halted before finishing.")
    call_root = folder_path_to_calling_script_attempting_to_save.split('QPU interfaces',1)[0]
    data_folder_path = os.path.join(call_root, 'Data output folder')
    path1 = os.path.join(data_folder_path, year_num)
    path2 = os.path.join(data_folder_path, year_num, month_num)
    full_folder_path_where_data_will_be_saved = os.path.join(data_folder_path, year_num, month_num, 'Data_' + day_num)
    
    # Trickle through all subfolders in the created full folder path where
    # data will be saved, and make the missing directories where needed.
    ## (TODO: looking through all folders has a real name other than "trickling"... I have forgotten it as of writing...)
    for lb_path_name in [path1, path2, full_folder_path_where_data_will_be_saved]:
        if not os.path.exists(lb_path_name):
            os.makedirs(lb_path_name)
    
    return full_folder_path_where_data_will_be_saved, folder_path_to_calling_script_attempting_to_save, name_of_measurement_that_ran

def save(
    timestamp,
    ext_keys,
    log_dict_list,
    
    time_vector,
    fetched_data_arr,
    fetched_data_scale,
    fetched_data_offset,
    resonator_freq_if_arrays_to_fft,
    integration_window_start,
    integration_window_stop,
    
    filepath_of_calling_script,
    use_log_browser_database,
    
    inner_loop_size,
    outer_loop_size,
    
    single_shot_repeats_to_discretise = 0,
    ordered_resonator_ids_in_readout_data = [],
    get_probabilities_on_these_states = [],
    
    save_complex_data = True,
    source_code_of_executing_file = '',
    append_to_log_name_before_timestamp = '',
    append_to_log_name_after_timestamp  = '',
    select_resonator_for_single_log_export = '',
    force_matrix_reshape_flip_row_and_column = False,
    suppress_log_browser_export = False,
    
    log_browser_tag = 'krizan',
    log_browser_user = 'Christian Križan',
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
        
        Returns: save path to calling script (string)
    '''
    
    # Fetch the save folder for the data to be exported.
    full_folder_path_where_data_will_be_saved, \
    folder_path_to_calling_script_attempting_to_save, \
    name_of_measurement_that_ran = make_and_get_save_folder( filepath_of_calling_script )
    
    # Format incoming lists of chars (folder paths, typically) into strings.
    if isinstance(full_folder_path_where_data_will_be_saved, list):
        full_folder_path_where_data_will_be_saved = "".join(full_folder_path_where_data_will_be_saved)
    if isinstance(folder_path_to_calling_script_attempting_to_save, list):
        folder_path_to_calling_script_attempting_to_save = "".join(folder_path_to_calling_script_attempting_to_save)
    if isinstance(name_of_measurement_that_ran, list):
        name_of_measurement_that_ran = "".join(name_of_measurement_that_ran)
    
    # Get index corresponding to integration_window_start and
    # integration_window_stop respectively.
    integration_start_index = np.argmin(np.abs(time_vector - integration_window_start))
    integration_stop_index  = np.argmin(np.abs(time_vector - integration_window_stop ))
    integration_indices     = np.arange(integration_start_index, integration_stop_index)
    
    # Acquire the DFT sample frequencies contained within the
    # fetched_data_arr trace. freq_arr contains the centres of
    # the (representable) segments of the discretised frequency axis.
    dt = time_vector[1] - time_vector[0]
    num_samples = len(integration_indices)
    freq_arr = np.fft.fftfreq(num_samples, dt)  # Get DFT frequency "axis"
    
    # Get IF frequencies, so that we can pick out indices in the FFT array.
    # Did the user not send any IF information? Then assume IF = 0 Hz.
    # Did the user drive one resonator on resonance? Then set its IF to 0.
    if len(resonator_freq_if_arrays_to_fft) == 0:
        resonator_freq_if_arrays_to_fft.append(0)
    else:
        for pp in range(len(resonator_freq_if_arrays_to_fft)):
            if resonator_freq_if_arrays_to_fft[pp] == []:
                resonator_freq_if_arrays_to_fft[pp] = 0
    
    # Note! The user may have done a frequency sweep. In that case,
    # _ro_freq_if will be an array
    integration_indices_list = []
    for _ro_freq_if in resonator_freq_if_arrays_to_fft:
        if (not isinstance(_ro_freq_if, list)) and (not isinstance(_ro_freq_if, np.ndarray)):
            _curr_item = [_ro_freq_if] # Cast to list if not list.
        else:
            _curr_item = _ro_freq_if
        # The user may have swept the frequency => many IFs.
        curr_array = []
        for if_point in _curr_item:
            curr_array.append( np.argmin(np.abs(freq_arr - if_point)) )
        integration_indices_list.append( curr_array )
    
    # Execute complex FFT. Every row of the resp_fft matrix,
    # contains the FFT of every time trace that was ever collected
    # using .store() -- meaning that for instance resp_fft[0,:]
    # contains the FFT of the first store event in the first repeat.
    
    # If the user swept the IF frequency, then picking whatever
    # frequency in the FFT that is closest to the list of IF frequencies
    # will return may identical indices.
    # Ie. something like [124, 124, 124, 124, 124, 125, 125, 125, 125]
    # Instead, we should demodulate the collected data.
    '''resp_fft = np.fft.fft(fetched_data_arr[:, 0, integration_indices], axis=-1) / num_samples'''
    
    # Build a processed_data.
    processed_data = []
    for _item in integration_indices_list:
        if len(_item) <= 1:
            print("WARNING: The current FFT method is not demodulating sufficiently. If you are using large averages, you should also expect a weird offset on your Y-axis.") # TODO
            resp_fft = np.fft.fft(fetched_data_arr[:, 0, integration_indices], axis=-1) / num_samples
            processed_data.append( 2 * resp_fft[:, _item[0]] )
        else:
            print("WARNING: Currently, resonator frequency sweeps are not FFT'd due to a lack of demodulation. The Y-axis offset following your sweep is thus completely fictional.") # TODO
            print("WARNING: The current FFT method is not demodulating sufficiently. If you are using large averages, you should also expect a weird offset on your Y-axis.") # TODO
            return_arr = (np.mean(np.abs(fetched_data_arr[:, 0, integration_indices]), axis=-1) +fetched_data_offset[0])*fetched_data_scale[0]
            
            ## TODO: The commented part below is not finished. ##
            
            ################# IF frequency sweep (case) #################
            # Every new row in resp_fft corresponds to another ("the next")
            # instance of .store() in the entire sequencer program.
            # If the user swept some IF frequency during the sequence,
            # then the current _item will be a list, triggering this case.
            
            # Each new entry in _item will be the index (in the FFT
            # for that particular row in the grand resp_fft matrix)
            # that corresponds to the peak we want.
            
            """ This is actually a good one, except noise output (FFT resolution error perhaps?)
            ll = 0
            return_arr = []
            for fft_row in resp_fft[:]:
                return_arr.append( \
                    fft_row[ _item[ ll ] ] \
                )
                ll += 1"""
            
            """return_arr = []
            if len(_item[:]) == inner_loop_size:
                # The frequency sweep happened as an inner loop.
                for ll in range(inner_loop_size*outer_loop_size):
                    fft_row = np.fft.fft(fetched_data_arr[ll, 0, integration_indices] * 2*np.cos(2*np.pi*_item[ll % inner_loop_size]), axis=-1) / num_samples
                    return_arr.append( \
                        fft_row[ np.argmin(np.abs(freq_arr - _item[len(_item[:])//2] )) ] \
                    )
                    # TODO Remove the stuff to the right: #fft_row[ _item[ (ll % inner_loop_size) ] ]
            else:
                pass
                # The frequency sweep happened as an outer loop.
                ee = 0
                ff = 0
                for fft_row in resp_fft[:]:
                    return_arr.append( fft_row[ _item[ee] ] )
                    ff += 1
                    if ff == outer_loop_size:
                        ff = 0
                        ee += 1"""
                        
            # We have picked the appropriate fq.-swept indices. Return!
            processed_data.append( 2 * np.array(return_arr) )
    
    # Has the user set up the calling script so that the X and Z axes are
    # reversed? I.e. "the graph is rotated -90° in the Log Browser."
    if (len(ext_keys) > 1) and (inner_loop_size != outer_loop_size):
        first_dict  = ext_keys[0]
        second_dict = ext_keys[1]
        if (len(first_dict.get('values')) == outer_loop_size) and (len(second_dict.get('values')) == inner_loop_size) and (not force_matrix_reshape_flip_row_and_column):
            
            print("Detected external key reversal in the calling script."+\
            " Will flip axes "+first_dict.get('name')+" and "+\
            second_dict.get('name')+". Note! This message is not from "+\
            "the \"force_matrix_reshape_flip_row_and_column\" flag!")
            
            tempflip = inner_loop_size
            inner_loop_size = outer_loop_size
            outer_loop_size = tempflip
    
    # And, save either complex or magnitude data with/without some
    # scale and offset. Reshape the data to account for repeats.
    # Also, take into account whether the user is running a discretisation
    # measurements, meaning that every data point on the "z axis" of
    # fetched_data_arr is in fact a new 2D-plane of inner_loop+outer_loop data.
    if len(get_probabilities_on_these_states) > 0:
        # We are running a discretisation measurement.
        assert single_shot_repeats_to_discretise >= 1, "Error: a measurement is requesting state discrimination, but reports that its saved data is less than 1 shot long. (No. shots = "+str(single_shot_repeats_to_discretise)+")"
        
        # processed_data[mm] contains, on an per-resonator basis,
        # every 2D slice (inner_loop_size, outer_loop_size) big,
        # repeated for single_shot_repeats_to_discretise iterations.
        # But, all of this data is given on a single line in a vector.
        
        # First, discretise every entry in the in processed_data[:]
        discriminated_data = discriminate(
            data_or_filepath_to_data = processed_data,
            i_provided_a_filepath = False,
            ordered_resonator_ids_in_readout_data = ordered_resonator_ids_in_readout_data
        )
        
        # discriminated_data[resonator] now contains only discrete values.
        # We will reshape this into a 3D-volume with the following dimensions:
        #   Rows:     outer_loop_size
        #   Columns:  inner_loop_size
        #   Depth:    single_shot_repeats_to_discretise
        for aa in range(len(discriminated_data)):
            disc_data = np.array(discriminated_data[aa])
            disc_data.shape = \
                (single_shot_repeats_to_discretise, inner_loop_size * outer_loop_size)
            for hh in range(len(disc_data)):
                cut = np.array(disc_data[hh])
                cut.shape = (outer_loop_size, inner_loop_size)
                disc_data[hh] = cut
            discriminated_data[aa] = disc_data
        
        # We now want to look for probabilities of some user-provided states.
        # Remove duplicates from get_probabilities_on_these_states.
        # get_probabilities_on_these_states will get some random order.
        get_probabilities_on_these_states = list(set(get_probabilities_on_these_states))
        
        # Which is the highest represented state in the entire system?
        num_states = 0
        for item in discriminated_data:
            highest_num_in_slice = np.max(item)
            if highest_num_in_slice > num_states:
                num_states = highest_num_in_slice
        num_states = num_states +1 # Keep in mind that 0 counts (= |0>)
        
        # How many qubits are there?
        # (by assumption, the same as the number of resonators).
        num_qubits = len(discriminated_data)
        
        ## TODO: get_probabilities_on_these_states is a list of states,
        ##       not just one state. This fact would mean that the user
        ##       could write ['11','02'] and get two plots, matching the
        ##       probabilities for being in |11> and |02> respectively.
        ##for blabla goes here
        for curr_checked_state in get_probabilities_on_these_states:
            
            # Let's ensure that the user-provided state to investigate,
            # matches the same number of provided resonator IDs.
            assert len(curr_checked_state) == len(discriminated_data), "Error: the number of qubits in the user-provided state to discriminate, does not match the number of readout resonators in the state discriminated readout."
            
            ''' Here is an algorithm for representing states with unique
                integer identifiers.
                        
                discriminated_data[mm] contains integer values 0 ->
                (No. states that were discriminated between).
                
                For instance, if the state discrimination is between states
                |0>, |1>, |2>, for two resonators assuming one qubit each,
                then we may assign integer identifiers to every two-qubit
                state like this:
                
                0 = |00>
                1 = |01>
                2 = |02>
                3 = |10>
                4 = |11>
                5 = |12>
                6 = |20>
                7 = |21>
                8 = |22>
                
                ... so the integers involved are 0 through (no_qubits ** no_states).
                
                So, take discretised_data[  LEN-1  ]  ## WHICH IS THE LSB IN THE TABLE ABOVE,
                and perform:
                    processed_data[  LEN-1  ] * no_states**0
                  + processed_data[  LEN-2  ] * no_states**1
                  = some_integer
                
                Store some_integer in a new 3D-volume that has the same dimensions as one of any of all entries in discretised_data[mm].
                
                This new 3D-volume, which has the same dimensions as processed_data[any], will contain integers only.
                    
                Now, convert the user supplied state (like ['11']) to an integer
                as was done above. '11' becomes integer 4 in this case.
                
                FOR EVERY XY-pixel in (the new 3D-volume) along the length of every shot,
                    probability of being in 11 in this pixel = (number of 4 along this Z-line) / (length_of_the_z_line)
                
                Now once the probabilites are determined for every pixel,
                    processed_data[0] = the probabilities matrix
                        
                        Look at """ # Store the post-processed data. """ below. Make sure that when discretising, only processed_data[0] is saved anyway.
                            Keep in mind that there are measurements that request seperate resonator exports.
                                How though? Hm.
                
                Since inner_loop_size and outer_loop_size will match with the correct dimensions of the probabilities_matrix, then we're good.
                
            '''
            
            
            num_qubits
            num_states
            
            
            
            
            
            
            
        assert 1 == 0, "Not finished"
        
    else:
        for mm in range(len(processed_data[:])):
            fetch = processed_data[mm]
            fetch.shape = (outer_loop_size, inner_loop_size)
            
            if not save_complex_data:
                processed_data[mm] = (np.abs(fetch) +fetched_data_offset[mm])*(fetched_data_scale[mm])
            else:
                # The user might have set some scale and offset.
                # The offset would in that case have been set as
                # a portion of the magnitude.
                fetch_imag = np.copy(np.imag(fetch))
                fetch_real = np.copy(np.real(fetch))
                #fetch_thetas = np.arctan2( fetch_imag, fetch_real ) # Keep in mind the quadrants! np.arctan2 gives the same values as np.angle()
                fetch_thetas = np.copy(np.angle( fetch ))
                
                # Add user-set offset (Note: can be negative; "add minus y")
                fetch_imag += np.copy(fetched_data_offset[mm]) * np.sin( fetch_thetas )
                fetch_real += np.copy(fetched_data_offset[mm]) * np.cos( fetch_thetas )
                fetch = fetch_real + fetch_imag*1j
                
                # Scale with some user-set scale and store.
                processed_data[mm] = fetch * fetched_data_scale[mm]
        
    # Did the user request to flip the processed data?
    # I.e. that every new repeat in some measurement will be a column
    # instead of a row in the processed data? Then fix this.
    # Every data file is either way always stored so that
    # one row = one repeat.
    if force_matrix_reshape_flip_row_and_column:
        for ee in range(len(processed_data[:])):
            processed_data[ee] = (processed_data[ee]).transpose()
        
    # Perform data export to file
    filepath_to_exported_h5_file = export_processed_data_to_file(
        filepath_of_calling_script = filepath_of_calling_script,
        
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        time_vector = time_vector,
        processed_data = processed_data,
        fetched_data_arr = fetched_data_arr,
        fetched_data_scale = fetched_data_scale,
        fetched_data_offset = fetched_data_offset,
        
        timestamp = timestamp,
        log_browser_tag = log_browser_tag,
        log_browser_user = log_browser_user,
        append_to_log_name_before_timestamp = append_to_log_name_before_timestamp,
        append_to_log_name_after_timestamp = append_to_log_name_after_timestamp,
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export,
        select_resonator_for_single_log_export = select_resonator_for_single_log_export,
        resonator_freq_if_arrays_to_fft = resonator_freq_if_arrays_to_fft,
        source_code_of_executing_file = source_code_of_executing_file
    )
    
    # Return the .h5 save path to the calling script
    return filepath_to_exported_h5_file

def export_processed_data_to_file(
    filepath_of_calling_script,
    
    ext_keys,
    log_dict_list,
    
    time_vector,
    processed_data,
    fetched_data_arr,
    fetched_data_scale,
    fetched_data_offset,
    
    timestamp,
    log_browser_tag = 'krizan',
    log_browser_user = 'Christian Križan',
    append_to_log_name_before_timestamp = '',
    append_to_log_name_after_timestamp = '',
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    select_resonator_for_single_log_export = '',
    resonator_freq_if_arrays_to_fft = [],
    source_code_of_executing_file = '',
    ):
    ''' Take the supplied (processed) data, and export it to Labber's
        Log Browser (if possible) and as a .hdf5 file using H5PY.
    '''
    
    # Fetch the save folder for the data to be exported.
    full_folder_path_where_data_will_be_saved, \
    folder_path_to_calling_script_attempting_to_save, \
    name_of_measurement_that_ran = make_and_get_save_folder( filepath_of_calling_script )
    
    # Format incoming lists of chars (folder paths, typically) into strings.
    if isinstance(full_folder_path_where_data_will_be_saved, list):
        full_folder_path_where_data_will_be_saved = "".join(full_folder_path_where_data_will_be_saved)
    if isinstance(folder_path_to_calling_script_attempting_to_save, list):
        folder_path_to_calling_script_attempting_to_save = "".join(folder_path_to_calling_script_attempting_to_save)
    if isinstance(name_of_measurement_that_ran, list):
        name_of_measurement_that_ran = "".join(name_of_measurement_that_ran)
    
    # Touch up on user-input strings in the calling script.
    if (not append_to_log_name_after_timestamp.startswith('_')) and (append_to_log_name_after_timestamp != ''):
        append_to_log_name_after_timestamp = '_' + append_to_log_name_after_timestamp
    if (not append_to_log_name_before_timestamp.startswith('_')) and (append_to_log_name_before_timestamp != ''):
        append_to_log_name_before_timestamp = '_' + append_to_log_name_before_timestamp
    if (not timestamp.startswith('_')) and (timestamp != ''):
        timestamp = '_' + timestamp
    
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
            savefile_string = name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp
        else:
            savefile_string = name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + '.hdf5'
        
        if not suppress_log_browser_export:# TODO see todo below, tl;dr: fix so that the user must not have Labber to post-process the H5PY.
            print("... assembling Log Browser-compatible .HDF5 log file: " + savefile_string)
            f = Labber.createLogFile_ForData(
                savefile_string,
                log_dict_list,
                step_channels = ext_keys,
                use_database  = use_log_browser_database
            )
            
            # Set project name, tag, and user in logfile.
            f.setProject(name_of_measurement_that_ran)
            f.setTags(log_browser_tag)
            f.setUser(log_browser_user)

            # Store the post-processed data.
            print("... storing processed data into the .HDF5 file.")
            # TODO: This part should cover an arbitrary number of fetched_data_arr
            #       arrays. And, this entire subroutine should be made fully
            #       generic.
            if (select_resonator_for_single_log_export == ''):
                if (len(resonator_freq_if_arrays_to_fft) > 1):
                    # Then store multiplexed!
                    # For every loop entry that is to be stored in this log:
                    for loop_i in range(len( (processed_data[0])[:] )):
                        f.addEntry({
                            (log_dict_list[0])['name']: (processed_data[0])[loop_i, :],
                            (log_dict_list[1])['name']: (processed_data[1])[loop_i, :]
                        })
                else:
                    for loop_i in range(len( (processed_data[0])[:] )):
                        f.addEntry({
                            (log_dict_list[0])['name']: (processed_data[0])[loop_i, :]
                        })
            else:
                # TODO This else-case must be removed.
                for log_i in range(len(log_dict_list[:])):
                    for loop_i in range(len( (processed_data[0])[:] )):
                        f.addEntry({
                            (log_dict_list[log_i])['name']: (processed_data[int(select_resonator_for_single_log_export)])[loop_i, :]
                        })
            
            # Check if the hdf5 file was created in the local directory.
            # This would happen if you change use_data to False in the
            # Labber.createLogFile_ForData call. If so, move it to an appropriate
            # directory. Make directories where necessary.
            success_message = " in the Log Browser directory!"
            save_path = os.path.join(full_folder_path_where_data_will_be_saved, savefile_string)  # Full save path
            if os.path.isfile(os.path.join(folder_path_to_calling_script_attempting_to_save, savefile_string)):
                shutil.move( os.path.join(folder_path_to_calling_script_attempting_to_save, savefile_string) , save_path)
                success_message = ", see " + save_path
            
            # Print success message.
            print("Data saved" + success_message)
    
    # Whether or not the Labber Log Browser export worked,
    # we still want to save the data in the H5PY format.
    # Since, the data can then be sent to automated processes without
    # relying on Labber.
    
    ####################################
    ''' SAVE AS H5PY-COMPATIBLE HDF5 '''
    ####################################
    
    # Make a name for the H5PY savefile-string, and get its save folder path.
    # Depending on the state of the Log Browser export, this string may change.
    savefile_string_h5py = name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + '.h5'
    save_path_h5py = os.path.join(full_folder_path_where_data_will_be_saved, savefile_string_h5py)
    
    # Create a H5PY-styled hdf5 file.
    with h5py.File(save_path_h5py, 'w') as h5f:
        if source_code_of_executing_file != '':
            datatype = h5py.string_dtype(encoding='utf-8')
            dataset  = h5f.create_dataset("saved_source_code", (len(source_code_of_executing_file), ), datatype)
            for kk, sourcecode_line in enumerate(source_code_of_executing_file):
                dataset[kk] = sourcecode_line
        for ff in range(len(ext_keys)):
            if (np.array((ext_keys[ff])['values'])).shape == (1,):
                h5f.attrs[(ext_keys[ff])['name']] = (ext_keys[ff])['values']
            else:
                h5f.create_dataset( (ext_keys[ff])['name'] , data = (ext_keys[ff])['values'] )
        
        h5f.create_dataset("time_vector",  data = time_vector)
        h5f.create_dataset("fetched_data_arr", data = fetched_data_arr)
        h5f.create_dataset("processed_data", data = processed_data)
        h5f.create_dataset("User_set_scale_to_Y_axis",  data = fetched_data_scale)
        h5f.create_dataset("User_set_offset_to_Y_axis", data = fetched_data_offset)
        
        print("Data saved using H5PY, see " + save_path_h5py)
    
    # Return the .h5 save path to the calling function
    return save_path_h5py
    