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
from data_discriminator import discriminate
from time_calculator import get_timestamp_string
from datetime import datetime # Needed for making the save folders.
from random import randint

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

def get_dict_for_step_list(
    step_entry_name,
    step_entry_object,
    step_entry_unit = '',
    axes = [],
    axis_parameter = ''
    ):
    ''' Return a formatted dict for the step list entries.
    '''
    # Ensure that the user is not accidentally inserting a NoneType object.
    assert step_entry_object[0] != None, \
        "Error! Attempted to format object "+str(step_entry_name)            +\
        ", but this object has a NoneType object type, "                     +\
        "which cannot be formatted. Likely, the user accidentally sent "     +\
        "a None as an function argument, and the Presto API itself did not " +\
        "detect the error. Check your arguments."
    
    # Make a suitable dict.
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
    full_folder_path_where_data_will_be_saved = os.path.join(data_folder_path, year_num, month_num, 'Data_' + month_num + day_num)
    
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
    data_to_store_consists_of_time_traces_only = False,
    source_code_of_executing_file = '',
    default_exported_log_file_name = 'default',
    append_to_log_name_before_timestamp = '',
    append_to_log_name_after_timestamp  = '',
    select_resonator_for_single_log_export = '',
    force_matrix_reshape_flip_row_and_column = False,
    suppress_log_browser_export = False,
    save_raw_time_data = False,
    
    log_browser_tag  = 'default',
    log_browser_user = 'default',
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
        
        If default_exported_log_file_name is left to 'default',
        then the files will take the name of the script which performed the
        measurement. The whole string "default" is merely replaced, meaning
        that something like arg. = 'qubit_1' + 'default', is a legal call,
        and will generate "qubit_1_blablabla" as an output filename.
        
        Returns: save path to calling script (string)
    '''
    
    # Print status to the user.
    print("Post-processing and saving data, please hold.")
    
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
    
    # Is the user only interested in time traces?
    if not data_to_store_consists_of_time_traces_only:
        
        # Send data to post-processor
        processed_data = post_process_time_trace_data(
            time_vector = time_vector,
            integration_window_start = integration_window_start,
            integration_window_stop = integration_window_stop,
            resonator_freq_if_arrays_to_fft = resonator_freq_if_arrays_to_fft,
            fetched_data_arr = fetched_data_arr,
            fetched_data_offset = fetched_data_offset,
            fetched_data_scale = fetched_data_scale,
        )
        
    else:
        # The user is only interested in time trace data.
        processed_data = []
        processed_data.append( fetched_data_arr[:, 0, :] )
    
    # Clear out fetched_data_arr if it's not needed anymore, to save on memory.
    if not save_raw_time_data:
        del fetched_data_arr
        fetched_data_arr = []
    
    # Has the user set up the calling script so that the X and Z axes are
    # reversed? I.e. "the graph is rotated -90° in the Log Browser."
    if (not force_matrix_reshape_flip_row_and_column):
        if (len(ext_keys) > 1) and (inner_loop_size != outer_loop_size):
            first_dict  = ext_keys[0]
            second_dict = ext_keys[1]
            if (len(first_dict.get('values')) == outer_loop_size):
                if (len(second_dict.get('values')) == inner_loop_size):
                    ##if (len(first_dict.get('values')) == outer_loop_size) and (len(second_dict.get('values')) == inner_loop_size) and (not force_matrix_reshape_flip_row_and_column):
                    
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
    if (len(get_probabilities_on_these_states) > 0) and (not data_to_store_consists_of_time_traces_only):
        # We are running a discretisation measurement.
        assert single_shot_repeats_to_discretise >= 1, "Error: a measurement is requesting state discrimination, but reports that its saved data is less than 1 shot long. (No. shots = "+str(single_shot_repeats_to_discretise)+")"
        
        # processed_data[mm] contains, on an per-resonator basis,
        # every 2D slice (inner_loop_size, outer_loop_size) big,
        # repeated for single_shot_repeats_to_discretise iterations.
        # But, all of this data is given on a single line in a vector.
        
        # First, discretise every entry in the in processed_data[:]
        discriminated_data, num_states = discriminate(
            states_that_will_be_checked = get_probabilities_on_these_states,
            data_or_filepath_to_data = processed_data,
            i_provided_a_filepath = False,
            ordered_resonator_ids_in_readout_data = ordered_resonator_ids_in_readout_data
        )
        
        ## TODO DEBUG
        ## For whatever dumbfuck reason, running discriminated_data
        ## changes processed_data so that it matches discriminated_data.
        ## Should not be a problem I think. processed_data gets properly
        ## blanked further down the code.
        
        # discriminated_data[resonator] now contains only discrete values.
        # We will reshape this into a 3D-volume with the following dimensions:
        #   Rows:     outer_loop_size
        #   Columns:  inner_loop_size
        #   Depth:    single_shot_repeats_to_discretise
        #
        #   ... once we have the probabilities counted!
        #
        for aa in range(len(discriminated_data)):
            disc_data = np.array(discriminated_data[aa])
            disc_data.shape = \
                (single_shot_repeats_to_discretise, inner_loop_size * outer_loop_size)
            # Let's not reshape every entry in discriminated_data to match
            # the (outer_loop_size, inner_loop_size) format just yet.
            # This enables using np.bincount later for probabilities.
            ##for hh in range(len(disc_data)):
            ##    cut = np.array(disc_data[hh])
            ##    cut.shape = (outer_loop_size, inner_loop_size)
            ##    disc_data[hh] = cut
            discriminated_data[aa] = disc_data
        
        # We now want to look for probabilities of some user-provided states.
        # Remove duplicates from get_probabilities_on_these_states.
        ## TODO Feature removed for now, since list(set( )) randomises everything.
        
        # Remake the discriminated data into an integer format.
        ''' Here is my way of representing n-qubit states with unique integers.
                    
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
            
            The possible integers are 0 through (no_qubits ** no_states).
            
            So, take discretised_data[  LEN-1  ]  ## THE LSB IN THE LUT ABOVE!
            and perform (for example, for a 4-qubit state):
                processed_data[  LEN-1  ] * no_states**0
              + processed_data[  LEN-2  ] * no_states**1
              + processed_data[  LEN-3  ] * no_states**2
              + processed_data[  LEN-4  ] * no_states**3
              = some_integer showing a unique 4-qubit state.
            
            Now, convert the user supplied state (like ['11']) to an integer
            as was done above. '11' becomes integer 4 (see LUT above) if
            we have a 2-qubit state with 3 possible states.
            
        '''
        integer_rep_matrix = np.zeros_like(discriminated_data[0])
        for ll in range(len(discriminated_data)):    
            integer_rep_matrix += np.array(discriminated_data[len(discriminated_data)-1-ll]) * (num_states**ll)
        
        # integer_rep_matrix is now the total discriminated system matrix,
        # where every integer corresponds to the n-qubit state held there.
        
        probability_vectors = [[]] * len(get_probabilities_on_these_states)
        for urr in range(len(get_probabilities_on_these_states)):
            curr_checked_state = get_probabilities_on_these_states[urr]
            
            # Let's ensure that the user-provided state to investigate,
            # matches the same number of provided resonator IDs.
            assert len(curr_checked_state) == len(discriminated_data), "Error: the number of qubits in the user-provided state to discriminate, does not match the number of readout resonators in the state discriminated readout."
            
            # Convert the sought-for state to an integer, see ~14 rows up how.
            int_rep = 0
            for ll in range(len(curr_checked_state)):
                int_rep += int(curr_checked_state[len(curr_checked_state)-1-ll]) * (num_states**ll)
            
            # NOTE! That we have yet not reshaped every entry in 
            # neither discriminated_data or integer_rep_matrix.
            # Every row in the matrix, corresponds merely to
            # the whole 2D-matrix (outer_loop_size * inner_loop_size)
            # unravelled to a single row. The final reshaping will happen soon.
            
            # int_rep is an n-qubit state that the user seeks a probability of.
            # Let's calculate how often int_rep (aka. curr_checked_state)
            # shows up *for every pixel* in integer_rep_matrix.
            ''' DO NOTE that specifying dtype='float64' is CRUCIAL.
                Otherwise, prob_vector will remain [0 0 0 0 0 0] always.'''
            prob_vector = np.zeros_like(integer_rep_matrix[0], dtype='float64')
            for curr_rowcol in range(len(integer_rep_matrix[0,:])):
                bix = np.bincount(integer_rep_matrix[:,curr_rowcol])
                try:
                    no_sought_vals_found = bix[int_rep]
                except IndexError:
                    # If this triggers, then there were no user-sought
                    # n-qubit states for any shot for this row+column value.
                    no_sought_vals_found = 0
                prob_vector[curr_rowcol] = no_sought_vals_found / single_shot_repeats_to_discretise
            
            # For the current state that the user is interested in,
            # fill in the resulting probabilities.
            probability_vectors[urr] = prob_vector
        
        # Now, we only have to reshape the data right.
        processed_data = [[]] * len(probability_vectors)
        for pv in range(len(probability_vectors)):
            fetch = probability_vectors[pv]
            fetch.shape = (outer_loop_size, inner_loop_size)
            processed_data[pv] = fetch
        
        # Error-check that the probability calculation is correct.
        # No "pixel" should have more than 100% probability.
        # TODO: This segment can probably be removed some time in the future.
        curr_pix_total = 0
        for all_rows in range(len(processed_data[0])):
            for all_cols in range(len((processed_data[0])[0])):
                for all_states_checked in range(len(processed_data)):
                    curr_pix_total += (processed_data[all_states_checked])[all_rows][all_cols]
                # Check whether this pixel has more than 100% probability.
                # The check should be valid to at least 14+ decimal places.
                assert round(curr_pix_total, 14) <= 1.0, \
                    "\nHalted! Pixel ("+str(all_rows)+","+str(str(all_cols)) +\
                    ") returned more than 100% probability for its state"    +\
                    " distribution. \n\nDebug information:\n"                +\
                    "Pixel total was: " + str(curr_pix_total)+"\n\nInteger " +\
                    "matrix:\n"+str(integer_rep_matrix)+"\n\nProbability-"   +\
                    "vectors matrix:\n"+str(probability_vectors)
                curr_pix_total = 0.0
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
        resonator_freq_if_arrays_to_fft = resonator_freq_if_arrays_to_fft,
        
        timestamp = timestamp,
        log_browser_tag = log_browser_tag,
        log_browser_user = log_browser_user,
        default_exported_log_file_name = default_exported_log_file_name,
        append_to_log_name_before_timestamp = append_to_log_name_before_timestamp,
        append_to_log_name_after_timestamp = append_to_log_name_after_timestamp,
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export,
        select_resonator_for_single_log_export = select_resonator_for_single_log_export,
        source_code_of_executing_file = source_code_of_executing_file,
        save_raw_time_data = save_raw_time_data
    )
    
    # Return the .h5 save path to the calling script
    return filepath_to_exported_h5_file

def post_process_time_trace_data(
    time_vector,
    integration_window_start,
    integration_window_stop,
    resonator_freq_if_arrays_to_fft,
    fetched_data_arr,
    fetched_data_offset,
    fetched_data_scale,
    ):
    ''' Using the provided arguments,
        1. Demodulate the time trace data held within fetched_data_arr.
        2. Perform FFT on segment held within the integration window bounds.
        3. Return the element at 0 Hz baseband as the (complex) datapoint(s)
           of choice.
    '''
    
    # Prepare list, that will contain the processed data.
    # Every new index of processed data, is another resonator.
    processed_data = []
    
    # Get index corresponding to integration_window_start and
    # integration_window_stop respectively.
    integration_start_index = np.argmin(np.abs(time_vector - integration_window_start))
    integration_stop_index  = np.argmin(np.abs(time_vector - integration_window_stop ))
    
    # Make a list of integers, where each integer corresponds to
    # an index along the integration window, that will be post-processed.
    integration_indices     = np.arange(integration_start_index, integration_stop_index)
    
    # Acquire parameters needed for FFT-ing the data.
    num_samples = len(integration_indices)
    ##dt = time_vector[1] - time_vector[0]
    ##freq_arr = np.fft.fftfreq(num_samples, dt)  # Get DFT frequency "axis"
    
    # Did the user not send any IF information? Then assume IF = 0 Hz.
    # Did the user drive one resonator on resonance? Then set its IF to 0.
    ## Note: all IF entries are converted into lists later on, anyhow.
    if len(resonator_freq_if_arrays_to_fft) == 0:
        print("Warning! No IF information provided to FFT. Assuming IF = 0 Hz.")
        # Append entry corresponding to this resonator.
        # Let's cast it to list already, since that would happen later
        # if we don't.
        resonator_freq_if_arrays_to_fft.append( [0] )
    else:
        # The user has provided some kind of IF data.
        for pp in range(len(resonator_freq_if_arrays_to_fft)):
            # Have the user supplied empty sweeps?
            if resonator_freq_if_arrays_to_fft[pp] == []:
                print("Warning! No IF sweep information provided to FFT at entry "+str(pp)+". Assuming IF = 0 Hz for this entry.")
                # Append entry corresponding to this resonator.
                # Let's cast it to list already, since that would happen later
                # if we don't.
                resonator_freq_if_arrays_to_fft[pp] = [0]
    
    # Instead of getting indices of some freq_arr, let's instead demodulate the
    # data, so that our sought-for frequency content is at baseband 0 Hz!
    for arr_ii in range(len(resonator_freq_if_arrays_to_fft)):
        
        # Grab the next resonator's IF frequency (array)!
        _ro_freq_if = resonator_freq_if_arrays_to_fft[arr_ii]
        
        # Parse the IF input, cast to lists if entries are non-list.
        if (not isinstance(_ro_freq_if, list)) and (not isinstance(_ro_freq_if, np.ndarray)):
            _ro_freq_if = [_ro_freq_if]
        
        # Print progress?
        if len(_ro_freq_if) > 1:
            print("Resonator " + \
                str(arr_ii+1) + " of " + \
                str(len(resonator_freq_if_arrays_to_fft))+":")
        
        # Get time t that corresponds to the integration indices.
        # This t is used later when doing the demodulation.
        ## How should the time vector look like for the demodulation
        ## carrier? Let's assume that the sampling window
        ## always starts at 0.0 seconds. Because, this is likely
        ## where the Presto instrument itself assumes that t = 0 is.
        ## This way, I am trying to keep the demodulation carrier
        ## in phase with the Presto's first demodulation carrier.
        t = np.linspace( \
            time_vector[integration_indices[0]], \
            time_vector[integration_indices[-1]], \
            len(integration_indices) \
        )
        
        # _ro_freq_if may either be single-valued, or list-valued.
        # List-valued entries correspond to IF sweeps, typically
        # resonator spectroscopy sweeps.
        # For such sweeps, we need to know the total number of measurement
        # points that will be stored.
        if (len(_ro_freq_if) > 1):
            total_points_to_store = ((fetched_data_arr[:, 0, integration_indices]).shape)[0]
            processed_data_sweep_arr = np.zeros(total_points_to_store, dtype=np.complex128)
            ## TODO: right now, it is assumed that all IF sweeps has the frequency
            ## sweep done using the repeat argument (= outer loop).
            ## I am currently unsure as to how this fact can be resolved in a good way.
            inner_loop_size = int(total_points_to_store/len(_ro_freq_if))
        else:
            # This array is checked later to determine whether there
            # was an IF sweep.
            processed_data_sweep_arr = []
        for arr_jj in range(len(_ro_freq_if)):
            
            # Which is the next IF to demodulate with?
            _current_if_to_demodulate = _ro_freq_if[arr_jj]
            
            # Print progress?
            if (len(_ro_freq_if) > 1) and ((arr_jj % 11) == 0):
                print("Progress: "+str(round(arr_jj/len(_ro_freq_if)*100,1))+" %")
            
            # _current_if_to_demodulate is either a single item (single IF)
            # or all items in a for-looped list of IFs.
            # Single-valued entries in _ro_freq_if results in just one FFT.
            
            # Fetch data to work on
            if len(_ro_freq_if) == 1:
                arr_to_fft = fetched_data_arr[:, 0, integration_indices]
            else:
                # Then this is an IF-sweep, and we may select smaller
                # chunks for FFT'ing.
                ## TODO: Right now, it is assumed that all IF sweeps
                ## has the frequency sweep in the repeat (outer loop) of
                ## the measurement. I am unsure of a good way to fix this fact.
                arr_to_fft = fetched_data_arr[inner_loop_size*arr_jj:(inner_loop_size*arr_jj)+inner_loop_size, 0, integration_indices]
            
            ## arr_to_fft is an object shaped (num_datapoints_in_measurement, len(integration_indices))
            ## So, that would be all (instrument-averaged) time traces,
            ## for every datapoint in the measurement plot.
            ##
            ## 2D-sweeps are at this point irrelevant. All values are
            ## merely provided on a single line; reshaping into 2D data
            ## happens later.
            ##
            ## Example: in a 50x4 sweep for some measurement with 50 points
            ##          swept over four values on some other variable,
            ##          then the arr_to_fft would be shaped
            ##          (200, len(integration_indices))
            
            # Demodulate!
            for ii in range(len(arr_to_fft)):
                ## FOR OLD FILES, WITH MEASUREMENT SCRIPTS WHERE THE IF    ##
                ## FREQUENCY WAS ALWAYS POSITIVE REGARDLESS OF LSB OR USB, ##
                ## YOU'LL HAVE TO SUPPLY A FORCED -f_demod NEGATION HERE:  ##
                #arr_to_fft[ii] = arr_to_fft[ii] * np.exp(+2j * np.pi * -_current_if_to_demodulate * t)
                arr_to_fft[ii] = arr_to_fft[ii] * np.exp(+2j * np.pi * _current_if_to_demodulate * t)
            
            # Perform FFT!
            resp_fft = np.fft.fft(arr_to_fft, axis=-1)
            
            # Is _ro_freq_if single-valued?
            #   => There is one IF for all datapoints.
            if len(_ro_freq_if) == 1:
                processed_data.append( 2/(num_samples) * resp_fft[:, 0] ) # Append result to processed_data.
            else:
                # We're looking at a frequency sweep with many IFs.
                # Only grab the datapoints currently valid for the
                # current IF being demodulated with.
                
                # Remember: in 2D sweeps, the final output is still just a
                # straight line of data where all (second dimension)-iterations
                # have been stringed onto eachother.
                # Example: you do a 3x6 pixel sweep; 6 traces of 3 datapoints.
                # The data that was sent to this function, then consisted of
                # [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3].
                # Thus, for IF sweeps, we need to select multiple points
                # to store for each IF. For the very first IF, we need to
                # grab all of the "1" datapoints in that FFT'd vector. Etc.
                # At this point, we must thus figure out the spacing between
                # every "1" in the sequence.
                ## TODO since there was a large change to keep bias sweeps
                ## on the *inner* loop of the measurements, this means
                ## that the datapoints are structured:
                ## [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
                ## The solution to this fact is very hacky as of now,
                ## since it assumes that all IF sweeps performs said IF
                ## sweep on the repeat (aka. outer) loop of the measurements.
                ## TODO: I am unsure of how to best solve this issue.
                '''for uu in range(0, total_points_to_store, len(_ro_freq_if)):
                    print("Selecting point: "+str(arr_jj + uu))
                    processed_data_sweep_arr[arr_jj + uu] = 2/(num_samples) * resp_fft[arr_jj + uu, 0]'''
                for uu in range(0, inner_loop_size):
                    ##processed_data_sweep_arr[arr_jj*inner_loop_size + uu] = 2/(num_samples) * resp_fft[arr_jj*inner_loop_size + uu, 0]
                    processed_data_sweep_arr[arr_jj*inner_loop_size + uu] = 2/(num_samples) * resp_fft[uu, 0]
            
            # At this point, we may free up some memory.
            del resp_fft
        
        # Did we process an IF frequency sweep?
        if processed_data_sweep_arr != []:
            # We did.
            processed_data.append( processed_data_sweep_arr )
            del processed_data_sweep_arr
        ## else:
        ##    Otherwise, we don't have to care, the FFT was done all in one go.
    
    # Return the post-processed data.
    return processed_data


def export_processed_data_to_file(
    filepath_of_calling_script,
    
    ext_keys,
    log_dict_list,
    
    processed_data,
    fetched_data_scale,
    fetched_data_offset,
    
    timestamp = 0,
    time_vector = [],
    fetched_data_arr = [],
    log_browser_tag  = 'default',
    log_browser_user = 'default',
    resonator_freq_if_arrays_to_fft = [],
    default_exported_log_file_name = 'default',
    append_to_log_name_before_timestamp = '',
    append_to_log_name_after_timestamp = '',
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    select_resonator_for_single_log_export = '',
    source_code_of_executing_file = '',
    save_raw_time_data = False
    ):
    ''' Take the supplied (processed) data, and export it to Labber's
        Log Browser (if possible) and as a .hdf5 file using H5PY.
    '''
    
    # Check if the timestamp is valid.
    if timestamp == 0:
        # Invalid. Make valid.
        timestamp = get_timestamp_string()
    
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
    
    # Touch up on user-input strings in the calling script. And, filename.
    add_user_string_to_the_filename_beginning = ''
    add_user_string_to_the_filename_end       = ''
    if default_exported_log_file_name != 'default':
        # The user wants something.
        if ('default' in default_exported_log_file_name):
            # The user still wants the default string appended.
            if default_exported_log_file_name.find('default') == 0:
                # User added something to the end.
                add_user_string_to_the_filename_end = default_exported_log_file_name.replace('default','')
                if add_user_string_to_the_filename_end[0] != '_':
                    add_user_string_to_the_filename_end = '_' + add_user_string_to_the_filename_end
            elif default_exported_log_file_name.find('default') >= 1:
                # User added something to the beginning.
                add_user_string_to_the_filename_beginning = default_exported_log_file_name.replace('default','')
                if add_user_string_to_the_filename_beginning[-1] != '_':
                    add_user_string_to_the_filename_beginning = add_user_string_to_the_filename_beginning + '_'
            else:
                # User added something somewhere.
                splitresult = default_exported_log_file_name.split('default')
                add_user_string_to_the_filename_beginning = splitresult[0]
                add_user_string_to_the_filename_end       = splitresult[1]
                del splitresult
        else:
            # The default substring is not in the requested string.
            # Then, rename everything.
            name_of_measurement_that_ran = str(default_exported_log_file_name)
            append_to_log_name_before_timestamp = ''
            append_to_log_name_after_timestamp  = ''
            timestamp = ''
    
    if (not append_to_log_name_after_timestamp.startswith('_')) and (append_to_log_name_after_timestamp != ''):
        append_to_log_name_after_timestamp = '_' + append_to_log_name_after_timestamp
    if (not append_to_log_name_before_timestamp.startswith('_')) and (append_to_log_name_before_timestamp != ''):
        append_to_log_name_before_timestamp = '_' + append_to_log_name_before_timestamp
    if (not timestamp.startswith('_')) and (timestamp != ''):
        timestamp = '_' + timestamp
    
    # Just to be sure, check whether there is an overwrite risk imminent.
    # Such an overwrite has in fact happened, typically when stitching
    # just a few files together from some command line script.
    safe_from_overwrite = False
    while (not safe_from_overwrite):
        h5py_string = add_user_string_to_the_filename_beginning + name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + add_user_string_to_the_filename_end + '.h5'
        log_browser_string = add_user_string_to_the_filename_beginning + name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + add_user_string_to_the_filename_end + '.hdf5'
        if ((os.path.exists(h5py_string)) or (os.path.exists(log_browser_string))):
            # The file exists! Let's change the timestamp and try again.
            timestamp = '_' + get_timestamp_string()
            print("WARNING! Filepath-induced overwrite detected, changing timestamp string. The new file name, without file ending, is:\n"+ add_user_string_to_the_filename_beginning + name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + add_user_string_to_the_filename_end)
        else:
            # The file does not exist. We're good to go.
            safe_from_overwrite = True
            del h5py_string
            del log_browser_string
    del safe_from_overwrite
    
    # Attempt an export to Labber's Log Browser!
    labber_import_worked = False
    try:
        import Labber
        labber_import_worked = True
    except ModuleNotFoundError:
        print("Could not import the Labber library; "                      + \
              "no data was saved in the Log Browser compatible format. "   + \
              "\n\nTo remedy this error, you should now do the following:" + \
              "\nFirst, make sure that the Labber Script folder is visible"+ \
              " to your Python environment. If you are running things via "+ \
              "Anaconda, then you have two options (one proper, one easy)."+ \
              " The proper way is to run >> conda-develop "                + \
              "C:\\Program Files\\Labber\\Script to make the Labber API "  + \
              "visible to your Python environment. If you are unfamiliar " + \
              "with conda-develop, then you must absolutely make sure "    + \
              "that you know where to find the conda.pth file in your "    + \
              "environment's folder (since people never tend to get the "  + \
              "command syntax right when using conda-develop).\n\n"        + \
              "The other, non-proper but easier method, is to skip"        + \
              " conda-develop. Instead, write sys.path.append(\"C:"        + \
              "\\Program Files\\Labber\\Script\") in the beginning of the "+ \
              "Python script that you are using to run your things.\n\n"   + \
              "If this problem persists, then check whether your Python "  + \
              "version is supported by the Labber API. This fact can be "  + \
              "seen in the Labber/Script folder:\nHead to the Script "     + \
              "folder, and make sure that there is a subfolder somewhere " + \
              "which matches your Python version. For instance, a folder " + \
              "named \"py39\" if you are using Python 3.9. If you do not " + \
              "see a folder whose number matches your Python version, then"+ \
              " your solution is to downgrade your Python version to a "   + \
              "version which is supported by your Labber API. For instance"+ \
              ", if you saw a folder named py37, then you may use Python " + \
              "3.7. Downgrading within Anaconda typically takes ages, the" + \
              " non-slow solution is to just uninstall Anaconda, and then "+ \
              "install an old version of Anaconda that uses the Python "   + \
              "version that you are after. Reinstalling all \"lost\" pip " + \
              "packages, is typically still much faster than downgrading " + \
              "Python.\n\nNote: the problem that you are facing has "      + \
              "nothing to do with the bitness of your programs, x86 or 64 "+ \
              "does not matter at all.")
    if labber_import_worked:
        # Create the log file. Note that the Log Browser API is bugged,
        # and adds a duplicate '.hdf5' file ending when using the database.
        if use_log_browser_database:
            savefile_string = add_user_string_to_the_filename_beginning + name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + add_user_string_to_the_filename_end
        else:
            savefile_string = add_user_string_to_the_filename_beginning + name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + add_user_string_to_the_filename_end + '.hdf5'
        
        if not suppress_log_browser_export:
            print("... assembling Log Browser-compatible .HDF5 log file: " + savefile_string)
            f = Labber.createLogFile_ForData(
                savefile_string,
                log_dict_list,
                step_channels = ext_keys,
                use_database  = use_log_browser_database
            )
            
            # Set project name, tag, and user in logfile.
            # Check whether the user wishes to set a custom tag and/or user
            # string in the exported file. If not, use 'Christian Križan' as
            # the user, 'krizan' as the tag.
            if log_browser_tag == 'default':
                log_browser_tag  = 'krizan'
            if log_browser_user == 'default':
                log_browser_user = 'Christian Križan'
            
            f.setProject(name_of_measurement_that_ran)
            f.setTags(log_browser_tag)
            f.setUser(log_browser_user)

            # Store the post-processed data.
            print("... storing processed data into the .HDF5 file.")
            # TODO:  This part should cover an arbitrary number of fetched_data_arr
            #        arrays. And, this entire subroutine should be made fully
            #        generic.
            if (select_resonator_for_single_log_export == ''):
                
                # Ensure that log_dict_list and processed_data matches.
                assert len(log_dict_list) == len(processed_data), "Error: the requested logs to store do not match the available amount of data to store. Likely, the log_dict_list in the calling script is erroneous."
                
                # The Log Browser API has sort-of strange expectations
                # as to how data is saved, as will be seen in the loops below. 
                
                # We can run a dict-update loop to construct the target dict
                # that we want to add to the log. But, we must know
                # beforehand what data we want to store into the dict keys.
                # And, only run f.addEntry once per storage of such data.
                # I.e. "one outer_loop" row typically of the measurement.
                
                # Here, it makes more sense to look at processed_data rather
                # than the length of the log_dict_list. Because, the latter
                # is set by the user, and the former is supposedly always
                # set by the algorithm of this very file you are looking at.
                dict_to_add_into_log = dict()
                for loop_i in range(len( (processed_data[0])[:] )):
                    for qq in range(len(processed_data)):
                        dict_to_add_into_log.update({
                            (log_dict_list[qq])['name']: (processed_data[qq])[loop_i, :]
                        })
                    f.addEntry(dict_to_add_into_log) # This line stores data.
                    dict_to_add_into_log.clear()
            else:
                # TODO1: This else-case must be removed.
                # TODO2: It can likely be removed if only looking at how long
                #        processed_data is.
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
    savefile_string_h5py = add_user_string_to_the_filename_beginning + name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + add_user_string_to_the_filename_end + '.h5'
    save_path_h5py = os.path.join(full_folder_path_where_data_will_be_saved, savefile_string_h5py)
    
    # Make a check whether there is a single-point log value that should not
    # be saved as an attribute.
    single_entry_logs = []
    for qq in range(len(log_dict_list)):
        single_entry_logs.append( (log_dict_list[qq])['name'] )
    
    # Create a H5PY-styled hdf5 file.
    with h5py.File(save_path_h5py, 'w') as h5f:
        if source_code_of_executing_file != '':
            datatype = h5py.string_dtype(encoding='utf-8')
            dataset  = h5f.create_dataset("saved_source_code", (len(source_code_of_executing_file), ), datatype)
            for kk, sourcecode_line in enumerate(source_code_of_executing_file):
                dataset[kk] = sourcecode_line
        for ff in range(len(ext_keys)):
            ##if ((np.array((ext_keys[ff])['values'])).shape == (1,)) and ( not ((ext_keys[ff])['name'] in single_entry_logs)):
            if (np.array((ext_keys[ff])['values'])).shape == (1,):
                h5f.attrs[(ext_keys[ff])['name']] = (ext_keys[ff])['values']
            else:
                h5f.create_dataset( (ext_keys[ff])['name'] , data = (ext_keys[ff])['values'] )
        
        if len(time_vector) != 0:
            h5f.create_dataset("time_vector",  data = time_vector)
        if (len(fetched_data_arr) != 0) and (save_raw_time_data):
            h5f.create_dataset("fetched_data_arr", data = fetched_data_arr)
        if len(resonator_freq_if_arrays_to_fft) != 0:
            h5f.create_dataset("resonator_freq_if_arrays_to_fft", data = resonator_freq_if_arrays_to_fft)
        h5f.create_dataset("processed_data", data = processed_data)
        h5f.create_dataset("User_set_scale_to_Y_axis",  data = fetched_data_scale)
        h5f.create_dataset("User_set_offset_to_Y_axis", data = fetched_data_offset)
        
        # h5f content for export data file stitching!
        # ext_keys will contain a lot of numpy arrays, which are not
        # JSON-compatible. These must be converted to the Python list datatype.
        h5f.attrs["ext_keys"] = json.dumps(convert_numpy_entries_in_ext_keys_to_list(ext_keys), indent = 4)
        h5f.attrs["log_dict_list"] = json.dumps(log_dict_list, indent = 4) 
        h5f.create_dataset("filepath_of_calling_script", data = filepath_of_calling_script)
        h5f.create_dataset("First_key_that_was_swept", data = (ext_keys[0])['name']) # Denote which value was the first one to be swept.
        h5f.create_dataset("Second_key_that_was_swept", data = (ext_keys[1])['name']) # Denote which value was the second one to be swept.
        
        # Report done.
        print( \
            "Data saved using H5PY at "          + \
            get_timestamp_string(pretty = True)  + \
            ", see " + save_path_h5py \
        )
    
    # Return the .h5 save path to the calling function
    return save_path_h5py

def export_raw_data_to_new_file(
    path_to_saved_file_containing_raw_data,
    use_log_browser_database,
    save_raw_time_data,
    ):
    ''' Cracks open a saved datafile, containing raw data,
        and exports the raw data into a new save file.
        
        Typically, this function can be used to test post-processing
        features.
    '''
    # Does the provided file exist?
    if not os.path.exists( path_to_saved_file_containing_raw_data ):
        raise OSError("Error! The user-provided filepath does not exist.")
    
    # Attempt to open the user-provided file, and extract all raw data.
    dict_containing_all_keys = {}
    dict_containing_all_attributes = {}
    resonator_freq_if_arrays_to_fft = []
    inner_loop_size = 0
    outer_loop_size = 0
    save_complex_data = True
    ext_keys = None
    log_dict_list = None
    fetched_data_arr = None
    with h5py.File(path_to_saved_file_containing_raw_data, 'r') as h5f:
        
        # Get all keys in the hdf5 file.
        keys_in_file = h5f.keys()
        for entry in keys_in_file:
            if (entry == "First_key_that_was_swept") or (entry == "First_key_that_was_swept"):
                dict_containing_all_keys.update({str(entry): (h5f[str(entry)][()]).decode("utf-8")})
            else:
                dict_containing_all_keys.update({str(entry): h5f[str(entry)][()]})
        
        # Get all attributes in the hdf5 file.
        attributes_in_file = h5f.attrs.keys()
        for entry in attributes_in_file:
            # ext_keys and log_dict_list needs special treatment later.
            if (entry != "ext_keys") and (entry != "log_dict_list"):
                dict_containing_all_attributes.update({str(entry): h5f.attrs[str(entry)]})
        
        # Extract ext_keys and log_dict_list for using the Labber Log browser.
        ext_keys = json.loads( h5f.attrs["ext_keys"] )
        log_dict_list = json.loads( h5f.attrs["log_dict_list"] )
        
        # Does the file contain IF information for the resonators.
        if 'resonator_freq_if_arrays_to_fft' in dict_containing_all_keys:
            # There is IF information!
            resonator_freq_if_arrays_to_fft = dict_containing_all_keys["resonator_freq_if_arrays_to_fft"]
        else:
            # There is no information on readout if in the file.
            # The value(s) has to be calculated.
            resonator_nco = dict_containing_all_attributes.get("readout_freq_nco")
            
            # Figure out how many resonators were present.
            for item in dict_containing_all_attributes:
                if  (item.startswith('readout_freq')) and \
                    (not (item.startswith('readout_freq_nco'))) and \
                    (not (item.startswith('readout_freq_span'))):
                    # This item is a candidate. Span or not?
                    # Check whether the end of that candidate, has the same
                    # ending as key words like _centre and _span.
                    # Example: readout_freq_A, and readout_freq_centre_A
                    if item.startswith('readout_freq_centre'):
                        # It's a span!
                        
                        ## Optimisation: assume that every _centre item,
                        ## has a corresponding _span item.
                        ## For instance, assume that there will always be
                        ## a readout_freq_centre_B, and a readout_freq_span_B.
                        
                        # Get the particular suffix to the word (like "_A")
                        # that identifies this frequency with some other span.
                        item_suffix = item.replace('readout_freq_centre','')
                        
                        # Files that contain *only* the parameter
                        # readout_freq_centre and readout_freq_span,
                        # will be caught too, since the suffix is ""
                        readout_freq_nco    = dict_containing_all_attributes['readout_freq_nco'+item_suffix][0]
                        readout_freq_centre = dict_containing_all_attributes['readout_freq_centre'+item_suffix][0]
                        readout_freq_span   = dict_containing_all_attributes['readout_freq_span'+item_suffix][0]
                        num_freqs           = dict_containing_all_attributes['num_freqs'][0]
                        ## TODO, instead of assuming num_freqs, one could look
                        ##       at the length of the readout array.
                        
                        readout_freq_centre_if = readout_freq_nco - readout_freq_centre
                        f_start = readout_freq_centre_if - readout_freq_span / 2
                        f_stop  = readout_freq_centre_if + readout_freq_span / 2
                        readout_freq_if_arr = np.linspace(f_start, f_stop, num_freqs)
                        
                        # Append the calculated readout frequency span.
                        ##resonator_freq_if_arrays_to_fft.append( np.abs(readout_freq_if_arr) )
                        print("Note: an instance of np.abs() was removed here, back when the readout post-processing code was upgraded. If your resonator IFs look weird, you may want to look into the code where this print statement is located.")
                        resonator_freq_if_arrays_to_fft.append( readout_freq_if_arr )
                        
                        del f_start, f_stop, readout_freq_if_arr
                        del readout_freq_nco, readout_freq_centre
                        del readout_freq_centre_if, item_suffix, num_freqs
                        
                    else:
                        # It's not a span. Append a fixed-point readout IF.
                        ## Is there a special entry for 'readout_freq_excited'?
                        if 'readout_freq_excited' in dict_containing_all_attributes:
                            readout_freq = dict_containing_all_attributes['readout_freq_excited']
                            assert (not ('readout_freq' in dict_containing_all_attributes)), "Halted! Could not figure out which readout frequency to use when calculating IF frequencies of a data file that lacked readout IF information."
                        else:
                            readout_freq = dict_containing_all_attributes['readout_freq']
                        
                        readout_freq_nco = dict_containing_all_attributes['readout_freq_nco']
                        readout_freq_if  = readout_freq_nco - readout_freq
                        resonator_freq_if_arrays_to_fft.append( np.abs(readout_freq_if) )
                        
                        del readout_freq_nco, readout_freq_if, readout_freq
            
            assert resonator_freq_if_arrays_to_fft != [], "Halted! The file did not contain information about IF frequencies, and such information was not calculatable from what was found within the file."
            print("The file did not contain information about IF frequencies used for the resonators. This value(s) was calculated to be: "+str(resonator_freq_if_arrays_to_fft))
        
        # Get the loop size parameters by looking at the processed data.
        # The first index corresponds to the resonator. Let's assume that
        # the resonators all have an equal-sized canvas.
        shape_to_analyse = np.array(dict_containing_all_keys["processed_data"]).shape
        outer_loop_size  = shape_to_analyse[1]
        inner_loop_size  = shape_to_analyse[2]
        del shape_to_analyse
        
        # Save complex data?
        save_complex_data = log_dict_list[0]['complex']
    
    # Attempt to pull the raw data out of the file, report if failure.
    fetched_data_fetch_successful = False
    try:
        fetched_data_arr = dict_containing_all_keys["fetched_data_arr"]
        fetched_data_fetch_successful = True
    except KeyError:
        # Failure, do nothing.
        pass
    if (not fetched_data_fetch_successful):
        raise KeyError("Error! Could not pull raw data from file: \""+str(path_to_saved_file_containing_raw_data)+"\" - check whether your measurement ran with the flag \"save_raw_time_data = True\"")
    del fetched_data_fetch_successful # Clean up.
    
    # Export extracted data to file.
    save(
        timestamp = get_timestamp_string(), # Timestamps are not saved. Get a new one.
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        time_vector = dict_containing_all_keys["time_vector"],
        fetched_data_arr = fetched_data_arr,
        fetched_data_scale = dict_containing_all_keys["User_set_scale_to_Y_axis"],
        fetched_data_offset = dict_containing_all_keys["User_set_offset_to_Y_axis"],
        resonator_freq_if_arrays_to_fft = resonator_freq_if_arrays_to_fft,
        integration_window_start = dict_containing_all_attributes["integration_window_start"],
        integration_window_stop  = dict_containing_all_attributes["integration_window_stop"],
        
        filepath_of_calling_script = dict_containing_all_keys["filepath_of_calling_script"].decode('UTF-8'),
        use_log_browser_database = use_log_browser_database,
        
        inner_loop_size = inner_loop_size,
        outer_loop_size = outer_loop_size,
        
        save_complex_data = save_complex_data,
        default_exported_log_file_name = 'ENTREPÔT' + 'default', # Append a prefix.
        
        ## TODO there are many more things that have not been
        ##      added here yet.
        ## TODO add support for state-discriminated measurements.
    )
    

def convert_numpy_entries_in_ext_keys_to_list( ext_keys_to_convert ):
    ''' Takes the ext_keys dict, and converts all numpy
        entries in the list to JSON-compatible Python lists.
        This subroutine enables storing ext_keys in h5py-compatible .h5 files.
        Storing ext_keys removes a large number of problems when
        stitching data files.
    '''
    # ext_keys is a list of dicts.
    for ce in range(len(ext_keys_to_convert)):
        current_entry = (ext_keys_to_convert[ce]).copy()
        if type(current_entry['values']) == list:
            # Then do nothing.
            pass
        else:
            current_entry['values'] = current_entry['values'].tolist()
            ext_keys_to_convert[ce] = current_entry.copy()
        del current_entry
    
    # Note the copy()-dance. Now, return the result.
    return (ext_keys_to_convert.copy())


def stitch(
    h5_files_to_stitch_as_list_or_folder,
    merge_if_x_and_z_axes_are_identical = False, # Halt to ensure there is no overwrite because of poor user arguments.
    use_this_scale  = None,
    use_this_offset = None,
    log_browser_tag  = 'default',
    log_browser_user = 'default',
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    select_resonator_for_single_log_export = '',
    delete_old_files_after_stitching = False,
    default_exported_log_file_name = 'default',
    ):
    ''' A function that stiches together exported data files.
        
        For all .h5 files in the .h5 file list argument,
        grab all data and stitch together one export.
        
        Finally, delete all old files if so requested.
    '''
    
    # Considering the usage cases that have happened, the following inputs
    # will be parsed: strings, lists of strings, lists of files, lists
    # of folders with files, lists of files and folders with files.
    if isinstance(h5_files_to_stitch_as_list_or_folder, str):
        # The user provided a string, rework it into a list.
        h5_files_to_stitch_as_list_or_folder = [h5_files_to_stitch_as_list_or_folder]
        
    elif isinstance(h5_files_to_stitch_as_list_or_folder, list):
        # The user provided a list.
        # Was it a list of chars?
        if len(h5_files_to_stitch_as_list_or_folder[0]) == 1:
            # Let's check.
            it_was_not_a_list_of_chars = False
            for item in h5_files_to_stitch_as_list_or_folder:
                if not ((len(item) == 1) and (isinstance(item, str))):
                    it_was_not_a_list_of_chars = True
            if not it_was_not_a_list_of_chars:
                # The user provided a list of chars, this fact is indicative of
                # an error in using the os package. Print warning.
                print(  "WARNING: the data stitcher was provided a list of "+\
                        "chars as input. This fact is indicative of a "+\
                        "coding error, there is likely some function that "+\
                        "does not parse filepaths correctly as they are "+\
                        "returned as results from other functions. For "+\
                        "instance, a measurement returning the path of "+\
                        "its final results file.")
                h5_files_to_stitch_as_list_or_folder = [''.join(h5_files_to_stitch_as_list_or_folder)]
            
            # Clean up.
            del it_was_not_a_list_of_chars
    else:
        # Unexpected input.
        raise TypeError( \
            "Error! Expected argument inputs of type string or "+\
            "list, but got input of type "+\
            str(type(h5_files_to_stitch_as_list_or_folder)))
    
    # At this point, the input is parsable. Let's build a list
    # of items to stitch.
    list_of_h5_files_to_stitch = []
    for item in h5_files_to_stitch_as_list_or_folder:
        # Is this item a filepath, or a directory of files to add?
        if os.path.isdir( item ):
            # The user provided a directory.
            # Append all .hdf5 files (.h5) in this directory.
            for file_item in os.listdir( item ):
                if (file_item.endswith('.h5')) or (file_item.endswith('.hdf5')):
                    print("Found file: \""+str(file_item)+"\"")
                    list_of_h5_files_to_stitch.append(file_item)
        
        elif os.path.isfile( item ):
            # It's a file. Add if .hdf5 file (.h5)
            if (item.endswith('.h5')) or (item.endswith('.hdf5')):
                print("Found file: \""+str(item)+"\"")
                list_of_h5_files_to_stitch.append(item)
        
        else:
            # The entry is not a directory, nor a file.
            print("WARNING: cannot understand provided path \""+str(item)+"\" → skipping.")
    
    # Let's check that the stitcher has something to work with.
    if len(list_of_h5_files_to_stitch) == 0:
        raise RuntimeError( \
            "Error! The data stitcher could not "+\
            "find any HDF5 data to work with. Check your arguments.")
    
    # At this point, we've acquired data to work with.
    
    # There may be different scales and offsets in the provided files.
    # We'll keep a running tab to make sure these scales and offsets do
    # not differ.
    scales_used_in_the_files = []
    offset_used_in_the_files = []
    for fff in range(len(list_of_h5_files_to_stitch)):
        
        # Grab the next file item.
        file_item = list_of_h5_files_to_stitch[fff]
        with h5py.File( file_item, 'r') as h5f:
            
            # How many resonators were read?
            res_read = (h5f["processed_data"][()].shape)[0]
            
            # Is the user overriding which scale to use?
            if use_this_scale != None:
                
                # The user is overriding the scaling to be used.
                # Is the user-supplied scaling usable?
                improper_argument = False
                try:
                    if len(use_this_scale[fff]) == res_read:
                        scales_used_in_the_files.append( use_this_scale[fff] )
                    else:
                        improper_argument = True
                except IndexError:
                    improper_argument = True
                if improper_argument:
                    raise TypeError("Error! Invalid scaling provided as input argument. Expected a list of "+str(len(list_of_h5_files_to_stitch))+" scale arrays, where each array was supposed to be a numpy array of length "+str(res_read)+", which corresponds to "+str(res_read)+" resonator(s).")
                del improper_argument
            
            else:
                # The user is not overriding which scaling to use!
                
                ## Check for missing scales.
                try:
                    scales_used_in_the_files.append( \
                        h5f["User_set_scale_to_Y_axis"][()] \
                    )
                except KeyError:
                    if randint(1,100) == 100:
                        # There is no scale, Gromit.
                        print("There is no scale, Gromit.")
                    
                    # Report that assumed offsets will be appended.
                    to_report = "["
                    to_assume = []
                    for tt in range(res_read):
                        
                        # Append assumption.
                        to_assume.append(np.array( [1.0] ))
                        
                        # Build something to report.
                        to_report += "1.0"
                        if tt != res_read-1:
                            # There are more "1.0" entries to add.
                            to_report += ", "
                        else:
                            # We're done.
                            to_report += "]"
                    
                    # Print warning.
                    print("WARNING: the file "+str(file_item)+" did not contain any information about the measurement's Y-axis scaling. Assuming " + to_report)
                    
                    # Append list of assumptions.
                    scales_used_in_the_files.append( to_assume )
                    
                    # Clean up.
                    del to_report
                    del to_assume
                    del res_read
            
            # Is the user overriding which offset to use?
            if use_this_offset != None:
                
                # The user is overriding the offset to be used.
                # Is the user-supplied offset usable?
                improper_argument = False
                try:
                    if len(use_this_offset[fff]) == res_read:
                        offset_used_in_the_files.append( use_this_offset[fff] )
                    else:
                        improper_argument = True
                except IndexError:
                    improper_argument = True
                if improper_argument:
                    raise TypeError("Error! Invalid offset provided as input argument. Expected a list of "+str(len(list_of_h5_files_to_stitch))+" offset arrays, where each array was supposed to be a numpy array of length "+str(res_read)+", which corresponds to "+str(res_read)+" resonator(s).")
                del improper_argument
            
            else:
                # The user is not overriding which offset to use!
                
                ## Check for missing offsets.
                try:
                    offset_used_in_the_files.append( \
                        h5f["User_set_offset_to_Y_axis"][()] \
                    )
                except KeyError:
                    if randint(1,100) == 100:
                        # There is no offset, Gromit.
                        print("There is no offset, Gromit.")
                    
                    # How many resonators were read?
                    res_read = (h5f["processed_data"][()].shape)[0]
                    
                    # Report that assumed offsets will be appended.
                    to_report = "["
                    to_assume = []
                    for tt in range(res_read):
                        
                        # Append assumption.
                        to_assume.append(np.array( [0.0] ))
                        
                        # Build something to report.
                        to_report += "0.0"
                        if tt != res_read-1:
                            # There are more "0.0" entries to add.
                            to_report += ", "
                        else:
                            # We're done.
                            to_report += "]"
                    
                    # Print warning.
                    print("WARNING: the file "+str(file_item)+" did not contain any information about the measurement's Y-axis offset. Assuming " + to_report)
                    
                    # Append list of assumptions.
                    offset_used_in_the_files.append( to_assume )
                    
                    # Clean up.
                    del to_report
                    del to_assume
                    del res_read
    
    # There may be time traces stored in the files.
    # We'll keep track so that no data files have data collected at
    # different time trace settings. Likewise, it's also possible that
    # the user is attempting to stitch together a file
    # that has no time trace information, with a file that does. In that
    # case, let's catch this, and halt.
    running_vector_of_time = []
    time_vectors_are_expected_in_the_data = False
    ## TODO: Somehow, stitch things together even if the data is
    ##       collected at non-identical times. TODO NOTE: currently disabled.
    
    # Where are we supposed to output the stitched data?
    # Let's create a blank variable for now, that will be filled-in
    # once it's time to check for a suitable filepath.
    filepath_of_calling_script = []
    
    # Identify the axes that were swept.
    first_swept_key_found  = False
    second_swept_key_found = False
    first_key  = []
    second_key = []
    
    # Prepare lists that will hold the measurement file axes' data,
    # as well as the measurement data held at these positions.
    list_of_swept_keys = []
    curr_swept_content = []
    # Do the same thing again, but with the stationary items
    # that are not swept.
    list_of_unswept_keys = []
    unswept_content      = []
    
    # Prepare to extract and hog all of the processed data fed
    # into the stitcher. Remember that processed_data feeds back information
    # from every resonator. The first entry of processed_data corresponds
    # to the first resonator (or first probability log), the second one to the
    # second resonator (or second probability log) and so on.
    big_fat_list_of_canvases = []
    big_fat_list_of_x_axes   = []
    big_fat_list_of_z_axes   = []
    big_fat_list_length_has_been_established = False
    
    # Prepare ext_keys and log_dict_list, these variables will be
    # dictified later.
    ext_keys = []
    log_dict_list = []
    resonator_freq_if_arrays_to_fft = []
    
    # Treat every provided item in the list!
    for current_filepath_item_in_list in list_of_h5_files_to_stitch:
        
        # Get data.
        with h5py.File(current_filepath_item_in_list, 'r') as h5f:
            
            # Notify the user.
            print("Extracting data: "+str(current_filepath_item_in_list))
            
            # Make sure that the time traces are compatible in the
            # provided data files.
            try:
                vector_of_time = h5f["time_vector"][()]
                # This worked, likely. Let's set the flag
                # "we expect there to be time traces" = True.
                time_vectors_are_expected_in_the_data = True
                
                # Do we know how the proper time trace for this stitching
                # should look like? If not, then set a reference.
                if running_vector_of_time == []:
                    running_vector_of_time = vector_of_time
                else:
                    if not (vector_of_time == running_vector_of_time).all():
                        raise NotImplementedError( \
                            "File \""+current_filepath_item_in_list+\
                            "\" contains data, collected with a time trace "+ \
                            "differing from every other time trace in the " + \
                            "other to-be-stitched files. Currently, the "   + \
                            "data stitcher does not support stitching data "+ \
                            "collected using different time traces. In "    + \
                            "practice, time traces will be identical if "   + \
                            "the sampling rate and sampling window length " + \
                            "are identical between measurements.")
            except KeyError:
                # There is no time information in this file.
                # But was there supposed to be any time information?
                if time_vectors_are_expected_in_the_data:
                    # Then the stitching is incompatible, at least as of now.
                    raise NotImplementedError( \
                        "Error! The file "+current_filepath_item_in_list    + \
                        " contained no legible time trace information. But "+ \
                        "previous file(s) did. Stitching files together "   + \
                        "with and without time trace information is "       + \
                        "currently not supported.")
            
            # At this point in the code, the time trace information (if any)
            # in the current file, matches the expected time trace information
            # based on all other files processed this far.
            
            # Where should we put the export once finished?
            # If this is not known by now, then let's get a filepath
            # and file name for realising said file export.
            if filepath_of_calling_script == []:
                try:
                    filepath_of_calling_script = os.path.abspath((h5f["filepath_of_calling_script"][()]).decode('utf-8'))
                    # Are you currently working from a different
                    # computer than that which made this data?
                    if not os.path.exists(filepath_of_calling_script):
                        # You are. Then get a better filepath.
                        filepath_of_calling_script = os.path.realpath(__file__)
                except KeyError:
                    # The first file-to-be-stitched contained insufficient
                    # information for figuring out where to store the
                    # stitched-together export. Let's just grab
                    # the path of the file which called the stitcher.
                    filepath_of_calling_script = os.path.realpath(__file__)
            
            # Our safety checks have all passed this far.
            
            # Let's check whether the file contains information about
            # which keys corresponds to the first (and second) one(s) swept
            # in the measurement.
            try:
                # Do we know which is the "first key" that we want to look for?
                if not first_swept_key_found:
                    # We don't. Try to get it. Update the list of swept keys.
                    first_key = str((h5f["First_key_that_was_swept"][()]).decode("utf-8"))
                    # Just to make sure we don't mess up the variable
                    # list_of_swept_keys, let's double-check that it doesn't
                    # already contain information about a second_key value.
                    if not second_swept_key_found:
                        # Good.
                        list_of_swept_keys.append(first_key)
                    else:
                        # Oh shit. Well, make sure that the first key
                        # is still the first value anyway.
                        list_of_swept_keys.insert(0, first_key)
                        # Check that the second key is indeed the second value.
                        assert list_of_swept_keys[1] == second_key, \
                            "Error! Irrecoverable discrepancy detected among first and second swept key entries in the provided measurement files. Some file did probably not provide a first_key entry, while providing a second_key entry."
                    first_swept_key_found = True
            except KeyError:
                # This file contains no information regarding which was the
                # first key to be swept in the measurement.
                ## But was it supposed to?
                if first_swept_key_found:
                    # Discrepancy detected!
                    raise KeyError( \
                    "Error! The data stitcher expected the first swept " + \
                    "axis of file \""+str(current_filepath_item_in_list) + \
                    "\" to be \""+first_key+"\", but this file did not " + \
                    "report that it contains data for this expected axis.")
                # Fine, the file was not expected to contain
                # an entry for the first swept key. But it still has none.
                raise KeyError( \
                    "Error! The file \""+str(current_filepath_item_in_list) + \
                    "\" contains no information regarding which key was "   + \
                    "the first one to be swept in the measurement. It is "  + \
                    "not possible to establish which axes are to be "       + \
                    "assigned to the stitched file, without resorting to "  + \
                    "guesswork.")
            try:
                # Do we know which is the "second key" that we want to look for?
                if not second_swept_key_found:
                    # We don't. Try to get it.
                    catch_key = h5f["Second_key_that_was_swept"][()]
                    # Catch whether there are some UTF-8 shenanigans going on.
                    if not isinstance(catch_key, str):
                        second_key = str(catch_key.decode("utf-8"))
                    else:
                        # This seems not to be the case.
                        second_key = catch_key
                    del catch_key
                    second_swept_key_found = True
            except KeyError:
                # This file contains no information regarding which was the
                # second key to be swept in the measurement. That could
                # very well be fine. Let's just confirm that there
                # was no expected second swept key in the measurement.
                if second_swept_key_found:
                    # Discrepancy detected!
                    raise KeyError( \
                    "Error! The data stitcher expected the second swept " + \
                    "axis of file \""+str(current_filepath_item_in_list) + \
                    "\" to be \""+second_key+"\", but this file did not " + \
                    "report that it contains data for this expected axis.")
            
            # By now, we know which are the swept axes of the measurement file.
            # And in fact, we know which are the swept axes of all files,
            # provided no upcoming file is misconfigured.
            
            # Let's get all other, swept, values of this file.
            if ext_keys == []:
                ext_keys = json.loads( h5f.attrs["ext_keys"] )
            if log_dict_list == []:
                log_dict_list = json.loads( h5f.attrs["log_dict_list"] )
            
            # Let's get information about what resonator IFs were used in the
            # measurement.
            # TODO This fetching needs to be looked at further.
            #      What if the user is stitching together several files
            #      using different readout IFs?
            if resonator_freq_if_arrays_to_fft == []:
                try:
                    resonator_freq_if_arrays_to_fft = h5f["resonator_freq_if_arrays_to_fft"][()]
                except KeyError:
                    print("Warning! Could not extract readout resonator IF information from the supplied file. This parameter was set to \"[]\"")
            
            # By now, we are ready to grab data held within this file.
            curr_processed_data = h5f["processed_data"][()]
            
            # Have we established how big the big_fat_list_of_canvases will be?
            # It will have the same number of entries as there are resonators
            # read out between the different files.
            if not big_fat_list_length_has_been_established:
                # This has not been established. Then, declare
                # the big_fat_list_of_canvases to be the same
                # length as the number of resonators (or probabilities)
                big_fat_list_of_canvases = [ [] for _ in range(4   )]##len(curr_processed_data)) ]
                big_fat_list_of_x_axes   = [ [] for _ in range(len(curr_processed_data)) ]
                big_fat_list_of_z_axes   = [ [] for _ in range(len(curr_processed_data)) ]
                big_fat_list_length_has_been_established = True
                
                ## The datastructure is the following:
                ## Imagine a 4-resonator readout, and 3 measurements
                ## to be stitched together.
                ## big_fat_list_of_canvases is thus:
                ##       **Res1**  **Res2**  **Res3**  **Res4**
                ##    [  [  a1,    [  b1,    [  c1,    [  d1,  
                ##          a2,       b2,       c2,       d2,  
                ##          a3  ]     b3  ]     c3  ]     d3  ]  ]
                ##
                ##  ... where a, b, c, are all 2D canvases of data.
                ##      Specifically, one ROW of big_fat_list, corresponds
                ##      to the first measurement file.
                ##      Whereas one COLUMN corresponds to a resonator or
                ##      a scoped probability.
                ##      Likewise, the x and z axes big_fat_lists,
                ##      contain the x or z axis that corresponds to
                ##      the 2D canvas contained at that "coordinate."
            
            # The final data will be stored in a (curr_)"canvas" with
            # some 2D dimension corresponding to the data collected at
            # resonator n in the currently treated file.
            # If there are 7 resonators, there will be 7 canvases held within
            # the currently processed file.
            for entry_idx in range(len(curr_processed_data)):
                # Grab a new canvas (for this resonator) to be added into
                # the big fat list. curr_canvas WILL be a 2D object,
                # although this object might have only one row of data.
                # Repeat the same procedure for getting the X and Z axes.
                curr_canvas = curr_processed_data[entry_idx]
                big_fat_list_of_canvases[entry_idx].append(curr_canvas)
                big_fat_list_of_x_axes[entry_idx].append( h5f[first_key][()] )
                ## It is possible that the 2D-sweep has merely a single
                ## swept dimension. In that case, let's catch this, and
                ## simply ignore the big_fat_list_of_z_axes.
                try:
                    big_fat_list_of_z_axes[entry_idx].append( h5f[second_key][()] )
                except KeyError:
                    # If faced with an "object doesn't exist" error,
                    # we'll assume that the sweep is one-dimensional.
                    # The second_key has been stored in a different way in the
                    # .hdf5 file.
                    big_fat_list_of_z_axes[entry_idx].append( h5f.attrs[second_key] )
            
            # At this point, big_fat_list_of_canvases[res] is the combined
            # data of every measurement file provided. Every row in the
            # big_fat_lists corresponds to one measurement file. Every
            # column correspond to a particular resonator or probability.
    
    # At this point, we have stored all data that there is to stitch together
    # in a massive variable. We may now assemble all data together.
    print("Data extracted successfully from all files!\n")
    
    ## Let's detect at this point whether we like to paint a big canvas,
    ## or whether every axis is identical, meaning that we'd like to
    ## merge files together.
    if merge_if_x_and_z_axes_are_identical:
        # If this input argument flag is set, then typically
        # the user expects that all axes are identical,
        # and that they merely wish to merge one-dimensional sweeps
        # together. First, we must check that the Z-axis is in fact just
        # a one-value thing.
        expected_x_axis = []
        expected_z_axis = []
        expected_x_axis_is_defined = False
        expected_z_axis_is_defined = False
        single_x_axis_for_this_resonator = [ [] for _ in range(len(big_fat_list_of_x_axes)) ]
        single_z_axis_for_this_resonator = [ [] for _ in range(len(big_fat_list_of_z_axes)) ]
        for current_resonator in range(len(big_fat_list_of_canvases)):
            # Grab expected X-axis if undefined.
            if not expected_x_axis_is_defined:
                expected_x_axis = (big_fat_list_of_x_axes[current_resonator])[0]
            for every_x_axis_in_this_resonator_entry in (big_fat_list_of_x_axes[current_resonator]):
                is_every_x_value_the_same_value = True
                for obj in range(len(expected_x_axis)):
                    if(expected_x_axis[obj] != every_x_axis_in_this_resonator_entry[obj]):
                        is_every_x_value_the_same_value = False
                assert is_every_x_value_the_same_value, \
                    "Error! The user requested a set of files to be merged, stating (by argument) that they would all share the same X-axis. But all axes are not identical, and thus the files cannot be merged this way."
                del is_every_x_value_the_same_value
            # If we reached this point, then the X-axis is fine for this
            # resonator entry.
            single_x_axis_for_this_resonator[current_resonator] = expected_x_axis
            
            # Grab expected Z-axis if undefined.
            if not expected_z_axis_is_defined:
                expected_z_axis = (big_fat_list_of_z_axes[current_resonator])[0]
            for every_z_axis_in_this_resonator_entry in (big_fat_list_of_z_axes[current_resonator]):
                is_every_z_value_the_same_value = True
                for obj in range(len(expected_z_axis)):
                    if(expected_z_axis[obj] != every_z_axis_in_this_resonator_entry[obj]):
                        is_every_z_value_the_same_value = False
                assert is_every_z_value_the_same_value, \
                    "Error! The user requested a set of files to be merged, stating (by argument) that they would all share the same X-axis. But all axes are not identical, and thus the files cannot be merged this way."
                del is_every_z_value_the_same_value
            # If we reached this point, then the Z-axis is fine for this
            # resonator entry.
            single_z_axis_for_this_resonator[current_resonator] = expected_z_axis
        
        ## One more check: for this merge method to work,
        ## we expect the Z-axis to be just a single value for
        ## all measurements.
        assert len(expected_z_axis) == 1, \
            "Error! All axes are identical in the supplied measurement files"+\
            ",which is OK for the requested data stiching method. But the "  +\
            "Z-axes in every file is not single-valued, which is not OK for "+\
            "this stitching method. The files cannot be merged. The length " +\
            "of the Z-axis in every supplied file was: "                     +\
            str(len(expected_z_axis))
        
        # If we've reached this point, then we have confirmed that
        # every axis (per resonator) is identical in this measurement.
        # We can go ahead an do the merge as expected.
        
        # For this merge, we will change the Z-axis, to reflect the N files
        # that we are about to merge.
        merged_file_index = []
        for make_list in range(len(big_fat_list_of_canvases[0])):
            merged_file_index.append(make_list + 1)
        merged_file_index = np.array(merged_file_index)
        
        # Create a new Z-axis key, insert it. Remove badness traces.
        new_key_to_be_inserted = ext_keys[1].copy()
        new_key_to_be_inserted['name'] = 'merged_file_index'
        new_key_to_be_inserted['unit'] = ''
        new_key_to_be_inserted['values'] = merged_file_index
        ext_keys.insert(1, new_key_to_be_inserted.copy())
        del new_key_to_be_inserted
        
        # Set up the final log dict list.
        ## Will we have to export a multiplexed readout? Or a file
        ## bearing several probability maps?
        if len(big_fat_list_of_canvases) > 1:
            ## TODO Add support for this kind of a merger of multiplexed readouts.
            ##      This will require some code to be rewritten here, 10 rows up, 20 rows down, or so. But it's not very hard at all.
            ##      IF I REMEMBER CORRECTLY then every new list entry in processed_data is just another resonator.
            raise NotImplementedError("Halted! For now, stitching of multiplexed readout files is not supported. Although adding this support is fairly straight forward.")
        else:
            # If we reach this point, we know that the log_dict_list is
            # fine just the way it is, and needs no modification.
            pass
        
        # Process the big fat canvas into something that can be
        # sent into the final export. We are still within the special
        # merger usage case, thus we know that every entry of
        # big_fat_list_of_canvases, is just another trace for the final data.
        # Simple enough.
        processed_data = []
        ## TODO figure out why the fuck there are [0] all over the damn place.
        for current_resonator in range(len(big_fat_list_of_canvases)):
            len_x = len( (big_fat_list_of_canvases[current_resonator])[0][0] )
            len_z = len(big_fat_list_of_canvases[current_resonator])
            final_canvas = np.zeros( (len_z, len_x) )
            for row in range(len_z):
                final_canvas[row] = (big_fat_list_of_canvases[current_resonator])[row][0]
            processed_data.append( final_canvas )
            del final_canvas
        
        ## TODO add support for scale and offset.
        print("WARNING: the data stitcher does currently not support scaling and offseting the data when used for mergers that you are attempting to achieve.")
        running_scale  = [1.0]
        running_offset = [0.0]
        
        # We may now go on to export the final, stitched, data.
        
    else:
        for current_resonator in range(len(big_fat_list_of_canvases)):
            # Get current big fat set of 2D-plots to merge.
            current_list_of_2D_arrays_to_merge = big_fat_list_of_canvases[current_resonator]
            current_list_of_corresponding_x_axes = big_fat_list_of_x_axes[current_resonator]
            current_list_of_corresponding_z_axes = big_fat_list_of_z_axes[current_resonator]
            
            raise NotImplementedError("Halted! Not finished.")
            
            
            # The list current_list_of_2D_arrays_to_merge contains N entries
            # where N is the number of files the user provided to the stitcher.
            # Every entry in current_list_of_2D_arrays_to_merge will be
            # merged together into one 2D array. That 2D array can have a single
            # pixel in Z-height, mind you. Or X-height for that matter.
            
            # Go through every axis and assemble the final X (or Z) axis
            # of the final canvas.
            for x_axis_assembly_sweep in range(len(current_list_of_corresponding_x_axes)):
                print(current_list_of_corresponding_x_axes[x_axis_assembly_sweep])
            for z_axis_assembly_sweep in range(len(current_list_of_corresponding_z_axes)):
                pass
            
            '''Using these axes values, assemble the final 2D canvas for this resonator.
            ## First, look at every axis [for resonator N], and make the final expected final_axis [for resonator N] for dims Z and X. Then fill this big-ass 2D plot with 0.0 in every pixel. When making the final 2D plot [for resonator N], go pixelpixel by pixelpixel ALONG THE AXES OF THE CURRENT_2D_PLOT and pick out every value that is to be added to some pixelpixel in the big-ass plot.
            ## If you notice that some entry in the big-ass plot IS NOT 0.0 --> THIS IS WHERE YOU ARE SUPPOSED TO TRIGGER THE OVERWRITE DETECTION, AND WARN THE USER THAT AN OVERWRITE HAS BEEN DETECTED AT VALUE pixelpixel.
            '''
    
    # Export combined data!
    filepath_to_exported_h5_file = export_processed_data_to_file(
        filepath_of_calling_script = filepath_of_calling_script,
    
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        processed_data = processed_data,
        fetched_data_scale = running_scale,
        fetched_data_offset = running_offset,
        resonator_freq_if_arrays_to_fft = resonator_freq_if_arrays_to_fft,
        
        timestamp = get_timestamp_string(),
        time_vector = running_vector_of_time,
        fetched_data_arr = [],
        log_browser_tag = log_browser_tag,
        log_browser_user = log_browser_user,
        default_exported_log_file_name = default_exported_log_file_name,
        append_to_log_name_before_timestamp = '_stitched',
        append_to_log_name_after_timestamp = '',
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export,
        select_resonator_for_single_log_export = select_resonator_for_single_log_export,
        save_raw_time_data = False
    )
    
    # Now, delete the old files.
    if delete_old_files_after_stitching:
        for item_to_delete in list_of_h5_files_to_stitch:
            # Does the file exist? Sanity check.
            if os.path.exists(item_to_delete):
                # Gentlemen, it has been a pleasure to play with you all tonight.
                os.remove(item_to_delete)
    return filepath_to_exported_h5_file
