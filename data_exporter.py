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
    data_to_store_consists_of_time_traces_only = False,
    source_code_of_executing_file = '',
    append_to_log_name_before_timestamp = '',
    append_to_log_name_after_timestamp  = '',
    select_resonator_for_single_log_export = '',
    force_matrix_reshape_flip_row_and_column = False,
    suppress_log_browser_export = False,
    save_raw_time_data = False,
    
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
    
    # Build a processed_data tensor for later.
    processed_data = []
    
    # Is the user only interested in time traces?
    if not data_to_store_consists_of_time_traces_only:
        
        # Get index corresponding to integration_window_start and
        # integration_window_stop respectively.
        integration_start_index = np.argmin(np.abs(time_vector - integration_window_start))
        integration_stop_index  = np.argmin(np.abs(time_vector - integration_window_stop ))
        integration_indices     = np.arange(integration_start_index, integration_stop_index)
        
        # Acquire the DFT sample frequencies contained within the
        # fetched_data_arr trace. freq_arr contains the centres of
        # the (representable) segments of the discretised frequency axis.
        dt = time_vector[1] - time_vector[0]
        '''num_samples = len(integration_indices)'''
        num_samples = 8*len(integration_indices)
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
        for _item in integration_indices_list:
            if len(_item) <= 1:
                '''resp_fft = np.fft.fft(fetched_data_arr[:, 0, integration_indices], axis=-1)'''
                arr_to_fft = fetched_data_arr[:, 0, integration_indices]
                new_arr = []
                for row in range(len(arr_to_fft)):
                    new_arr.append(np.append(arr_to_fft[row], np.zeros([1,len(arr_to_fft[0])*7])))
                new_arr = np.array(new_arr)
                resp_fft = np.fft.fft(new_arr, axis=-1)
                
                # At this point, to clear up memory, we should clear out
                # new_arr, arr_to_fft, and possibly also fetched_data_arr
                del new_arr
                del arr_to_fft
                
                '''processed_data.append( 2/num_samples * resp_fft[:, _item[0]] )'''
                processed_data.append( 2/(num_samples/8) * resp_fft[:, _item[0]] ) # TODO: Should, or should not, the num_samples part be modified after zero-padding the data?
            else:
                print("WARNING: Currently, resonator frequency sweeps are not FFT'd due to a lack of demodulation. The Y-axis offset following your sweep is thus completely fictional.") # TODO
                print("WARNING: The current FFT method is not demodulating sufficiently. If you are using large averages, you may experience a weird offset on your Y-axis.") # TODO
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
        
    else:
        # The user is only interested in time trace data.
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
        
        timestamp = timestamp,
        log_browser_tag = log_browser_tag,
        log_browser_user = log_browser_user,
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
    log_browser_tag = 'krizan',
    log_browser_user = 'Christian Križan',
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
    
    # First, check if the timestamp is valid.
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
            # TODO:  This part should cover an arbitrary number of fetched_data_arr
            #        arrays. And, this entire subroutine should be made fully
            #        generic.
            # TODO2: Is not in fact the TODO above fixed now?
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
    savefile_string_h5py = name_of_measurement_that_ran + append_to_log_name_before_timestamp + timestamp + append_to_log_name_after_timestamp + '.h5'
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
        
        print("Data saved using H5PY, see " + save_path_h5py)
    
    # Return the .h5 save path to the calling function
    return save_path_h5py


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
    list_of_h5_files_to_stitch,
    delete_old_files_after_stitching = False,
    halt_if_x_and_z_axes_are_identical = True, # Halt to ensure there is no overwrite because of poor user arguments.
    use_this_scale = [1.0],
    use_this_offset = [0.0],
    log_browser_tag = 'krizan',
    log_browser_user = 'Christian Križan',
    use_log_browser_database = True,
    suppress_log_browser_export = False,
    select_resonator_for_single_log_export = ''
    ):
    ''' A function previously known as "stitch_exported_data_files."
    
        For all .h5 files in the .h5 file list argument,
        grab all data and stitch together one export.
        
        Finally, delete all old files if requested.
    '''
    try:
        assert len(list_of_h5_files_to_stitch) != 0, \
            "Error: the stitched routine was called for exported data files. " + \
            "But, no data filepaths were provided."
    except TypeError as e:
        raise TypeError("Error: the data export stitcher was called with a non-list argument. The full type error is: \n"+str(e))
    
    # There may be different scales and offsets in the files.
    # We'll keep a running tab to make sure these scales and offsets do
    # not differ.
    running_scale  = []
    running_offset = []
    ## TODO: Grab the scale and offset data, and re-scale + re-offset the data.
    ##       The arguments use_this_scale and use_this_offset have been
    ##       prepared for this purpose.
    
    # There may be different time traces in the files.
    # We'll keep track so that no data files have data collected at
    # different time trace settings.
    running_vector_of_time = []
    ## TODO: Somehow, stitch things together even if the data is
    ##       collected at non-identical times.
    
    # There will be an attempt at grabbing a filepath for data exporting.
    filepath_of_calling_script = []
    
    # Prepare lists that will be appended onto in the upcoming with-case loop.
    list_of_swept_keys = []
    swept_content      = []
    list_of_unswept_keys = []
    unswept_content      = []
    
    # Prepare a canvas. Its X and Z sizes are the sizes of the swept keys.
    # These axes represent the swept keys. While, the content of the entry
    # (so Y) is the processed_data entry for this XZ key.
    canvas = []
    canvas_axis_x = []
    canvas_axis_z = []
    previous_x_axes = []
    previous_z_axes = []
    
    # Prepare ext_keys and log_dict_list, these variables will be
    # dictified later.
    ext_keys = []
    log_dict_list = []
    
    # Treat every provided item in the list!
    for current_filepath_item_in_list in list_of_h5_files_to_stitch:
        
        # Get data.
        with h5py.File(current_filepath_item_in_list, 'r') as h5f:
            
            # Notify the user.
            print("Stitching file: "+str(current_filepath_item_in_list))
            
            # Did this file have a scale set for its data?
            try:
                scale = h5f["User_set_scale_to_Y_axis"][()]
                if running_scale == []:
                    running_scale = scale
                else:
                    if not (scale == running_scale).all():
                        raise NotImplementedError("File \""+current_filepath_item_in_list+"\" has a different scale than all previous files in the stitching; the stitcher does currently not support re-scaling.")
            except KeyError:
                # "There is no scale, Gromit."
                pass
            
            # Did this file have an offset to its data?
            try:
                offset = h5f["User_set_offset_to_Y_axis"][()]
                if running_offset == []:
                    running_offset = offset
                else:
                    if not (offset == running_offset).all():
                        raise NotImplementedError("File \""+current_filepath_item_in_list+"\" has a different offset than all previous files in the stitching; the stitcher does currently not support re-offsetting the data files.")
            except KeyError:
                # "There is no offset, Gromit."
                pass
            
            # Make sure that the time traces are compatible.
            try:
                vector_of_time = h5f["time_vector"][()]
                if running_vector_of_time == []:
                    running_vector_of_time = vector_of_time
                else:
                    if not (vector_of_time == running_vector_of_time).all():
                        raise NotImplementedError("File \""+current_filepath_item_in_list+"\" contains data collected with a time trace, that in turn differs from every other time trace in all other stitched files. Currently, the data stitcher does not support data collected with different time traces. In practice, time traces will be identical if the sampling rate and sampling window length are identical between the two measurements.")
            except KeyError:
                # "There is no time, Gromit."
                pass
            
            # Get filepath and file name for making a file export.
            if filepath_of_calling_script == []:
                try:
                    filepath_of_calling_script = os.path.abspath((h5f["filepath_of_calling_script"][()]).decode('UTF-8'))
                except KeyError:
                    # "There is no path, Gromit."
                    filepath_of_calling_script = os.path.realpath(__file__)
            
            # Grab data! List all .keys(), remove the known culprits,
            # and make arrays.
            ##list_of_swept_keys = [] # See outside of the with-case
            ##swept_content      = [] # See outside of the with-case
            for item in h5f.keys():
                if item.startswith('fetched_data_arr'):
                    # Raw data will not be stitched, that's kind of the point.
                    print("WARNING: File \""+current_filepath_item_in_list+"\" contains raw data. This data will not be excluded from the stitched-together file export. The whole point of having the data stitcher is to avoid loading the primary PC memory with hundreds of gibibytes of data.")
                elif ( \
                    (item != 'time_vector') and \
                    (item != 'processed_data') and \
                    (item != 'User_set_scale_to_Y_axis') and \
                    (item != 'User_set_offset_to_Y_axis') and \
                    (item != 'filepath_of_calling_script') and \
                    (item != 'First_key_that_was_swept') and \
                    (item != 'Second_key_that_was_swept') ):
                    
                    # Was this parameter one that's assumed to be an
                    # axis in Labber's Log Browser? Then move it to
                    # place 0 or 1.
                    first_key = str((h5f["First_key_that_was_swept"][()]).decode("utf-8"))
                    second_key = str((h5f["Second_key_that_was_swept"][()]).decode("utf-8"))
                    first_key_found = False
                    
                    # If the entry at first_key, or second_key, was stored
                    # as an attribute, then said value should go in
                    # unswept_content, not swept content.
                    try:
                        dummy1 = h5f[ first_key ][()]
                        del dummy1
                    except KeyError:
                        # The value is an attribute.
                        first_key = "FIRST_KEY_IS_NOT_A_SWEPT_VALUE"
                        ## TODO:    But this entry should still be at the
                        ##          first place of the unswept_content list!
                    try:
                        dummy2 = h5f[ second_key ][()]
                        del dummy2
                    except KeyError:
                        # The value is an attribute.
                        second_key = "SECOND_KEY_IS_NOT_A_SWEPT_VALUE"
                        ## TODO:    But this entry should still be at the
                        ##          second place of the unswept_content list!
                    
                    # Let's get the swept values.
                    if not (item in list_of_swept_keys):
                        # There was a swept parameter that should be added.
                        if (item == first_key):
                            # This item must be in the start.
                            list_of_swept_keys.insert(0,str(item))
                            swept_content.insert(0,h5f[str(item)][()])
                            first_key_found = True
                        elif (item == second_key):
                            if first_key_found:
                                # Then insert at position 2, so [1]
                                list_of_swept_keys.insert(1,str(item))
                                swept_content.insert(1,h5f[str(item)][()])
                            else:
                                # Then insert in the very beginning, so [0]
                                list_of_swept_keys.insert(0,str(item))
                                swept_content.insert(0,h5f[str(item)][()])
                        else:
                            # Just append this swept item to the list anyhow.
                            list_of_swept_keys.append(str(item))
                            swept_content.append(h5f[str(item)][()])
                        
                    else:
                        # The swept quantity was in the previous list.
                        # Let's update its values.
                        swept_content[list_of_swept_keys.index(str(item))] = (h5f[str(item)][()])
            
            ## Update the canvas with new values!
            
            # Get values.
            curr_processed_data = h5f["processed_data"][()]
            
            # First, grab static measurement variables first.
            # This way, we can "repair" a measurement with only a single
            # "swept" datapoint.
            ##list_of_unswept_keys = [] # See outside of the with-case
            ##unswept_content      = [] # See outside of the with-case
            for item in h5f.attrs.keys():
                if  ((item != 'ext_keys') and \
                    (item != 'log_dict_list') ):
                    # There was an unswept parameter that might need appendage.
                    if not (item in list_of_unswept_keys):
                        list_of_unswept_keys.append(str(item))
                        unswept_content.append(h5f.attrs[str(item)])
            
            # Store what x- and z axes are being used, in order to look
            # out for overwrites.
            try:
                previous_x_axes.append(swept_content[0])
            except IndexError:
                # There are apparently *no* swept values.
                # Grab the first value of the unswept ones.
                swept_content[0] = unswept_content[0]
                previous_x_axes.append(swept_content[0])
            try:
                previous_z_axes.append(swept_content[0])
            except IndexError:
                # There is only a single swept value.
                swept_content[1] = unswept_content[1]
                previous_z_axes.append(swept_content[1])
            
            # Remember, processed_data consists of n entries for n resonators.
            # Every entry processed_data[n] contains the XZ canvas for that
            # resonator (or, discriminated state).
            
            # Figure out how many resonator (or discriminated state) entries
            # the processed data contains, and what index_x and index_z to
            # insert at.
            if len(canvas) == 0:
                # Simple choice.
                canvas = curr_processed_data
                canvas_axis_x = swept_content[0]
                canvas_axis_z = swept_content[1]
            else:
                # Not so simple choice, there are already entries in the
                # canvas.
                if np.array_equal(canvas_axis_x, swept_content[0]):
                    ## All x-values are identical. Should we simply append more rows?
                    # Ensure that the order is correct (entries fall / rise)
                    if  (((swept_content[1])[0] > canvas_axis_z[-1]) and
                        ((swept_content[1])[-1] > canvas_axis_z[0])):
                        # My new entries all rise from where the canvas stops.
                        new_canvas = []
                        for res in range(len(canvas)):
                            new_canvas.append(np.append(canvas[res], curr_processed_data[res], axis = 0))
                        canvas = np.array(new_canvas)
                        canvas_axis_z = np.append(canvas_axis_z, swept_content[1])
                    elif (((swept_content[1])[-1] < canvas_axis_z[0]) and
                         ((swept_content[1])[0] < canvas_axis_z[-1])):
                        # My new entries all fall below the previous entries
                        # in the canvas.
                        new_canvas = []
                        for res in range(len(canvas)):
                            new_canvas.append(np.append(curr_processed_data[res], canvas[res], axis = 0))
                        canvas = np.array(new_canvas)
                        canvas_axis_z = np.append(swept_content[1], canvas_axis_z)
                    elif (swept_content[1] in previous_z_axes):
                        # We detected an overwrite risk! Halt?
                        if halt_if_x_and_z_axes_are_identical:
                            raise ValueError("Halted! Overwrite risk detected. The file \""+str(current_filepath_item_in_list)+"\" has an identical x-axis to all other previous files, but the new file's z-axis risks overwriting data that has already been added. Set the function argument \"halt_if_x_and_z_axes_are_identical = False\" to ignore and attempt to append the new data anyway.")
                        # At this point, we try to simply append new resonator / discrimination entries and hope for the best.
                        canvas = np.append(canvas, curr_processed_data, axis = 0)
                    else:
                        raise NotImplementedError("Halted! Interleaving data is currently not supported.") # TODO
                elif np.array_equal(canvas_axis_z, swept_content[1]):
                    ## All z-values are identical. Should we simply append more columns?
                    # Ensure that the order is correct (entries fall / rise)
                    if  (((swept_content[0])[0] > canvas_axis_x[-1]) and
                        ((swept_content[0])[-1] > canvas_axis_x[0])):
                        # My new entries all rise from where the canvas stops.
                        new_canvas = []
                        for res in range(len(canvas)):
                            new_canvas.append(np.append(canvas[res], curr_processed_data[res], axis = -1)) # N.B. appended column-wise
                        canvas = np.array(new_canvas)
                        canvas_axis_x = np.append(canvas_axis_x, swept_content[0])
                    elif (((swept_content[0])[-1] < canvas_axis_x[0]) and
                         ((swept_content[0])[0] < canvas_axis_x[-1])):
                        # My new entries all fall below the previous entries
                        # in the canvas.
                        new_canvas = []
                        for res in range(len(canvas)):
                            new_canvas.append(np.append(curr_processed_data[res], canvas[res], axis = -1)) # N.B. appended column-wise
                        canvas = np.array(new_canvas)
                        canvas_axis_x = np.append(swept_content[0], canvas_axis_x)
                    elif (swept_content[0] in previous_x_axes):
                        # We detected an overwrite risk! Halt?
                        if halt_if_x_and_z_axes_are_identical:
                            raise ValueError("Halted! Overwrite risk detected. The file \""+str(current_filepath_item_in_list)+"\" has an identical z-axis to all other previous files, but the new file's x-axis risks overwriting data that has already been added. Set the function argument \"halt_if_x_and_z_axes_are_identical = False\" to ignore and attempt to append the new data anyway.")
                        # At this point, we try to simply append new resonator / discrimination entries and hope for the best.
                        for res in range(len(curr_processed_data)):
                            canvas = np.append(canvas, curr_processed_data, axis = 0) # N.B. should be axis = 0, not 1.
                    else:
                        raise NotImplementedError("Halted! Interleaving data is currently not supported.") # TODO
                else:
                    # There are no common axes. All hope is lost; all data entries must be interleaved.
                    raise NotImplementedError("Halted! Interleaving data is currently not supported.") # TODO
                    
            # Get initial values for the ext_keys and log_dict_list keys.
            if ext_keys == []:
                ext_keys = json.loads( h5f.attrs["ext_keys"] )
            elif log_dict_list == []:
                log_dict_list = json.loads( h5f.attrs["log_dict_list"] )
    
    # Prepare the ext_keys and log_dict_list from the stitched data files.
    for li in range(len(ext_keys)):
        list_item_dict = ext_keys[li].copy()
        if list_item_dict['name'] in list_of_swept_keys:
            # This value in the ext_keys list has been changed, and must
            # be updated in order not to get a Labber Log Browser error.
            # The Log browser error, like many Labber API errors,
            # will be indistinct and unhelpful. The issue is that there
            # is no correct key entry corresponding to the new plot axes.
            # We must make these axes here (by updating ext_keys correctly).
            idx_of_new_value = list_of_swept_keys.index( list_item_dict['name'] )
            if idx_of_new_value == 0:
                list_item_dict['values'] = canvas_axis_x
            elif idx_of_new_value == 1:
                list_item_dict['values'] = canvas_axis_z
            else:
                list_item_dict['values'] = swept_content[idx_of_new_value]
                # TODO: Now, how should you deal with static values
                # that change from measurement to measurement, like
                # "coupler_bias_min" and "coupler_bias_max"?
                # Currently, I have chosen to ignore it.
        ext_keys[li] = list_item_dict.copy()
        del list_item_dict
    
    # Export combined data!
    filepath_to_exported_h5_file = export_processed_data_to_file(
        filepath_of_calling_script = filepath_of_calling_script,
    
        ext_keys = ext_keys,
        log_dict_list = log_dict_list,
        
        processed_data = canvas,
        fetched_data_scale = running_scale,
        fetched_data_offset = running_offset,
        
        timestamp = get_timestamp_string(),
        time_vector = running_vector_of_time,
        fetched_data_arr = [],
        log_browser_tag = log_browser_tag,
        log_browser_user = log_browser_user,
        append_to_log_name_before_timestamp = '_stitched',
        append_to_log_name_after_timestamp = '',
        use_log_browser_database = use_log_browser_database,
        suppress_log_browser_export = suppress_log_browser_export,
        select_resonator_for_single_log_export = select_resonator_for_single_log_export,
        save_raw_time_data = False
    )
    
    # Now, delete the old files.
    if delete_old_files_after_stitching:
        raise NotImplementedError("Halted! Deleting old files not yet implemented.")
        for item_to_delete in list_of_h5_files_to_stitch:
            # TODO DELETE!
            pass #TODO
    
    return filepath_to_exported_h5_file
    