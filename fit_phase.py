#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import os
import h5py
import numpy as np
from numpy import hanning as von_hann
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from fit_input_checker import verify_supported_input_argument_to_fit

def fit_phase_offset(
    raw_data_or_path_to_data,
    control_phase_arr = [],
    i_renamed_the_control_phase_arr_to = '',
    plot_for_this_many_seconds = 0.0,
    verbose = True
    ):
    ''' Given a filepath or raw data, fit the sinusoidal allegedly held within
        to yield the its phase offset from 0.
        
        raw_data_or_path_to_data can have one of the following data structures:
            > Folder   (string)
            > Filepath (string)
            > Raw data (list of numbers, or numpy array of numbers)
            > List of filepaths ( list of (see "filepath" above) )
            > List of raw data  ( list of (see "raw data" above) )
    '''
    
    # What did the user provide?
    verify = verify_supported_input_argument_to_fit(raw_data_or_path_to_data)
    the_user_provided_a_folder           = verify[0]
    the_user_provided_a_file             = verify[1]
    the_user_provided_raw_data           = verify[2]
    the_user_provided_a_list_of_files    = verify[3]
    the_user_provided_a_list_of_raw_data = verify[4]
    
    # If the user provided a single filepath, convert that path
    # into a list with only that path. Then do the list thing with everything.
    list_of_h5_files_to_fit = []
    if the_user_provided_a_file:
        raw_data_or_path_to_data = [ raw_data_or_path_to_data ]
        the_user_provided_a_file = False
        the_user_provided_a_list_of_files = True
    elif the_user_provided_a_folder:
        root_path = raw_data_or_path_to_data
        raw_data_or_path_to_data = []
        for file_in_folder in os.listdir( root_path ):
            raw_data_or_path_to_data.append( os.path.join(root_path,file_in_folder) )
        the_user_provided_a_folder = False
        the_user_provided_a_list_of_files = True
    if the_user_provided_a_list_of_files:
        # Ensure that only .hdf5 files (.h5) get added.
        for file_item in raw_data_or_path_to_data:
            if (file_item.endswith('.h5')) or (file_item.endswith('.hdf5')):
                if verbose:
                    print("Found file: \""+str(file_item)+"\"")
                list_of_h5_files_to_fit.append(file_item)
    
    # If the user provided raw data, or a list of raw data, then
    # cast everything into a list anyhow.
    if the_user_provided_raw_data:
        raw_data_or_path_to_data = [ raw_data_or_path_to_data ]
        the_user_provided_raw_data = False
        the_user_provided_a_list_of_raw_data = True
    
    # At this point, we either have a good list of files to work with,
    # or a list of raw data to work with.
    list_of_fitted_values = [0] * len(raw_data_or_path_to_data)
    for kk in range(len(raw_data_or_path_to_data)):
        
        # Grab the current item to fit.
        current_fit_item = raw_data_or_path_to_data[kk]
        
        # Get data.
        if the_user_provided_a_list_of_files:
            # The user provided a filepath to data.
            with h5py.File(os.path.abspath( current_fit_item ), 'r') as h5f:
                extracted_data = h5f["processed_data"][()]
                
                if i_renamed_the_control_phase_arr_to == '':
                    control_phase_arr_values = h5f[ "control_phase_arr" ][()]
                else:
                    control_phase_arr_values = h5f[i_renamed_the_control_phase_arr_to][()]
        
        else:
            # Then the user provided the data raw.
            
            # Catch bad user inputs.
            type_and_length_is_safe = False
            try:
                if len(control_phase_arr) != 0:
                    type_and_length_is_safe = True
            except TypeError:
                pass
            assert type_and_length_is_safe, "Error: the phase offset fitter was provided raw data to fit, but the data for the control phase array could not be used for fitting the data. The argument \"control_phase_arr\" was: "+str(control_phase_arr)
            
            # Assert fittable data.
            assert len(control_phase_arr) == len((current_fit_item[0])[0]), "Error: the user-provided raw data does not have the same length as the provided time control_phase_arr data."
            
            # Accept cruel fate and move on.
            extracted_data = current_fit_item
            control_phase_arr_values = control_phase_arr
        
        # If complex data provided, convert all values to magnitude.
        # Check "resonator 0" at Z-axis point 0.
        if type(((extracted_data[0])[0])[0]) == np.complex128:
            mag_vals_matrix = np.zeros_like(extracted_data, dtype = 'float64')
            for res in range(len( extracted_data )):
                for z_axis_val in range(len( extracted_data[res] )):
                    for x_axis_val in range(len( (extracted_data[res])[z_axis_val] )):
                        ((mag_vals_matrix[res])[z_axis_val])[x_axis_val] = (np.abs(((extracted_data[res])[z_axis_val])[x_axis_val])).astype('float64')
        else:
            mag_vals_matrix = extracted_data.astype('float64')
        
        # mag_vals_matrix now consists of the magnitude values
        # Assuming a two-resonator multiplexed readout, with 256 phase array
        # calues, then mag_vals_matrix has a shape of (2, 1, 256).
        
        # Report start?
        if verbose:
            if the_user_provided_a_list_of_files:
                print("Performing phase offset fitting on " + current_fit_item + "...")
            else:
                print("Comencing phase offset fitting on the provided raw data...")
        
        # There may be multiple resonators involved (a multiplexed readout).
        # Grab every trace (as in, a bias sweep will have many traces),
        # fit the trace, and store the result as a tuple of (value, error).
        fitted_values = [[]] * len(mag_vals_matrix)
        
        for current_res_ii in range(len( mag_vals_matrix )):
            # Note! See above comment on multiplexed sweeps being disabled for now.
            ## # Select current sweep values for this particular resonator.
            ## curr_control_phase_arr_values = control_phase_arr_values[current_res_ii]
            for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
                
                # Get current trace.
                current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
                
                # Try to fit current trace.
                try:
                    optimal_vals_x, fit_err_x = fit_phase(
                        phases = control_phase_arr_values,
                        oscillating_data = current_trace_to_fit
                    )
                    ##optimal_vals_x, fit_err_x = fit_phase(
                    ##    phases = curr_control_phase_arr_values,
                    ##    oscillating_data = current_trace_to_fit
                    ##)
                    
                    # Grab fitted values.
                    phase_offset = optimal_vals_x[3]
                    fit_error = fit_err_x[3]
                    
                    # Print result.
                    if verbose:
                        print("Phase offset from cosine fit of data: " + str(phase_offset) + " ±" + str(fit_error/2))
                    
                    # Store fit and its plusminus error bar.
                    ## Warning: append will append to both resonators unless
                    ## you are very darn careful at this step.
                    previous_content_in_fitted_values = (fitted_values[current_res_ii]).copy()
                    previous_content_in_fitted_values.append((phase_offset, fit_error/2))
                    fitted_values[current_res_ii] = previous_content_in_fitted_values.copy()
                    del previous_content_in_fitted_values
                    
                    # Plot?
                    if plot_for_this_many_seconds != 0.0:
                        # Get trace data using the fitter's function and acquired values.
                        fit_curve = cosine_function(
                            t         = control_phase_arr_values,
                            y_offset  = optimal_vals_x[0],
                            amplitude = optimal_vals_x[1],
                            period    = optimal_vals_x[2],
                            phase     = optimal_vals_x[3]
                        )
                        plt.plot(control_phase_arr_values, current_trace_to_fit, color="#034da3")
                        plt.plot(control_phase_arr_values, fit_curve, color="#ef1620")
                        plt.title('Local accumulated phase')
                        plt.ylabel('Demodulated amplitude [FS]')
                        plt.xlabel('Phase of virtual-Z gate [rad]')
                        
                        # If inserting a positive time for which we want to plot for,
                        # then plot for that duration of time. If given a negative
                        # time, then instead block the plotted display.
                        if plot_for_this_many_seconds > 0.0:
                            plt.show(block=False)
                            plt.pause(plot_for_this_many_seconds)
                            plt.close()
                        else:
                            plt.show(block=True)
                
                except RuntimeError:
                    # Fit failure.
                    optimal_vals_x  = [float("nan"), float("nan"), float("nan"), float("nan")]
                    fit_err_x       = [float("nan"), float("nan"), float("nan"), float("nan")]
                    
                    # Grab fitted values.
                    phase_offset = optimal_vals_x[3]
                    fit_error = fit_err_x[3]
                    
                    # Print result.
                    if the_user_provided_a_list_of_files:
                        print("Phase fit failure! Cannot fit: "+str(raw_data_or_path_to_data))
                    else:
                        print("Phase fit failure! Cannot fit the provided raw data.")
                    
                    # Store failed fit and its failed plusminus error bar.
                    (fitted_values[current_res_ii]).append((phase_offset, fit_error/2))
        
        # Append!
        list_of_fitted_values[kk] = fitted_values
    
    return list_of_fitted_values

def fit_conditional_phase_offset(
    raw_data_or_path_to_data_for_condition_OFF,
    raw_data_or_path_to_data_for_condition_ON,
    conditional_qubit_is,
    control_phase_arr = [],
    i_renamed_the_control_phase_arr_to = '',
    plot_for_this_many_seconds = 0.0,
    verbose = True
    ):
    ''' Given two datasets (files, typically), run the fit_phase_offset
        function to gather the phases of a conditional run of a virtual-Z
        phase-swept gate measurement.
        
        The fitter returns how much phase was added by the ON condition.
        
        Example:
        - CZ₂₀ is set to be OFF in a virtual-Z Ramsey measurement.
        - CZ₂₀ is set to be ON in another virtual-Z Ramsey measurement.
        - This fitter figures out how much phase was added by the CZ₂₀ gate.
        
        TODO:   In the future, I think it makes sense to try and automate
                away the "conditional_qubit_is" argument. As in, check
                which readout resonator returns straight lines.
                Then assign that resonator number as the conditional qubit.
    '''
    # Assert legal input arguments for conditional_qubit_is
    assert ((conditional_qubit_is.lower() == 'a') or  \
            (conditional_qubit_is.lower() == 'b')),   \
        "Error! Illegal input argument. The conditional qubit may only be " + \
        "either qubit 'A' or qubit 'B'."
    
    # List measurements to fit.
    file_list_to_fit = [ \
        raw_data_or_path_to_data_for_condition_OFF,
        raw_data_or_path_to_data_for_condition_ON
    ]
    fit_result_list = []
    for curr_filepath in file_list_to_fit:
        fit_result_list.append( \
            fit_phase_offset(
                raw_data_or_path_to_data = curr_filepath,
                control_phase_arr = control_phase_arr,
                i_renamed_the_control_phase_arr_to = i_renamed_the_control_phase_arr_to,
                plot_for_this_many_seconds = plot_for_this_many_seconds,
                verbose = verbose
            )
        )
    
    # Given which qubit was the conditional qubit, we may now grab the
    # phase of the qubit that was NOT the conditional one.
    resonator_index_with_sought_for_data = 1 if (conditional_qubit_is.lower() == 'a') else 0
    
    # We will now dig out the wanted phase value "val" in the data structure.
    # Assume here that sgt = resonator_index_with_sought_for_data
    ##  val in (val, errorbar) of res=sgt of Z=0 of file list entry 0.
    ##  val in (val, errorbar) of res=sgt of Z=0 of file list entry 1.
    ##     fit_result_list....[file_entry = 0][0... TODO can't quite explain this guy][res = sgt][z = 0][0 = val]
    ##     fit_result_list....[file_entry = 1][0... TODO can't quite explain this guy][res = sgt][z = 0][0 = val]
    phase_when_off = fit_result_list[0][0][resonator_index_with_sought_for_data][0][0]
    phase_when_on  = fit_result_list[1][0][resonator_index_with_sought_for_data][0][0]
    added_phase = phase_when_on - phase_when_off
    
    return added_phase

def cosine_function(
    t,
    y_offset,
    amplitude,
    period,
    phase
    ):
    ''' Function to be fitted against.
    '''
    return np.abs(amplitude) * np.cos(2*np.pi*(1/period) * t + phase) + y_offset

def fit_phase(phases, oscillating_data):
    ''' Grab submitted data and perform a fit to find the phase offset
        of the oscillating curve described by the data.
    '''
    x = phases
    y = oscillating_data
    
    # What is the current amplitude and y_offset?
    amplitude = (np.max(y)-np.min(y)) / 2
    y_offset  = np.min(y) + amplitude
    
    # What is the cosine's resolution? = Delta t, but where time val. t = x
    delta_x = x[1]-x[0]
    
    # What is the periodicity? Make discrete FFT, select discr. value
    # with highest amplitude in the frequency domain.
    fq_axis = np.fft.rfftfreq(len(x), delta_x)
    fft     = np.fft.rfft(y)
    period  = 1 / (fq_axis[1 + np.argmax( np.abs(fft[1:]) )])
    
    # Establish an estimate of what phase the cosine is currently on,
    # where 1.0 => the cosine starts at amplitude A,
    # ... and this phase corresponds to arccos( begins_at )
    begins_at = (y[0] - y_offset) / amplitude
    
    # It's unreasonable that the first sample point is somehow
    # higher (or lower) than the amplitude of the cosine waveform.
    # Take care of this, and assign a starting phase.
    if begins_at > 1:
        starting_phase = np.arccos(1)
    elif begins_at < -1:
        starting_phase = np.arccos(-1)
    else:
        starting_phase = np.arccos(begins_at)
    
    # Perform fit with decaying_cosine_function as the model function,
    # and the initial guess of parameters as seen below.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = cosine_function,
        xdata = x,
        ydata = y,
        p0    = (y_offset, amplitude, period, starting_phase)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    return optimal_vals, fit_err
