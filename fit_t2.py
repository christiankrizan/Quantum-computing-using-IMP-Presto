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
from math import isnan
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from fit_input_checker import verify_supported_input_argument_to_fit

def fit_exponential_decay_t2_echo(
    raw_data_or_path_to_data,
    delay_arr = [],
    i_renamed_the_delay_arr_to = '',
    plot_for_this_many_seconds = 0.0
    ):
    ''' From supplied data or datapath, fit the *refocused* qubit dephasing
        decay to find out the exponent of said dephasing, "T₂ echo."
        
        A failed fit (due to illegibly noisy input, for instance)
        will return a NaN ±NaN result.
        
        In principle, the decay is essentially = e^(-t/T2_echo).
        Meaning that when t = T2_echo => e^(-1) = 0.367879 for some normalised
        peak-to-peak value.
        
        A key difference between a regular T1 and a T2_echo fit,
        is that the Y-axis value at t -> infinity, will correspond to
        some mixed state between |0⟩ and |1⟩.
        
        This value goes into the fitter as an initial guess.
        
        plot_for_this_many_seconds will show a plot for a given number of
        seconds. If provided a negative value, then the plot will remain
        "blocking" until closed by the user.
        
        raw_data_or_path_to_data can have one of the following data structures:
            > Folder   (string)
            > Filepath (string)
            > Raw data (list of numbers, or numpy array of numbers)
            > List of filepaths ( list of (see "filepath" above) )
            > List of raw data  ( list of (see "raw data" above) )
    '''
    
    ## TODO: Somehow, add support so that the user can provide themselves
    ##       raw data for multiplexed T1 experiments too.
    
    ## TODO: Solve the multiplexed sweep issue. All multiplexed sweeps
    ##       where the x-axis is not identical for all qubits+resonators,
    ##       will (likely?) have to be exported seperately.
    ##       Meaning that this fitter will output some gibberish.
    ##       Perhaps a "select resonator" function is necessary?
    
    # What did the user provide?
    verify = verify_supported_input_argument_to_fit(raw_data_or_path_to_data)
    the_user_provided_a_folder           = verify[0]
    the_user_provided_a_file             = verify[1]
    the_user_provided_raw_data           = verify[2]
    the_user_provided_a_list_of_files    = verify[3]
    the_user_provided_a_list_of_raw_data = verify[4]
    
    # The T2_echo dephasing experiment could have been done with a
    # multiplexed readout. Thus, we declare a list for storing what time sweeps
    # were done for the different qubits.
    delay_arr_values = []
    
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
    list_of_fitted_values = []
    for current_fit_item in raw_data_or_path_to_data:
        
        # Get data.
        if the_user_provided_a_list_of_files:
            # The user provided a filepath to data.
            with h5py.File(os.path.abspath( current_fit_item ), 'r') as h5f:
                extracted_data = h5f["processed_data"][()]
                
                if i_renamed_the_delay_arr_to == '':
                    delay_arr_values = h5f[ "delay_arr" ][()]
                else:
                    delay_arr_values = h5f[i_renamed_the_delay_arr_to][()]
                
                # Note! Multiplexed functionality disabled for now. TODO
        
        else:
            # Then the user provided the data raw.
            
            # Catch bad user inputs.
            type_and_length_is_safe = False
            try:
                if len(delay_arr) != 0:
                    type_and_length_is_safe = True
            except TypeError:
                pass
            assert type_and_length_is_safe, "Error: the T₂_echo decay fitter was provided raw data to fit, but the data for the delay time array could not be used for fitting the data. The argument \"delay_arr\" was: "+str(delay_arr)
            
            # Assert fittable data.
            assert len(delay_arr) == len((current_fit_item[0])[0]), "Error: the user-provided raw data does not have the same length as the provided time delay (array) data."
            
            # Accept cruel fate and move on.
            extracted_data = current_fit_item
            delay_arr_values = delay_arr
        
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
        
        # mag_vals_matrix now consists of the magnitude values,
        # making up a T₂_echo decay.
        
        # Report start!
        if the_user_provided_a_list_of_files:
            print("Performing T₂_echo decay fitting on " + current_fit_item + "...")
        else:
            print("Commencing T₂_echo decay fitting of provided raw data...")
        
        # There may be multiple resonators involved (a multiplexed readout).
        # Grab every trace (as in, a bias sweep will have many traces),
        # fit the trace, and store the result as a tuple of (value, error).
        fitted_values = [[]] * len(mag_vals_matrix)
        
        for current_res_ii in range(len( mag_vals_matrix )):
            # Note! See above comment on multiplexed sweeps being disabled for now.
            ## # Select current sweep values for this particular resonator.
            ## curr_delay_arr_values = delay_arr_values[current_res_ii]
            for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
                
                # Get current trace.
                current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
                
                # Try to fit current trace.
                try:
                    optimal_vals_x, fit_err_x = fit_echoed_decay_single(
                        delays = delay_arr_values,
                        decaying_data = current_trace_to_fit
                    )
                    ##optimal_vals_x, fit_err_x = fit_decay(
                    ##    delays = curr_delay_arr_values,
                    ##    decaying_data = current_trace_to_fit
                    ##)
                    
                    # Grab fitted values.
                    t2_echo_time = optimal_vals_x[0]
                    fit_error = fit_err_x[0]
                    
                    # Print result.
                    print("T₂_echo from exponential decay fit of data: " + str(t2_echo_time) + " ±" + str(fit_error))
                    
                    # Store fit and its plusminus error bar.
                    ## Warning: append will append to both resonators unless
                    ## you are very darn careful at this step.
                    previous_content_in_fitted_values = (fitted_values[current_res_ii]).copy()
                    previous_content_in_fitted_values.append((t2_echo_time, fit_error))
                    fitted_values[current_res_ii] = previous_content_in_fitted_values.copy()
                    del previous_content_in_fitted_values
                    
                    ## Here, I have preserved the previous code snippet that
                    ## was patched with the .copy()-dance above.
                    ## Writing just "(fitted_values[current_res_ii]).append("
                    ## Seems to have been working this far. But I foresee
                    ## user inputs that might break that code snippet.
                    ## Case in point, multiplexed readouts.
                    #(fitted_values[current_res_ii]).append((t2_echo_time, fit_error))
                    
                    # Plot?
                    if plot_for_this_many_seconds != 0.0:
                        # Get trace data using the fitter's function and acquired values.
                        ## Make larger fit vector.
                        fit_vector_x = np.linspace(delay_arr_values[0], delay_arr_values[-1], 500)
                        fit_curve = exponential_decay_towards_mixed_state(
                            t          = fit_vector_x,
                            T2_echo    = optimal_vals_x[0],
                            y_nonmixed = optimal_vals_x[1],
                            y_mixed    = optimal_vals_x[2],
                        )
                        plt.plot(delay_arr_values, current_trace_to_fit, color="#034da3")
                        plt.plot(fit_vector_x, fit_curve, color="#ef1620")
                        plt.title('Ramsey spectroscopy with one refocusing pulse')
                        plt.ylabel('Demodulated amplitude [FS]')
                        plt.xlabel('Delay after the initial π/2 pulse [s]')
                        
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
                    optimal_vals_x  = [float("nan"), float("nan"), float("nan")]
                    fit_err_x       = [float("nan"), float("nan"), float("nan")]
                    
                    # Grab fitted values.
                    t2_echo_time = optimal_vals_x[0]
                    fit_error = fit_err_x[0]
                    
                    # Print result.
                    if i_provided_a_filepath:
                        print("T₂_echo fit failure! Cannot fit: "+str(data_or_filepath_to_data))
                    else:
                        print("T₂_echo fit failure! Cannot fit the provided raw data.")
                    
                    # Store failed fit and its failed plusminus error bar.
                    ## TODO this part here has probably not been protected against usage cases with multiplexed readouts.
                    (fitted_values[current_res_ii]).append((t2_echo_time, fit_error))
        
        # Append!
        list_of_fitted_values.append( fitted_values )
    
    # We're done.
    return fitted_values
    
def exponential_decay_towards_mixed_state(
    t,
    T2_echo,
    y_nonmixed,
    y_mixed
    ):
    ''' Function to be fitted against.
    '''
    return (y_nonmixed - y_mixed) * np.exp(-t / T2_echo) + y_mixed

def fit_echoed_decay_single(delays, decaying_data):
    ''' Grab submitted data of a single refocused Ramsey spectroscopy trace
        and perform a fit to find the exponentially decaying oscillatory curve.
        And specifically, the T2_echo part of the temporal exponent at which
        this decay happens.
    '''
    # Set time t to "delays" and get total duration.
    t = delays
    duration_of_experiment = t[-1] - t[0]
    
    # Find a good first guess for the curve's amplitude and offset.
    # Is the decay happening from ground to mixed, or from excited to mixed?
    y_nonmixed = decaying_data[0]
    y_mixed    = decaying_data[-1]
    y_offset   = y_mixed
    
    # Find a good first guess for T2_echo.
    # 0.36787944117 ~= e^(-t/T2_echo) [plus offset] where t = T2_echo.
    pop_val_for_likely_T2_echo = \
        0.36787944117 * np.abs(y_nonmixed - y_mixed) + y_offset
    
    # Get index for value closest to to this y value for the to-be-guessed T2_echo
    T2_echo = delays[np.abs(decaying_data - pop_val_for_likely_T2_echo).argmin()]
    
    # Perform fit with decaying_cosine_function as the model function,
    # and the initial guess of parameters as seen below.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = exponential_decay_towards_mixed_state,
        xdata = t,
        ydata = decaying_data,
        p0    = (T2_echo, y_nonmixed, y_mixed)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    return optimal_vals, fit_err
    


def fit_oscillating_exponential_decay_t2_asterisk(
    raw_data_or_path_to_data,
    delay_arr = [],
    i_renamed_the_delay_arr_to = '',
    plot_for_this_many_seconds = 0.0
    ):
    ''' From supplied data or datapath, fit the qubit dephasing decay
        to find out the exponent of said dephasing, T₂*.
        
        A failed fit (due to illegibly noisy input, for instance)
        will return a NaN ±NaN result.
        
        plot_for_this_many_seconds will show a plot for a given number of
        seconds. If provided a negative value, then the plot will remain
        "blocking" until closed by the user.
        
        raw_data_or_path_to_data can have one of the following data structures:
            > Folder   (string)
            > Filepath (string)
            > Raw data (list of numbers, or numpy array of numbers)
            > List of filepaths ( list of (see "filepath" above) )
            > List of raw data  ( list of (see "raw data" above) )
    '''
    
    ## TODO: Somehow, add support so that the user can provide themselves
    ##       raw data for multiplexed T2* experiments too.
    
    ## TODO: Solve the multiplexed sweep issue. All multiplexed sweeps
    ##       where the x-axis is not identical for all qubits+resonators,
    ##       will (likely?) have to be exported seperately.
    ##       Meaning that this fitter will output some gibberish.
    ##       Perhaps a "select resonator" function is necessary?
    
    # What did the user provide?
    verify = verify_supported_input_argument_to_fit(raw_data_or_path_to_data)
    the_user_provided_a_folder           = verify[0]
    the_user_provided_a_file             = verify[1]
    the_user_provided_raw_data           = verify[2]
    the_user_provided_a_list_of_files    = verify[3]
    the_user_provided_a_list_of_raw_data = verify[4]
    
    # The T2* dephasing experiment could have been done with a multiplexed
    # readout. Thus, we declare a list for storing what time sweeps
    # were done for the different qubits.
    delay_arr_values = []
    
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
    list_of_fitted_values = []
    for current_fit_item in raw_data_or_path_to_data:
        
        # Get data.
        if the_user_provided_a_list_of_files:
            # The user provided a filepath to data.
            with h5py.File(os.path.abspath( current_fit_item ), 'r') as h5f:
                extracted_data = h5f["processed_data"][()]
                
                if i_renamed_the_delay_arr_to == '':
                    delay_arr_values = h5f[ "delay_arr" ][()]
                else:
                    delay_arr_values = h5f[i_renamed_the_delay_arr_to][()]
                
                # Note! Multiplexed functionality disabled for now. TODO
        
        else:
            # Then the user provided the data raw.
            
            # Catch bad user inputs.
            type_and_length_is_safe = False
            try:
                if len(delay_arr) != 0:
                    type_and_length_is_safe = True
            except TypeError:
                pass
            assert type_and_length_is_safe, "Error: the T1 decay fitter was provided raw data to fit, but the data for the delay time array could not be used for fitting the data. The argument \"delay_arr\" was: "+str(delay_arr)
            
            # Assert fittable data.
            assert len(delay_arr) == len((current_fit_item[0])[0]), "Error: the user-provided raw data does not have the same length as the provided time delay (array) data."
            
            # Accept cruel fate and move on.
            extracted_data = current_fit_item
            delay_arr_values = delay_arr
        
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
        
        # mag_vals_matrix now consists of the magnitude values,
        # making up a T₂* dephasing oscillatory decay.
        
        # Report start!
        if the_user_provided_a_list_of_files:
            print("Performing T₂* dephasing oscillatory decay fitting on " + current_fit_item + "...")
        else:
            print("Commencing T₂* dephasing oscillatory decay fitting of provided raw data...")
        
        # There may be multiple resonators involved (a multiplexed readout).
        # Grab every trace (as in, a bias sweep will have many traces),
        # fit the trace, and store the result as a tuple of (value, error).
        fitted_values = [[]] * len(mag_vals_matrix)
        
        for current_res_ii in range(len( mag_vals_matrix )):
            # Note! See above comment on multiplexed sweeps being disabled for now.
            ## # Select current sweep values for this particular resonator.
            ## curr_delay_arr_values = delay_arr_values[current_res_ii]
            for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
                
                # Get current trace.
                current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
                
                # Try to fit current trace.
                try:
                    optimal_vals_x, fit_err_x = fit_oscillatory_decay_single(
                        delays = delay_arr_values,
                        decaying_data = current_trace_to_fit
                    )
                    ##optimal_vals_x, fit_err_x = fit_oscillatory_decay_single(
                    ##    delays = curr_delay_arr_values,
                    ##    decaying_data = current_trace_to_fit
                    ##)
                    
                    # Grab fitted values.
                    t2_asterisk_time = optimal_vals_x[0]
                    fit_error = fit_err_x[0]
                    y_offset_from_fit = optimal_vals_x[2]
                    y_offset_error_from_fit = fit_err_x[2]
                    
                    # Print results.
                    print("T₂* from exponential decay fit of data: " + str(t2_asterisk_time) + " ±" + str(fit_error))
                    print("Offset in magnitude [FS]: "+str(y_offset_from_fit)+" ±"+str(y_offset_error_from_fit))
                    
                    # Store fit and its plusminus error bar.
                    ## Warning: append will append to both resonators unless
                    ## you are very darn careful at this step.
                    previous_content_in_fitted_values = (fitted_values[current_res_ii]).copy()
                    previous_content_in_fitted_values.append((t2_asterisk_time, fit_error))
                    fitted_values[current_res_ii] = previous_content_in_fitted_values.copy()
                    del previous_content_in_fitted_values
                    
                    ## Here, I have preserved the previous code snippet that
                    ## was patched with the .copy()-dance above.
                    ## Writing just "(fitted_values[current_res_ii]).append("
                    ## Seems to have been working this far. But I foresee
                    ## user inputs that might break that code snippet.
                    ## Case in point, multiplexed readouts.
                    #(fitted_values[current_res_ii]).append((t2_asterisk_time, fit_error))
                    
                    # Plot?
                    if plot_for_this_many_seconds != 0.0:
                        # Get trace data using the fitter's function and acquired values.
                        ## Make larger fit vector.
                        fit_vector_x = np.linspace(delay_arr_values[0], delay_arr_values[-1], 500)
                        fit_curve = exponential_oscillatory_decay(
                            t           = fit_vector_x,
                            T2_asterisk = optimal_vals_x[0],
                            amplitude   = optimal_vals_x[1],
                            y_offset    = optimal_vals_x[2],
                            frequency   = optimal_vals_x[3],
                            phase       = optimal_vals_x[4],
                        )
                        plt.plot(delay_arr_values, current_trace_to_fit, color="#034da3")
                        plt.plot(fit_vector_x, fit_curve, color="#ef1620")
                        plt.title('Ramsey spectroscopy')
                        plt.ylabel('Demodulated amplitude [FS]')
                        plt.xlabel('Delay after the initial π/2 pulse [s]')
                        
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
                    optimal_vals_x  = [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]
                    fit_err_x       = [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]
                    
                    # Grab fitted values.
                    t2_asterisk_time = optimal_vals_x[0]
                    fit_error = fit_err_x[0]
                    
                    # Print result.
                    if i_provided_a_filepath:
                        print("T₂* fit failure! Cannot fit: "+str(data_or_filepath_to_data))
                    else:
                        print("T₂* fit failure! Cannot fit the provided raw data.")
                    
                    # Store failed fit and its failed plusminus error bar.
                    ## TODO this part here has probably not been protected against usage cases with multiplexed readouts.
                    (fitted_values[current_res_ii]).append((t2_asterisk_time, fit_error))
        
        # Append!
        list_of_fitted_values.append( fitted_values )
    
    # We're done.
    return fitted_values

def exponential_oscillatory_decay(
    t,
    T2_asterisk,
    amplitude,
    y_offset,
    frequency,
    phase
    ):
    ''' Function to be fitted against.
    '''
    return amplitude * np.exp(-t / T2_asterisk) * np.cos(2.0*np.pi * frequency * t + phase) + y_offset

def fit_oscillatory_decay_single(delays, decaying_data):
    ''' Grab submitted data of a single Ramsey spectroscopy trace and perform
        a fit to find the exponentially decaying oscillatory curve.
        And specifically, the T2* part of the temporal exponent at which
        this decay happens.
    '''
    
    # Set time t to "delays" and get total duration.
    t = delays
    duration_of_experiment = t[-1] - t[0]
    
    # Find a good first guess for the curve's amplitude and offset.
    y_excited = np.max(decaying_data)
    y_ground  = np.min(decaying_data)
    y_peak_to_peak = y_excited - y_ground
    amplitude = 0.5 * y_peak_to_peak
    y_offset = y_ground + amplitude
    
    # Find a good first guess for T2*.
    # 0.36787944117 ~= e^(-t/T2*) [plus offset and squiggles] where t = T2*.
    pop_val_for_likely_T2_asterisk = 0.36787944117 * amplitude + y_offset
    
    # Get index for value closest to to this y value for the to-be-guessed T2*
    # With one modification: we could in theory accidentally pick a part of
    # the oscillating curve that is currently on its negative oscillation.
    # Thus, let's rectify the oscillating curve so that we're more
    # likely to hit a good guess.
    rectified_decaying_curve = np.abs(decaying_data - y_offset) + y_offset
    
    # Get index for value closest to to this y value for the to-be-guessed T2*
    T2_asterisk = delays[np.abs(rectified_decaying_curve - pop_val_for_likely_T2_asterisk).argmin()]
    
    # What is the cosine's resolution? = Delta t
    delta_x = delays[1]-delays[0]
    
    # What is the oscillation's frequency? Make discrete FFT,
    # select discr. value with highest amplitude in the frequency domain.
    fq_axis   = np.fft.rfftfreq(len(delays), delta_x)
    fft       = np.fft.rfft(decaying_data)
    frequency = fq_axis[1 + np.argmax( np.abs(fft[1:] ))]
    
    # Establish an estimate of what phase the cosine is currently on,
    # where 1.0 => the cosine starts at amplitude A,
    # ... and this phase corresponds to arccos( begins_at )
    begins_at = (decaying_data[0] - y_offset) / amplitude
    
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
        f     = exponential_oscillatory_decay,
        xdata = t,
        ydata = decaying_data,
        p0    = (T2_asterisk, amplitude, y_offset, frequency, starting_phase)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    return optimal_vals, fit_err