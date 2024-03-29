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

def fit_exponential_decay_t1(
    raw_data_or_path_to_data,
    delay_arr = [],
    i_renamed_the_delay_arr_to = '',
    plot_for_this_many_seconds = 0.0
    ):
    ''' From supplied data or datapath, fit the energy relaxation decay
        to find out the exponent of which the energy decays, T₁.
        
        A failed fit (due to illegibly noisy input, for instance)
        will return a NaN ±NaN result.
        
        In principle, the decay is essentially = e^(t/T1).
        Meaning that when t = T1 => e^(-1) = 0.367879 for some normalised
        peak-to-peak value.
        
        Meaning in turn that the returned value T1 is ~roughly~ the time value
        at which the qubit has lost 63.21% of its excited state population.
        
        This value goes into the fitter as an initial guess.
        
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
    
    # The T1 energy relaxation experiment could have been done with a
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
        # making up a T1 decay.
        
        # Report start!
        if the_user_provided_a_list_of_files:
            print("Performing T₁ energy relaxation decay fitting on " + current_fit_item + "...")
        else:
            print("Commencing T₁ energy relaxation decay fitting on the provided raw data...")
        
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
                    optimal_vals_x, fit_err_x = fit_decay(
                        delays = delay_arr_values,
                        decaying_data = current_trace_to_fit
                    )
                    ##optimal_vals_x, fit_err_x = fit_decay(
                    ##    delays = curr_delay_arr_values,
                    ##    decaying_data = current_trace_to_fit
                    ##)
                    
                    # Grab fitted values.
                    t1_time = optimal_vals_x[0]
                    fit_error = fit_err_x[0]
                    
                    # Print result.
                    print("T₁ from exponential decay fit of data: " + str(t1_time) + " ±" + str(fit_error))
                    
                    # Store fit and its plusminus error bar.
                    ## Warning: append will append to both resonators unless
                    ## you are very darn careful at this step.
                    previous_content_in_fitted_values = (fitted_values[current_res_ii]).copy()
                    previous_content_in_fitted_values.append((t1_time, fit_error))
                    fitted_values[current_res_ii] = previous_content_in_fitted_values.copy()
                    del previous_content_in_fitted_values
                    
                    ## Here, I have preserved the previous code snippet that
                    ## was patched with the .copy()-dance above.
                    ## Writing just "(fitted_values[current_res_ii]).append("
                    ## Seems to have been working this far. But I foresee
                    ## user inputs that might break that code snippet.
                    ## Case in point, multiplexed readouts.
                    #(fitted_values[current_res_ii]).append((t1_time, fit_error))
                    
                    # Plot?
                    if plot_for_this_many_seconds != 0.0:
                        # Get trace data using the fitter's function and acquired values.
                        fit_curve = exponential_decay(
                            t         = delay_arr_values,
                            T1        = optimal_vals_x[0],
                            y_excited = optimal_vals_x[1],
                            y_ground  = optimal_vals_x[2]
                        )
                        plt.plot(delay_arr_values, current_trace_to_fit, color="#034da3")
                        plt.plot(delay_arr_values, fit_curve, color="#ef1620")
                        plt.title('Energy relaxation from the excited state')
                        plt.ylabel('Demodulated amplitude [FS]')
                        plt.xlabel('Delay after the initial π pulse [s]')
                        
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
                    t1_time = optimal_vals_x[0]
                    fit_error = fit_err_x[0]
                    
                    # Print result.
                    if the_user_provided_a_list_of_files:
                        print("T1 energy relaxation decay fit failure! Cannot fit: "+str(raw_data_or_path_to_data))
                    else:
                        print("T1 energy relaxation decay fit failure! Cannot fit the provided raw data.")
                    
                    # Store failed fit and its failed plusminus error bar.
                    ## TODO this part here has probably not been protected against usage cases with multiplexed readouts.
                    (fitted_values[current_res_ii]).append((t1_time, fit_error))
        
        # Append!
        list_of_fitted_values.append( fitted_values )
    
    # We're done.
    return list_of_fitted_values

def exponential_decay(
    t,
    T1,
    y_excited,
    y_ground
    ):
    ''' Function to be fitted against.
    '''
    return (y_excited - y_ground) * np.exp(-t / T1) + y_ground


def fit_decay(delays, decaying_data):
    ''' Grab submitted data and perform a fit to find the exponential decay.
    '''
    
    # Set time t to "delays" and get total duration.
    t = delays
    duration_of_experiment = t[-1] - t[0]
    
    # Find a good first guess for T1.
    # 0.36787944117 = e^(t/T1) where t = T1.
    # Thus 0.367879... of the y_max - y_min is a decent initial guess.
    y_excited = decaying_data[0]
    y_ground  = decaying_data[-1]
    pop_val_for_likely_T1 = 0.36787944117 * (y_excited - y_ground) + y_ground
    
    # Get index for value closest to to this y value for the to-be-guessed T1
    T1 = delays[np.abs(decaying_data - pop_val_for_likely_T1).argmin()]
    
    # Perform fit with decaying_cosine_function as the model function,
    # and the initial guess of parameters as seen below.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = exponential_decay,
        xdata = t,
        ydata = decaying_data,
        p0    = (T1, y_excited, y_ground)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    return optimal_vals, fit_err