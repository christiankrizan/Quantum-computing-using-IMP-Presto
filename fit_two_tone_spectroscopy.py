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
from math import isnan, sqrt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from smoothener import filter_and_interpolate

def fit_resonance_peak(
    data_or_filepath_to_data,
    control_freq_arr = [],
    coupler_amp_arr  = [],
    i_provided_a_filepath = True,
    i_renamed_the_control_freq_arr_to = '',
    i_renamed_the_coupler_amp_arr_to  = '',
    plot_for_this_many_seconds = 0.0,
    number_of_times_to_filter_noisy_raw_curve = 10,
    ):
    ''' From supplied data or datapath, fit a Lorentzian to find the
        dip of the resonance curve bottom. The goal is to extract
        a frequency to use for the res_f. Do note that there are more accurate
        fitting methods than just using a Lorentzian.
        
        A failed fit (due to illegibly noisy input, for instance)
        will return a NaN ±NaN result.
        
        plot_for_this_many_seconds will show a plot for a given number of
        seconds. If provided a negative value, then the plot will remain
        "blocking" until closed by the user.
        
        i_provided_a_filepath sets whether you as a user
        provided a filepath (to data) that is to be fitted by the code,
        or whether you provided raw data straight away.
    '''
    
    # The resonator spectroscopy experiment could have been done with a
    # multiplexed readout. Thus, we declare a list for storing different
    # traces.
    frequency_values = []
    
    # Get data.
    if i_provided_a_filepath:
        # The user provided a filepath to data.
        if (not isinstance(data_or_filepath_to_data, str)):
            data_or_filepath_to_data = data_or_filepath_to_data[0]
            assert isinstance(data_or_filepath_to_data, str), "Error: the two-tone spectroscopy fitter was provided a non-string datatype, and/or a list whose first element was a non-string datatype. Expected string (filepath to data)."
        with h5py.File(os.path.abspath(data_or_filepath_to_data), 'r') as h5f:
            extracted_data = h5f["processed_data"][()]
            
            if i_renamed_the_control_freq_arr_to == '':
                control_freq_arr_values = h5f[ "control_pulse_01_freq_arr" ][()]
            else:
                control_freq_arr_values = h5f[i_renamed_the_control_freq_arr_to][()]
            
            try:
                if i_renamed_the_coupler_amp_arr_to == '':
                    coupler_amp_arr_values = h5f[ "coupler_amp_arr" ][()]
                else:
                    coupler_amp_arr_values = h5f[i_renamed_the_coupler_amp_arr_to][()]
            except KeyError:
                # This data file does not contain any information about
                # a coupler sweep. We'll ignore this for now.
                pass
            
            # Note! Multiplexed functionality disabled for now. TODO
        
    else:
        ## TODO Add catches and checks for the coupler_amp_arr too.
        
        # Then the user provided the data raw.
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the two-tone spectroscopy fitter was provided a string type. Expected raw data. The provided variable was: "+str(data_or_filepath_to_data)
        
        # Catch bad user inputs.
        type_and_length_is_safe = False
        try:
            if len(control_freq_arr) != 0:
                type_and_length_is_safe = True
        except TypeError:
            pass
        assert type_and_length_is_safe, "Error: the two-tone spectroscopy fitter was provided raw data to fit, but the data for the frequency array could not be used for fitting the data. The argument \"control_freq_arr\" was: "+str(control_freq_arr)
        
        # Assert fittable data.
        assert len(control_freq_arr) == len((data_or_filepath_to_data[0])[0]), "Error: the user-provided raw data does not have the same length as the provided frequency (array) data."
        
        # Accept cruel fate and move on.
        extracted_data = data_or_filepath_to_data
        control_freq_arr_values = control_freq_arr
    
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
    # making up a two-tone spectroscopy kind-of curve.
    
    # Report start!
    if i_provided_a_filepath:
        print("Performing resonance peak fitting on " + data_or_filepath_to_data + "...")
    else:
        print("Commencing resonance peak fitting on provided raw data...")
    
    # There may be multiple resonators involved (a multiplexed readout).
    # Grab every trace (as in, a bias sweep will have many traces),
    # fit the trace, and store the result as a tuple of (value, error).
    fitted_values = [[]] * len(mag_vals_matrix)
    
    for current_res_ii in range(len( mag_vals_matrix )):
        # Note! See above comment on multiplexed sweeps being disabled for now.
        ## # Select current sweep values for this particular resonator.
        ## curr_control_freq_arr_values = control_freq_arr_values[current_res_ii]
        for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
            
            # Get current trace.
            current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
            
            # Filter and interpolate the trace?
            if number_of_times_to_filter_noisy_raw_curve > 0:
                current_trace_to_fit = filter_and_interpolate(
                    datapoints_to_filter_and_interpolate = current_trace_to_fit,
                    number_of_times_to_filter_and_interpolate = number_of_times_to_filter_noisy_raw_curve
                )
            
            # Try to fit current trace.
            try:
                optimal_vals, fit_err, fit_curve = fit_gaussian(
                    frequencies = control_freq_arr_values,
                    datapoints  = current_trace_to_fit,
                )
                
                # Grab fitted values. The x0 gives the resonator dip, hopefully.
                resonator_peak = optimal_vals[1]
                fit_error      = fit_err[1]
                
                # Print result.
                print("Gaussian fit of two-tone spectroscopy data: " + str(resonator_peak) + " ±" + str(fit_error/2))
                
                # Store fit and its plusminus error bar.
                (fitted_values[current_res_ii]).append((resonator_peak, fit_error/2))
                
                # Plot?
                if plot_for_this_many_seconds != 0.0:
                    # Get trace data using the fitter's function and acquired values.
                    plt.plot(control_freq_arr_values, current_trace_to_fit, color="#034da3")
                    plt.plot(control_freq_arr_values, fit_curve, color="#ef1620")
                    plt.title('Two-tone spectroscopy')
                    plt.ylabel('Demodulated amplitude [FS]')
                    plt.xlabel('Control tone frequency [Hz]')
                    
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
                optimal_vals  = [float("nan"), float("nan"), float("nan")]
                fit_err       = [float("nan"), float("nan"), float("nan")]
                
                # Grab fitted values.
                resonator_peak = optimal_vals[1]
                fit_error      = fit_err[1]
                
                # Print result.
                if i_provided_a_filepath:
                    print("Two-tone spectroscopy fit failure! Cannot fit: "+str(data_or_filepath_to_data))
                else:
                    print("Two-tone spectroscopy fit failure! Cannot fit the provided raw data.")
                
                # Store failed fit and its failed plusminus error bar.
                (fitted_values[current_res_ii]).append((resonator_peak, fit_error/2))
    
    # We're done.
    return fitted_values

def fit_gaussian( frequencies, datapoints ):
    ''' Grab submitted data of a resonator spectroscopy run, and perform a
        Gaussian fit to find the resonator's peak along the frequency axis.
    '''
    
    # Set x-axis value x to "frequencies"
    x = frequencies
    
    # Get a first guess of the y-offset.
    offset_guess = ( datapoints[0] + datapoints[-1] )/2
    
    # Get a guess of where the frequency peak is supposed to be located.
    # Probably, at the highest peak of the data.
    mu_guess = x[np.abs(datapoints - np.max(datapoints)).argmin()]
    
    # Get a guess of the standard deviation for the curve to be fitted.
    ## We have
    ## num = 2.5066282746310002
    ## e = 2.718281828459045
    ## offset = [known]
    ## g(x) = (1 / (sigma * num)) * (e)^((-(x-mu)^2)/(2*sigma^2)) + offset
    ##
    ## Assume x = mu, a point where we know both g(x) and x.
    ## Assume sigma != 0
    ## [known] - offset = (1 / (sigma * num)) * 1
    ## 1 / (( [known] - offset ) * num) = sigma, under these assumptions
    ## Remember, [known] is g(x) where x = mu. So, the maximum value.
    sigma_guess = 1 / (( np.max(datapoints) - offset_guess ) * 2.5066282746310002) 
    
    # Guess scalar of curve. Let's just pick something.
    scalar_guess = 1.0
    
    # Perform fit with gaussian_function as the model function,
    # and the initial guess of parameters as seen below.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = gaussian_function,
        xdata = x,
        ydata = datapoints,
        p0    = (sigma_guess, mu_guess, scalar_guess, offset_guess)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    fit_curve = gaussian_function(
        x      = x,
        sigma  = optimal_vals[0],
        mu     = optimal_vals[1],
        scalar = optimal_vals[2],
        offset = optimal_vals[3]
    )
    
    return optimal_vals, fit_err, fit_curve
    
def gaussian_function(
    x,
    sigma,
    mu,
    scalar,
    offset
    ):
    ''' Function to be fitted against.
        The value 2.5066282746310002 = sqrt( 2 * pi ).
        sigma is the Gaussian curve's standard deviation.
        mu is the Gaussian curve's expected value.
    '''
    return scalar * ((1 / (sigma * 2.5066282746310002)) * (2.718281828459045)**((-(x-mu)**2)/(2*sigma**2))) + offset