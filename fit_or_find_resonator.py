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

#  This fitting script follows the theory outlined by Keegan Mullins       ##
#  See this thesis here: https://scholar.colorado.edu/downloads/q237ht07m  ##


def fit_or_find_resonator_dip_or_lorentzian(
    data_or_filepath_to_data,
    pulse_freq_arr = [],
    i_provided_a_filepath = True,
    i_renamed_the_pulse_freq_arr_to = '',
    plot_for_this_many_seconds = 0.0,
    number_of_times_to_filter_noisy_raw_curve = 10,
    report_lowest_resonance_point_of_filtered_curve = True
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
        provided a filepath (to data) that is to be fitted bu the code,
        or whether you provided raw data straight away.
        
        report_lowest_resonance_point_of_filtered_curve will set whether
        the reported value is from a Lorentzian fit, which usually
        is in fact a pretty bad estimate, or whether to instead report
        to the user what is the lowest point of the (filtered) resonator
        trace. The latter of which is usually adequate.
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
            assert isinstance(data_or_filepath_to_data, str), "Error: the Lorentzian resonator fitter was provided a non-string datatype, and/or a list whose first element was a non-string datatype. Expected string (filepath to data)."
        with h5py.File(os.path.abspath(data_or_filepath_to_data), 'r') as h5f:
            extracted_data = h5f["processed_data"][()]
            
            if i_renamed_the_pulse_freq_arr_to == '':
                pulse_freq_arr_values = h5f[ "readout_pulse_freq_arr" ][()]
            else:
                pulse_freq_arr_values = h5f[i_renamed_the_pulse_freq_arr_to][()]
            
            # Note! Multiplexed functionality disabled for now. TODO
        
    else:
        # Then the user provided the data raw.
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the T2_echo decay fitter was provided a string type. Expected raw data. The provided variable was: "+str(data_or_filepath_to_data)
        
        # Catch bad user inputs.
        type_and_length_is_safe = False
        try:
            if len(pulse_freq_arr) != 0:
                type_and_length_is_safe = True
        except TypeError:
            pass
        assert type_and_length_is_safe, "Error: the T2_echo decay fitter was provided raw data to fit, but the data for the delay time array could not be used for fitting the data. The argument \"pulse_freq_arr\" was: "+str(pulse_freq_arr)
        
        # Assert fittable data.
        assert len(pulse_freq_arr) == len((data_or_filepath_to_data[0])[0]), "Error: the user-provided raw data does not have the same length as the provided time delay (array) data."
        
        # Accept cruel fate and move on.
        extracted_data = data_or_filepath_to_data
        pulse_freq_arr_values = pulse_freq_arr
    
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
    # making up a resonator Lorentzian shape thing.
    
    # Report start!
    if i_provided_a_filepath:
        print("Performing Lorentzian resonance fitting on " + data_or_filepath_to_data + "...")
    else:
        print("Commencing Lorentzian resonance fitting provided raw data...")
    
    # There may be multiple resonators involved (a multiplexed readout).
    # Grab every trace (as in, a bias sweep will have many traces),
    # fit the trace, and store the result as a tuple of (value, error).
    fitted_values = [[]] * len(mag_vals_matrix)
    
    for current_res_ii in range(len( mag_vals_matrix )):
        # Note! See above comment on multiplexed sweeps being disabled for now.
        ## # Select current sweep values for this particular resonator.
        ## curr_pulse_freq_arr_values = pulse_freq_arr_values[current_res_ii]
        for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
            
            # Get current trace.
            current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
            
            # Try to fit current trace.
            try:
                optimal_vals, fit_err, fit_curve = fit_lorentzian(
                    frequencies = pulse_freq_arr_values,
                    datapoints  = current_trace_to_fit,
                    no_filterings = number_of_times_to_filter_noisy_raw_curve,
                    report_lowest_resonance_point_of_filtered_curve = report_lowest_resonance_point_of_filtered_curve
                )
                
                # Grab fitted values. The x0 gives the resonator dip, hopefully.
                resonator_dip     = optimal_vals[0]
                fit_or_find_error = fit_err[0]
                
                # Print result.
                if (not report_lowest_resonance_point_of_filtered_curve):
                    print("Resonator frequency from Lorentzian fit of data: " + str(resonator_dip) + " ±" + str(fit_or_find_error/2))
                else:
                    print("Lowest point of filtered resonator data: " + str(resonator_dip) + " ±" + str(fit_or_find_error/2))
                
                # Store fit and its plusminus error bar.
                (fitted_values[current_res_ii]).append((resonator_dip, fit_or_find_error/2))
                
                # Plot?
                if plot_for_this_many_seconds != 0.0:
                    # Get trace data using the fitter's function and acquired values.
                    plt.plot(pulse_freq_arr_values, current_trace_to_fit, color="#034da3")
                    plt.plot(pulse_freq_arr_values, fit_curve, color="#ef1620")
                    plt.title('Resonator spectroscopy')
                    plt.ylabel('Demodulated amplitude [FS]')
                    plt.xlabel('Stimulus tone frequency [Hz]')
                    
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
                resonator_dip     = optimal_vals_x[0]
                fit_or_find_error = fit_err_x[0]
                
                # Print result.
                if i_provided_a_filepath:
                    print("Resonator fit failure! Cannot fit: "+str(data_or_filepath_to_data))
                else:
                    print("Resonator fit failure! Cannot fit the provided raw data.")
                
                # Store failed fit and its failed plusminus error bar.
                (fitted_values[current_res_ii]).append((resonator_dip, fit_or_find_error/2))
    
    # We're done.
    return fitted_values

def fit_lorentzian(
    frequencies,
    datapoints,
    no_filterings,
    report_lowest_resonance_point_of_filtered_curve
    ):
    ''' Grab submitted data of a resonator spectroscopy run, and perform a
        Lorentzian fit to find the resonator's dip along the frequency axis.
    '''
    
    # Set x-axis value x to "frequencies"
    x = frequencies
    
    # The datapoints curve will very likely be very noisy. Clean it up a bit,
    # and interpolate the curve so that the number of datapoints match.
    # This curve will only be used for grabbing estimates.
    ## FILTER!
    datapoints_to_filter = datapoints
    number_of_times_to_filter = no_filterings
    for ii in range(number_of_times_to_filter):
        filtered_datapoints = []
        if len(datapoints_to_filter) > 2:
            for curr_datapoint in range(0,len(datapoints_to_filter),2):
                next_val = (datapoints_to_filter[curr_datapoint] + datapoints_to_filter[curr_datapoint+1])/2
                filtered_datapoints.append(next_val)
            del next_val
        
        ## INTERPOLATE!
        interpolated_filtered_datapoints = []
        for curr_datapoint in range(len(filtered_datapoints)-1): # The -1 prevents the list index from going out of range.
            interpolated_filtered_datapoints.append(filtered_datapoints[curr_datapoint])
            interpolated_filtered_datapoints.append( (filtered_datapoints[curr_datapoint] + filtered_datapoints[curr_datapoint+1]) / 2 )
        # The for-loop does not catch the very last value. Set it manually,
        # as a "finally" clause. Also, what do we do with the very last
        # *interpolation*? Let's alternate whether we extend the filtered
        # curve at the beginning, or at the end, using the first (or last)
        # value (respectively). This alternation-thing puts the final,
        # smoothened curve, close to the original curve's shape.
        # Source: me, Christian. You can try to always, only, just append the
        # final filtered_datapoints-value TWICE if you like. The smoothened
        # curve will slowly drift towards the left as the number of averages
        # increases.
        if (ii % 2) == 0:
            interpolated_filtered_datapoints.append(filtered_datapoints[-1]) # Once
            interpolated_filtered_datapoints.append(filtered_datapoints[-1]) # Twice, this is not a copy-paste error, see comment above.
        else:
            interpolated_filtered_datapoints.append(filtered_datapoints[-1])
            interpolated_filtered_datapoints.insert(0,filtered_datapoints[0])
        if ii != (number_of_times_to_filter-1):
            datapoints_to_filter = interpolated_filtered_datapoints
    
    # At this point, we've made the interpolated, filtered curve of the
    # input datapoints. Let's numpy-ify this curve.
    interpolated_filtered_datapoints = np.array(interpolated_filtered_datapoints)
    
    # Get an y-value showing where (in y) the filtered curve dips.
    # An, an estimate for the y-value where the to-be-fitted Lorentzian
    # would be flat. This would be the same as our guess for the offset.
    estimate_for_y_value_at_x0 = np.min(interpolated_filtered_datapoints)
    guess_of_offset = (interpolated_filtered_datapoints[0] + interpolated_filtered_datapoints[-1]) / 2
    
    # Guess of height difference peak-to-peak of the final Lorentzian
    lorentzian_peak_to_peak = guess_of_offset - estimate_for_y_value_at_x0
    
    # The Lorentzian needs a full-width half-maximum value.
    y_value_where_fwhm_is_expected = lorentzian_peak_to_peak / 2 + estimate_for_y_value_at_x0
    crossed_fwhm_value_from_the_left  = False
    crossed_fwhm_value_from_the_right = False
    idx_where_lower_fwhm_is_expected = 0
    idx_where_upper_fwhm_is_expected = len(x)-1
    for curr_idx_looked_at in range(len(x)):
        # Does the next point in the sweep cross the fwhm boundary?
        if (not crossed_fwhm_value_from_the_left):
            if (interpolated_filtered_datapoints[curr_idx_looked_at] < y_value_where_fwhm_is_expected):
                # We crossed the boundary from the left side.
                idx_where_lower_fwhm_is_expected = curr_idx_looked_at
                crossed_fwhm_value_from_the_left = True
        if (not crossed_fwhm_value_from_the_right):
            if (interpolated_filtered_datapoints[-curr_idx_looked_at] < y_value_where_fwhm_is_expected):
                # We crossed the index from the right side.
                idx_where_upper_fwhm_is_expected = curr_idx_looked_at
                crossed_fwhm_value_from_the_right = True
    # Now, we may make an estimate for the full-width half-maximum.
    fwhm_estimate = np.abs(x[idx_where_upper_fwhm_is_expected] - x[idx_where_lower_fwhm_is_expected])
    
    # Get the x0 of the Lorentzian
    x_value_at_x0 = x[interpolated_filtered_datapoints.argmin()]
    
    # Get a "scalar" for the fitter's Lorentzian.
    # To find this value, we get a sneak-peak max-min of sought-for lorentzian.
    # Then, we scale the amplitude that we are finally going to fit against,
    # with the differences in values between the fit and the sneak-peak.
    guess_curve = lorentzian_function(
        x                       = x,
        x0                      = x_value_at_x0,
        full_width_half_maximum = fwhm_estimate,
        amplitude               = -lorentzian_peak_to_peak, # NOTE: negative amplitude.
        offset                  = guess_of_offset,
    )
    delta_x_of_guess_curve  = np.max(guess_curve) - np.min(guess_curve)
    delta_x_of_sought_curve = lorentzian_peak_to_peak
    scalar_from_first_guess = delta_x_of_sought_curve / delta_x_of_guess_curve
    lorentzian_amp = (lorentzian_peak_to_peak - guess_of_offset) * scalar_from_first_guess + guess_of_offset
    
    # Perform fit with lorentzian_function as the model function,
    # and the initial guess of parameters as seen below.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = lorentzian_function,
        xdata = x,
        ydata = interpolated_filtered_datapoints,
        p0    = (x_value_at_x0, fwhm_estimate, lorentzian_amp, guess_of_offset)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    # Now, as for what to report.
    if (not report_lowest_resonance_point_of_filtered_curve):
        fit_curve = lorentzian_function(
            x                       = x,
            x0                      = optimal_vals[0],
            full_width_half_maximum = optimal_vals[1],
            amplitude               = optimal_vals[2],
            offset                  = optimal_vals[3],
        )
    else:
        # If this clause is happening, then we'd like to ignore
        # the fit and instead grab the interpolated, filtered curve.
        # As an error bar, let's set the difference between the fit dip
        # and the bottom of the interpolated, filtered curve.
        lowest_dip = x[interpolated_filtered_datapoints.argmin()]
        fit_err[0]  = np.abs(optimal_vals[0] - lowest_dip)
        fit_err[1:] = np.float("nan")
        optimal_vals[0]  = lowest_dip
        optimal_vals[1:] = np.float("nan")
        fit_curve = interpolated_filtered_datapoints
    
    return optimal_vals, fit_err, fit_curve
    
    
def lorentzian_function(
    x,
    x0,
    full_width_half_maximum,
    amplitude,
    offset
    ):
    ''' Function to be fitted against.
    '''
    return amplitude * (1/np.pi) * (full_width_half_maximum/2) / ((x-x0)**2 + (full_width_half_maximum/2)**2) + offset
    