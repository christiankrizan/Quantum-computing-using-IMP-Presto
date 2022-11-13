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

def fit_oscillating_exponential_decay_t2_asterisk(
    data_or_filepath_to_data,
    delay_arr = [],
    i_provided_a_filepath = True,
    i_renamed_the_delay_arr_to = '',
    plot_fit_for_these_many_seconds = 0.0
    ):
    ''' From supplied data or datapath, fit the qubit dephasing decay
        to find out the exponent of said dephasing.
        
        The goal is to extract T2*
    '''
    
    ## TODO: Somehow, add support so that the user can provide themselves
    ##       raw data for multiplexed T2* experiments too.
    
    ## TODO: Solve the multiplexed sweep issue. All multiplexed sweeps
    ##       where the x-axis is not identical for all qubits+resonators,
    ##       will (likely?) have to be exported seperately.
    ##       Meaning that this fitter will output some gibberish.
    ##       Perhaps a "select resonator" function is necessary?
    
    # The T2* dephasing experiment could have been done with a multiplexed
    # readout. Thus, we declare a list for storing what time sweeps
    # were done for the different qubits.
    delay_arr_values = []
    
    # Get data.
    if i_provided_a_filepath:
        # The user provided a filepath to data.
        if (not isinstance(data_or_filepath_to_data, str)):
            data_or_filepath_to_data = data_or_filepath_to_data[0]
            assert isinstance(data_or_filepath_to_data, str), "Error: the T2* dephasing fitter was provided a non-string datatype, and/or a list whose first element was a non-string datatype. Expected string (filepath to data)."
        with h5py.File(os.path.abspath(data_or_filepath_to_data), 'r') as h5f:
            extracted_data = h5f["processed_data"][()]
            
            if i_renamed_the_delay_arr_to == '':
                delay_arr_values = h5f[ "delay_arr" ][()]
            else:
                delay_arr_values = h5f[i_renamed_the_delay_arr_to][()]
            
            # Note! Multiplexed functionality disabled for now. TODO
    
    else:
        # Then the user provided the data raw.
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the T2* dephasing fitter was provided a string type. Expected raw data. The provided variable was: "+str(data_or_filepath_to_data)
        
        # Catch bad user inputs.
        type_and_length_is_safe = False
        try:
            if len(delay_arr) != 0:
                type_and_length_is_safe = True
        except TypeError:
            pass
        assert type_and_length_is_safe, "Error: the T2* dephasing fitter was provided raw data to fit, but the data for the delay time array could not be used for fitting the data. The argument \"delay_arr\" was: "+str(delay_arr)
        
        # Assert fittable data.
        assert len(delay_arr) == len((data_or_filepath_to_data[0])[0]), "Error: the user-provided raw data does not have the same length as the provided time delay (array) data."
        
        # Accept cruel fate and move on.
        extracted_data = data_or_filepath_to_data
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
    # making up a T2* dephasing oscillatory decay.
    
    # Report start!
    if i_provided_a_filepath:
        print("Performing T2* dephasing oscillatory decay fitting on " + data_or_filepath_to_data + "...")
    else:
        print("Commencing T2* dephasing oscillatory decay fitting of provided raw data...")
    
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
            
            # Fit current trace.
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
            
            # Print result.
            print("T2* from exponential decay fit of data: " + str(t2_asterisk_time) + " ±" + str(fit_error/2))
            
            # Store fit and its plusminus error bar.
            (fitted_values[current_res_ii]).append((t2_asterisk_time, fit_error/2))

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
    # 0.36787944117 ~= e^(t/T2*) [plus offset and squiggles] where t = T2*.
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