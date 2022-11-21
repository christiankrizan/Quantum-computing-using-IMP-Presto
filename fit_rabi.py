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

def fit_amplitude(
    data_or_filepath_to_data,
    control_amp_arr = [],
    i_provided_a_filepath = True,
    i_renamed_the_control_amp_arr_to = '',
    plot_for_this_many_seconds = 0.0
    ):
    ''' From supplied data or datapath,
        fit the Rabi oscillations to get a "period" in amplitude.
        This "amplitude period" / 2 = the π-pulse amplitude.
        
        i_provided_a_filepath sets whether you as a user
        provided a filepath (to data) that is to be fitted bu the code,
        or whether you provided raw data straight away.
    '''
    
    ## TODO: Somehow, add support so that the user can provide themselves
    ##       raw data for multiplexed rabi sweeps too.
    
    ## TODO: Solve the multiplexed sweep issue. All multiplexed sweeps
    ##       where the x-axis is not identical for all qubits+resonators,
    ##       will (likely?) have to be exported seperately.
    ##       Meaning that this fitter will output some gibberish.
    ##       Perhaps a "select resonator" function is necessary?
    
    # The Rabi oscillation could have been done with a multiplexed readout.
    # Thus, we declare a list for storing what amplitude sweeps
    # were done for the different qubits.
    control_amp_values = []
    
    # Get data.
    if i_provided_a_filepath:
        # The user provided a filepath to data.
        if (not isinstance(data_or_filepath_to_data, str)):
            data_or_filepath_to_data = data_or_filepath_to_data[0]
            assert isinstance(data_or_filepath_to_data, str), "Error: the Rabi amplitude fitter was provided a non-string datatype, and/or a list whose first element was a non-string datatype. Expected string (filepath to data)."
        with h5py.File(os.path.abspath(data_or_filepath_to_data), 'r') as h5f:
            extracted_data = h5f["processed_data"][()]
            
            if i_renamed_the_control_amp_arr_to == '':
                control_amp_values = h5f[ "control_amp_arr" ][()]
            else:
                control_amp_values = h5f[i_renamed_the_control_amp_arr_to][()]
            
            # Note! Multiplexed functionality disabled for now, since the
            # split files do not contain both control_amp_arr_A and
            # control_amp_arr_B.
            ##found_control_data = False
            ##for try_this in ['control_amp_arr', 'control_amp_arr_A', 'control_amp_arr_B']:
            ##    try:
            ##        if i_renamed_the_control_amp_arr_to == '':
            ##            control_amp_values.append( h5f[ try_this ][()] )
            ##        else:
            ##            control_amp_values.append( h5f[i_renamed_the_control_amp_arr_to][()] )
            ##        found_control_data = True
            ##    except KeyError:
            ##        # Then do nothing.
            ##        pass
            ##if (not found_control_data):
            ##    raise KeyError("Error: could not find any control port amplitude data in the file. Try passing the name of the control amp. arr. as an appropriate argument to the fit_rabi fitting function.")
    else:
        # Then the user provided the data raw.
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the Rabi amplitude fitter was provided a string type. Expected raw data. The provided variable was: "+str(data_or_filepath_to_data)
        
        # Catch bad user inputs.
        type_and_length_is_safe = False
        try:
            if len(control_amp_arr) != 0:
                type_and_length_is_safe = True
        except TypeError:
            pass
        assert type_and_length_is_safe, "Error: the Rabi fitter was provided raw data to fit, but the data for the control port amplitude could not be used for fitting the data. The argument \"control_amp_arr\" was: "+str(control_amp_arr)
        
        # Assert fittable data.
        assert len(control_amp_arr) == len((data_or_filepath_to_data[0])[0]), "Error: the user-provided raw data does not have the same length as the provided control amplitude data."
        
        # Accept cruel fate and move on.
        extracted_data = data_or_filepath_to_data
        control_amp_values = control_amp_arr
    
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
    # making up Rabi oscillations.
    
    # Report start!
    if i_provided_a_filepath:
        print("Performing Rabi fitting on " + data_or_filepath_to_data + "...")
    else:
        print("Commencing Rabi fitting of provided raw data...")
    
    # There may be multiple resonators involved (a multiplexed readout).
    # Grab every trace (as in, a bias sweep will have many traces),
    # fit the trace, and store the result as a tuple of (value, error).
    fitted_values = [[]] * len(mag_vals_matrix)
    
    for current_res_ii in range(len( mag_vals_matrix )):
        # Note! See above comment on multiplexed sweeps being disabled for now.
        ## # Select current sweep values for this particular resonator.
        ## curr_control_amp_values = control_amp_values[current_res_ii]
        for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
            
            # Get current trace.
            current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
            
            # Try to fit current trace.
            try:
                optimal_vals_x, fit_err_x = fit_periodicity(
                    x = control_amp_values,
                    y = current_trace_to_fit
                )
                ##optimal_vals_x, fit_err_x = fit_periodicity(
                ##    x = curr_control_amp_values,
                ##    y = current_trace_to_fit
                ##)
                
                # Grab fitted values.
                one_rabi_cycle_period = optimal_vals_x[3]
                fit_error = fit_err_x[3]
                pi_amplitude = one_rabi_cycle_period / 2
                
                # Print result.
                print("Rabi fit of data: " + str(pi_amplitude) + " ±" + str(fit_error/2))
                
                # Store fit and its plusminus error bar.
                (fitted_values[current_res_ii]).append((pi_amplitude, fit_error/2))
                
                # Plot?
                if plot_for_this_many_seconds != 0.0:
                    # Get trace data using the fitter's function and acquired values.
                    fit_curve = decaying_cosine_function(
                        t          = control_amp_values,
                        y_offset   = optimal_vals_x[0],
                        amplitude  = optimal_vals_x[1],
                        decay_rate = optimal_vals_x[2],
                        period     = optimal_vals_x[3],
                        phase      = optimal_vals_x[4]
                    )
                    plt.plot(control_amp_values, current_trace_to_fit, color="#034da3")
                    plt.plot(control_amp_values, fit_curve, color="#ef1620")
                    plt.title('Rabi oscillations (amplitude sweep)')
                    plt.ylabel('Demodulated readout amplitude [FS]')
                    plt.xlabel('π-pulse amplitude [FS]')
                    
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
                one_rabi_cycle_period = optimal_vals_x[3]
                fit_error = fit_err_x[3]
                pi_amplitude = one_rabi_cycle_period / 2
                
                # Print result.
                if i_provided_a_filepath:
                    print("Rabi oscillation fit failure! Cannot fit: "+str(data_or_filepath_to_data))
                else:
                    print("Rabi oscillation fit failure! Cannot fit the provided raw data.")
                
                # Store failed fit and its failed plusminus error bar.
                (fitted_values[current_res_ii]).append((pi_amplitude, fit_error/2))
    
    # We're done.
    return fitted_values
    
def decaying_cosine_function(
    t,
    y_offset,
    amplitude,
    decay_rate,
    period,
    phase
    ):
    ''' Function to be fitted against.
    '''
    return amplitude * np.cos(2*np.pi*(1/period) * t + phase) * np.exp(-t/decay_rate) + y_offset

def fit_periodicity(x, y):
    ''' Grab submitted data and perform a fit to find the periodicity.
        Where: amplitude, y_offset, decay_rate, period and starting_phase
        are initial guesses to the function parameters.
    '''
    # What is the current amplitude and y_offset?
    amplitude = (np.max(y)-np.min(y)) / 2
    y_offset  = np.min(y) + amplitude
    
    # Make an estimate of the exponential decay time.
    decay_rate = (np.max(x)-np.min(x)) / 2
    
    # What is the cosine's resolution? = Delta t, but where time val. t = x
    delta_x = x[1]-x[0]
    
    # What is the Rabi periodicity? Make discrete FFT, select discr. value
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
        f     = decaying_cosine_function,
        xdata = x,
        ydata = y,
        p0    = (y_offset, amplitude, decay_rate, period, starting_phase)
    )
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    return optimal_vals, fit_err