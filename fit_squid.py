#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import os
import h5py
import numpy as np
import random
from numpy import hanning as von_hann
from math import isnan, sqrt, cos
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from fit_input_checker import verify_supported_input_argument_to_fit
##from smoothener import filter_and_interpolate

def plot_legal_bias_point_intervals(
    resonator_start_stop = [ [(0.0, 0.0), (0.0, 0.0)] ],
    qubit_start_stop     = [ [(0.0, 0.0), (0.0, 0.0)] ],
    plot_for_this_many_seconds = 0.0,
    x_axis_unit = 'A'
    ):
    ''' Plot a bar graph showing all intervals where
        a SQUID pairwise coupler, as a function of threading magnetostatic
        flux, is not particularly impacting the resonator or qubit
        frequencies.
        
        Syntax:
        resonator_start_stop:  list of intervals[  (tuples of: start,stop)  ]
        qubit_start_stop:      list of intervals[  (tuples of: start,stop)  ]
        
        Feeding a negative number to plot_for_this_many_seconds,
        will instead block the plot display until the user closes that window.
    '''
    
    # Make canvas.
    plt.title('Intervals with low frequency shifts per coupler bias')
    plt.xlabel('Applied bias into coupler line ['+str(x_axis_unit)+']')
    
    # Make names for ticks.
    y_ticks_list = []
    
    # Create axis iterators.
    axis_iterator = 0
    
    # Define colour space.
    colours = []
    colour_counter = 0 # Set counter for later, used when plt.plotting.
    for hh in range( len(resonator_start_stop) + len(qubit_start_stop) ):
        # Note that 0xffffff is removed from the set. No colour may be white.
        colours.append( \
            str(hex(random.randint(0, (256**3)-2))).replace('0x',''))
        # Pad hex string!
        while len(colours[-1]) < 6:
            colours[-1] = '0' + colours[-1]
        # Append suitable hex character.
        colours[-1] = '#' + colours[-1]
    
    # Add resonators.
    resonator_counter = 1
    for item_ii in range(len(resonator_start_stop)):
        for interval_tuple_jj in resonator_start_stop[ item_ii ]:
            plt.plot( \
                interval_tuple_jj, \
                (-1*(item_ii+axis_iterator),-1*(item_ii+axis_iterator)), \
                color = colours[ colour_counter ] \
            )
        # The next resonator is plotted in other colours.
        colour_counter += 1
        
        # Append resonator name to y-axis list (actual axis added later).
        y_ticks_list.append('Resonator '+str(resonator_counter))
        resonator_counter += 1
    
    # Keep track of where on the canvas we'll draw a new line.
    axis_iterator += len(resonator_start_stop)
    
    # Add qubits.
    qubit_counter = 1
    for item_ii in range(len(qubit_start_stop)):
        for interval_tuple_jj in qubit_start_stop[ item_ii ]:
            plt.plot( \
                interval_tuple_jj, \
                (-1*(item_ii+axis_iterator),-1*(item_ii+axis_iterator)), \
                color = colours[ colour_counter ] \
            )
        
        # The next qubit is plotted in other colours.
        colour_counter += 1
        
        # Append qubit name to y-axis list (actual axis added later).
        y_ticks_list.append('Qubit '+str(qubit_counter))
        qubit_counter += 1
    
    # Keep track of where on the canvas we'll draw a new line.
    axis_iterator += len(qubit_start_stop)
    
    # Make y-axis ticks.
    plt.yticks( \
        -1 * np.arange( len(resonator_start_stop) + len(qubit_start_stop)), \
        y_ticks_list
    )
    
    # Plot?
    if plot_for_this_many_seconds != 0.0:
        # If inserting a positive time for which we want to plot for,
        # then plot for that duration of time. If given a negative
        # time, then instead block the plotted display.
        if plot_for_this_many_seconds > 0.0:
            plt.show(block=False)
            plt.pause(plot_for_this_many_seconds)
            plt.close()
        else:
            plt.show(block=True)

def find_coupler_frequency_from_avoided_two_level_crossings(
    known_frequency_vector,
    known_magnetostatic_flux_vector,
    initial_guess_of_phi_0 = 0.0,
    initial_guess_of_static_flux_offset = 0.0,
    plot_for_this_many_seconds = 0.0,
    flux_quantum_unit = "[unit]"
    ):
    ''' For this function in its current form, the user must know the
        periodicity of the 
    '''
    # TODO: at some point in the future, it would be very nice if
    #       this routine could simply be fed with a filepath, and the
    #       frequency+flux vectors could simply be extracted from that file.
    
    # Report start!
    print("Performing fit of coupler frequency vs. DC flux data.")
    
    # Begin with a numpy-ification of the input vectors. Make them into x, y.
    # Also, sort them nicely, assuming x will be sorted.
    x = np.array(known_magnetostatic_flux_vector)
    y = np.array(known_frequency_vector)
    y = y[x.argsort()]
    x.sort()
    
    # For the fitting routine, we need to estimate the following parameters:
    # → phi_0,
    # → freq_of_untuned_coupler,
    # → static_flux_offset,
    
    # Make an estimate of the phi_0.
    # Note that phi_0 will have the same unit as the user is entering in phi.
    if initial_guess_of_phi_0 == 0.0:
        phi_0 = 2 * x[0]
    else:
        phi_0 = initial_guess_of_phi_0
    
    # Make an estimate of the coupler frequency.
    # The assumed order of the coupler frequency is 2 * qubit frequency.
    freq_of_untuned_coupler = 2 * y[0]
    
    # Make an estimate of the initial static flux. Ideally, this value is zero.
    if initial_guess_of_static_flux_offset != 0.0:
        static_flux_offset = initial_guess_of_static_flux_offset
    else:
        static_flux_offset = 0.0
    
    # Make a vector for storing fitted values.
    fitted_values = []
    
    # Try to fit current trace.
    try:
        # Perform fit with tunable_coupler_frequency as the model function,
        # and the initial guess of parameters as seen below.
        optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
            f     = tunable_coupler_frequency_function,
            xdata = x,
            ydata = y,
            p0    = (phi_0, freq_of_untuned_coupler, static_flux_offset)
        )
        
        # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
        # These values are optimised for minimising (residuals)^2 of
        # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
        fit_error = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
        
        # Report fitted values.
        print("Frequency of coupler: " + str(optimal_vals[1]) + " Hz ±" + str(fit_error[1]) + " Hz")
        print("Phi₀: " + str(optimal_vals[0]) + " "+flux_quantum_unit+" ±" + str(fit_error[0]) + " "+flux_quantum_unit)
        print("Static flux offset: " + str(optimal_vals[2]) + " "+flux_quantum_unit+" ±" + str(fit_error[2]) + " "+flux_quantum_unit)
        
        # Store fit and its plusminus error bar.
        for ii in range(len(optimal_vals)):
            fitted_values.append((optimal_vals[ii], fit_error[ii]))
        
        # Plot?
        if plot_for_this_many_seconds != 0.0:
            
            # Get plotting limits, try to make them symmetric about 0.0
            low_plot_lim  = x[0] - np.abs(x[1])
            high_plot_lim = x[-1] - np.abs(x[-2])
            if np.abs(low_plot_lim) > np.abs(high_plot_lim):
                # The lower limit is further away from 0. Adjust upper limit.
                high_plot_lim = np.sign(high_plot_lim) * np.abs(low_plot_lim)
            else:
                # The upper limit is further away from 0. Adjust lower limit.
                low_plot_lim = np.sign(low_plot_lim) * np.abs(high_plot_lim)
            
            # Get trace data using the fitter's function and acquired values.
            fit_plot_x_vector = np.linspace(low_plot_lim, high_plot_lim, 200)
            fit_curve = tunable_coupler_frequency_function(
                phi                     = fit_plot_x_vector,
                phi_0                   = optimal_vals[0],
                freq_of_untuned_coupler = optimal_vals[1],
                static_flux_offset      = optimal_vals[2],
            )
            
            # Set figure size, and font sizes.
            plt.figure(figsize = (15,12))
            font = {'family' : 'Arial',
                    'weight' : 'normal',
                    'size'   : 21}
            plt.rc('font', **font)
            
            # Plot!
            ## When plotting, rescale the x-axis to fit the fitted phi_0!
            plt.plot(fit_plot_x_vector / optimal_vals[0], fit_curve, linewidth = 3.0, label="Period: "+str(optimal_vals[0])+" "+flux_quantum_unit+" ±"+str(fit_error[0])+" "+flux_quantum_unit, color="#034da3")
            plt.plot(x / optimal_vals[0], y, 'o', label="f_cpl: "+str(optimal_vals[1])+" Hz ±"+str(fit_error[1])+" Hz", color="#ef1620")
            plt.title('Coupler DC flux periodicity', fontsize = 35, pad = 20)
            plt.ylabel('Frequency [Hz]', fontsize = 24)
            plt.xlabel('DC flux [Φ/Φ_0]', fontsize = 24)
            plt.legend()
            
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
        fit_error     = [float("nan"), float("nan"), float("nan")]
        
        # Grab fitted values.
        for ii in range(len(optimal_vals)):
            fitted_values.append((optimal_vals[ii], fit_error[ii]))
        
        # Print result.
        print("Coupler periodicity fit failure!")
    
    # We're done.
    return fitted_values
    
def tunable_coupler_frequency_function(
    phi,
    phi_0,
    freq_of_untuned_coupler,
    static_flux_offset,
    ):
    ''' The formula below is a variant of (2) in McKay et al. 2016,
        https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.6.064007
        
        The static flux offset, takes into account that there was some
        residual flux threading the coupler SQUID, that was locked in place
        when the SQUID became superconducting.
    '''
    return np.abs(freq_of_untuned_coupler) * np.sqrt( np.abs( np.cos( np.pi * (static_flux_offset + phi)/phi_0 ) ) )



def extract_coupling_factor_from_coupler_bias_sweep(
    raw_data_or_path_to_data,
    coupler_amp_arr = [],
    i_renamed_the_coupler_amp_arr_to = '',
    plot_for_this_many_seconds = 0.0
    ):
    ''' From supplied data or datapath, acquire the resonator or qubit
        coupling factors.
        
        A failed fit (due to illegibly noisy input, for instance)
        will return a NaN ±NaN result.
        
        The coupling factor is extracted using the canonical approach,
        illustrated for instance Fig. (9) here:
        https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.024070
        
        Let it be known that there are other methods.
        
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
                
                if i_renamed_the_coupler_amp_arr_to == '':
                    coupler_amp_arr_values = h5f[ "coupler_amp_arr" ][()]
                else:
                    coupler_amp_arr_values = h5f[i_renamed_the_coupler_amp_arr_to][()]
        
        else:
            # Then the user provided the data raw.
            
            # Catch bad user inputs.
            type_and_length_is_safe = False
            try:
                if len(coupler_amp_arr) != 0:
                    type_and_length_is_safe = True
            except TypeError:
                pass
            assert type_and_length_is_safe, "Error: the coupler bias sweep fitter was provided raw data to fit, but the data for the coupler DC bias array could not be used for fitting the data. The argument \"coupler_amp_arr\" was: "+str(coupler_amp_arr)
            
            # Assert fittable data.
            assert len(coupler_amp_arr) == len((current_fit_item[0])[0]), "Error: the user-provided raw data does not have the same length as the provided coupler DC bias (array) data."
            
            # Accept cruel fate and move on.
            extracted_data = current_fit_item
            coupler_amp_arr_values = coupler_amp_arr
        
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
        # that made up the coupler bias sweep.
        
        # Report start!
        if the_user_provided_a_list_of_files:
            print("Performing coupler bias sweep fitting on " + current_fit_item + "...")
        else:
            print("Commencing coupler bias sweep fitting of provided raw data...")
        
        assert 1 == 0, str(mag_vals_matrix.shape)
        
    
    assert 1 == 0, "Not implemented!"





def fit_two_tone_spectroscopy_vs_coupler_bias_DEPRECATED(
    data_or_filepath_to_data,
    control_freq_arr = [],
    coupler_amp_arr  = [],
    i_provided_a_filepath = True,
    i_renamed_the_control_freq_arr_to = '',
    i_renamed_the_coupler_amp_arr_to  = '',
    plot_for_this_many_seconds = 0.0,
    number_of_times_to_filter_noisy_raw_curve = 0,
    ):
    ''' From supplied data or datapath, grab two-tone spectroscopy traces.
        At the maximum absolute value from the "floor" (which, might be a
        ceiling if the resonator moves with increasing coupler flux),
        grab a single datapoint for every value of applied coupler DC bias
        that is being swept. Then, reconstruct the entire trace.
        
        The reconstructed trace, is fitted against an expected curve,
        yielding asymptotes where there are avoided two-level crossings.
        The unit, for which there is an avoided two-level crossing,
        will be the same as the coupler axis data was provided in.
        
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
                ## This data file does not contain any information about
                ## a coupler sweep. However, it is mandatory for
                ## this fitting routine that there must be a coupler axis.
                raise RuntimeError("Error! The coupler bias sweep fitter cannot find any provided data for a swept axis bearing the coupler's biasing.")
            
            # Note! Multiplexed functionality disabled for now. TODO
    
    else:
        ## TODO Add catches and checks for the coupler_amp_arr too.
        
        # Then the user provided the data raw.
        assert (not isinstance(data_or_filepath_to_data, str)), "Error: the two-tone spectroscopy (vs. coupler bias) fitter was provided a string type. Expected raw data. The provided variable was: "+str(data_or_filepath_to_data)
        
        # Catch bad user inputs.
        type_and_length_is_safe = False
        try:
            if len(control_freq_arr) != 0:
                type_and_length_is_safe = True
        except TypeError:
            pass
        assert type_and_length_is_safe, "Error: the two-tone spectroscopy fitter (vs. coupler bias) was provided raw data to fit, but the data for the frequency array could not be used for fitting the data. The argument \"control_freq_arr\" was: "+str(control_freq_arr)
        
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
        print("Performing coupler bias sweep fitting on " + data_or_filepath_to_data + "...")
    else:
        print("Commencing coupler bias sweep fitting on provided raw data...")
    
    # There may be multiple resonators involved (a multiplexed readout).
    # Grab every trace (as in, a bias sweep will have many traces),
    # fit the trace, and store the result as a tuple of (value, error).
    fitted_values = [[]] * len(mag_vals_matrix)
    
    for current_res_ii in range(len( mag_vals_matrix )):
        # Note! See above comment on multiplexed sweeps being disabled for now.
        ## # Select current sweep values for this particular resonator.
        ## curr_control_freq_arr_values = control_freq_arr_values[current_res_ii]
        
        # Prepare to store "the trace" that the abs() of every fit yields
        # for the highest point; this trace will trace out how the qubit
        # frequency shifts with the coupler bias.
        qubit_frequency_trip = []
        
        for current_z_axis_value in range(len( mag_vals_matrix[current_res_ii] )):
            
            # Get current trace.
            current_trace_to_fit = (mag_vals_matrix[current_res_ii])[current_z_axis_value]
            
            # Filter and interpolate the trace?
            if number_of_times_to_filter_noisy_raw_curve > 0:
                current_trace_to_fit = filter_and_interpolate(
                    datapoints_to_filter_and_interpolate = current_trace_to_fit,
                    number_of_times_to_filter_and_interpolate = number_of_times_to_filter_noisy_raw_curve
                )
            
            # Get result and store.
            # The resonance could have either been going up or going down,
            # if something for some reason moved around with a shifting
            # tunable coupler's bias. Let's find out where (likely) the
            # floor/ceiling is, by looking at the median. Then find whether
            # the "biggest" negative is bigger then the "biggest" positive.
            noise_level_fs_unit = np.median(current_trace_to_fit)
            deviations_in_current_trace_to_fit = current_trace_to_fit - noise_level_fs_unit
            if np.abs(np.min(deviations_in_current_trace_to_fit)) > np.abs(np.max(deviations_in_current_trace_to_fit)):
                # If true, then the peak is going *downwards*
                # In principle, not supposed to happen. In reality, more so.
                ## Find which index along the frequency axis this happens.
                freq_val_peak = control_freq_arr_values[(deviations_in_current_trace_to_fit).argmin()]
                qubit_frequency_trip.append(freq_val_peak)
            else:
                # If false, then the peak is going upwards (as expected).
                freq_val_peak = control_freq_arr_values[(deviations_in_current_trace_to_fit).argmax()]
                qubit_frequency_trip.append(freq_val_peak)
        
        # NOTE! At this point, we step back one step in the indentation,
        #       because, the fit assumes that all of the Z-axis values
        #       make up a trace. The Z-axis is part of the fit, so to speak.
        #       In contrast to most other fitting routines around here,
        #       in this file that you have open, you will not perform
        #       one fit per z-axis value.
        
        # Try to fit current trace.
        try:
            optimal_vals, fit_err, fit_curve = fit_coupler_bias_curve_from_avoided_two_level_crossings(
                frequencies    = control_freq_arr_values,
                coupler_values = coupler_amp_arr_values,
                datapoints     = qubit_frequency_trip,
            )
            raise NotImplementedError("Halted! Function not finished.") # TODO not finished.
            # Grab fitted values. The x0 gives the resonator dip, hopefully.
            resonator_peak = optimal_vals[1]
            fit_error      = fit_err[1]
            
            # Print result.
            print("TODO" + str(resonator_peak) + " ±" + str(fit_error))
            
            # Store fit and its plusminus error bar.
            (fitted_values[current_res_ii]).append((resonator_peak, fit_error))
            
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
            (fitted_values[current_res_ii]).append((resonator_peak, fit_error))
    
    # We're done.
    return fitted_values
    
def fit_coupler_bias_curve_from_avoided_two_level_crossings_DEPRECATED(
    frequencies,
    coupler_values,
    datapoints
    ):
    ''' Grab submitted data of a two-tone spectroscopy, performed while
        sweeping the applied magnetostatic flux onto a connected tunable
        SQUID coupler.
    '''
    
    # Numpy-ify the input datapoints array.
    datapoints = np.array(datapoints)
    
    # At this point, we assume the following:
    # 1. The sweep was done over one full flux quantum period. Meaning,
    #    that we may split the input data in two chunks, fit both sets of data,
    #    find asymptotes, and from this information establish what the
    #    [coupler unit] to flux quantum conversion is. And, find the static
    #    offset due to the residual magnetostatic flux threading the coupler.
    # 2. On fit failure, let's assume that the sweep was only done over
    #    one-half of a full flux quantum period. From this information,
    #    we cannot find the static offset from 0. But, we can figure out
    #    a bias value to feed into processes that rely on the fit.
    
    ## First attempt: make two datasets.
    assume_input_data_was_done_over_one_full_period = True
    datapoints_first, datapoints_second  = np.split(datapoints, 2)
    # TODO things happen.
    #try:
    ## datapoints_first
    ## datapoints_second
    #except TODO:
    #    in this exception, set assume_input_data_was_done_over_one_full_period to False.
    assume_input_data_was_done_over_one_full_period = False
    
    ##if (not assume_input_data_was_done_over_one_full_period):
    ##
    ##  TODO for now, let's only work with the edge case, that the
    ##       sweep was only done over one-half of a flux quantum period.
    ##       The stuff below can probably be looped in a fancy way.
    
    # Get a first guess of the y-offset. This part may be a bit tricky.
    # There is no actual guarantee that the sweep was done around
    # the actual qubit frequency. We may first guess that it was.
    f_01_guess = ( frequencies[0] + frequencies[-1] )/2
    
    # My idea:
    # Mirror the datapoints curve about the f_01_guess.
    # If there is a common offset to every value, then remove this offset.
    # I believe this method would place the f_01_guess closer to the assumed
    # |0⟩ → |1⟩ transition. Then, make a new guess.
    f_01_guess = f_01_guess - np.min(np.abs(datapoints - f_01_guess))
    
    # I believe this new guess to be "close enough" for the fit to run.
    ## TODO: However, the method outlined above would only improve the
    ##       f_01_guess if the original f_01_guess was guessed higher in
    ##       frequency than the actual |0⟩ → |1⟩ transition.
    ##           To understand this: try and draw a two-tone-spectroscopy
    ##           versus a coupler bias sweep. Then, make a deliberately
    ##           bad guess where the qubit |0⟩ → |1⟩ transition is located,
    ##           and make sure that your guess has a *lower* frequency
    ##           than the actual |0⟩ → |1⟩ transition. Then follow along
    ##           with my algorithm above. You'll see that there is no
    ##           improvement as to the quality of the f_01_guess.
    
    # Guess the "anharmonicity" thing. The value does probably not correspond
    # to an actual anharmonicity. Looking at the equation below, when
    # Delta = anharmonicity, the function will sky-rocket towards a vertical
    # asymptote. Meaning that the second split of the curve will correspond
    # to the anharmonicity, since the frequency detuning is seen as the x-axis.
    # The asymptote, we assume to be located roughly at 2/3 along the
    # full length of the coupler sweep. However, the 2/3 assumption
    # assumes that the first asymptote is at zero.
    anharm_something_guess = np.abs(                    \
        coupler_values[int(len(coupler_values)*(2/3))] - \
        coupler_values[int(len(coupler_values)*(1/3))]   \
    )
    
    # Let's make a guess for how offset the curve is. Its x is assumed to
    # have its first asymptote at 0.0 -- meaning that the offset for now,
    # to make the curve fit, should be where we assume to have our first
    # vertical asymptote.
    x_offset_guess = coupler_values[int(len(coupler_values)*(1/3))]
    
    # As for guessing the shape of g, I frankly have no idea.
    g_something = 180.65
    
    # Scale? Assume 1.0 for now.
    x_scalar_guess = 1.0
    
    # Perform fit with gaussian_function as the model function,
    # and the initial guess of parameters as seen below.
    assert 1 == 0, 'Cannot fit.'
    with np.errstate(invalid='ignore'):
        optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
            f     = qubit_dispersive_shift_vs_coupler_flux_function,
            xdata = coupler_values,
            ydata = datapoints,
            p0    = (x_scalar_guess, x_offset_guess, f_01_guess, g_something, anharm_something_guess)
        )
    
    assert 1 == 0, str(optimal_vals)
    
    ## TODO DEBUG!
    plt.plot(coupler_values, datapoints, color="#034da3")
    
    coupler_values = np.linspace(0.084, +0.766, len(coupler_values)*4)
    TODO_f_01_guess_val = np.linspace(f_01_guess, f_01_guess, len(coupler_values))
    plt.plot(coupler_values, TODO_f_01_guess_val, color="#31ad55")
    
    plot_me_todo = qubit_dispersive_shift_vs_coupler_flux_function(
        x = coupler_values,
        x_scalar = 1.0,
        x_offset = x_offset_guess,
        f_01 = f_01_guess,
        g_something = 180.65,
        anharm_something = anharm_something_guess
    )
    plt.plot(coupler_values, plot_me_todo, color="#ee1620")
    plt.title('Largest deviation from median of signal')
    plt.xlabel('Coupler bias [FS]')
    plt.ylabel('Control tone frequency [Hz]')
    plt.show()
    
    assert 1 == 0, "Halted! Remember to add, that the fitter must also return the \"symmetry imbalance\" about the 0 FS bias axis. The user should be told how much static offset there is, i.e. how much bias really corresponds to a true zero."
    raise NotImplementedError("Halted! This function exists but has not yet been worked on.")

def qubit_dispersive_shift_vs_coupler_flux_function_DEPRECATED(
    x,
    x_scalar,
    x_offset,
    f_01,
    g_something,
    anharm_something
    ):
    raise NotImplementedError("Halted! Not finished.")
    ''' Function to be fitted against.
        
        The formulas here follow (145) in Krantz2019,
        Appl. Phys. Rev. 6, 021318 (2019); doi: 10.1063/1.5089550
        https://aip.scitation.org/doi/10.1063/1.5089550
    '''
    ## TODO DEBUG chi_disp_shift = -1 * (g_01 ** 2) / Delta * (1/(1+(Delta/anharmonicity)))
    
    ## TODO tunable bus: return scalar * 6.28318530718 * freq_01 * (abs( np.cos( phi / phi_0 ) )) ** (1/2)
    return f_01 + -(((g_something/6.28318530718) ** 2) / ((-x*x_scalar+x_offset)/(6.28318530718))) * (1/(1+(((-x*x_scalar+x_offset)/(6.28318530718))/(anharm_something/(6.28318530718)))))