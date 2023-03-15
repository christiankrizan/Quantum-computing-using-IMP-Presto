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
from smoothener import filter_and_interpolate

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
    
    #assert 1 == 0, "TODO lf( [[(-9.4, -8.17), (-4.47, -1.51), (1.63, 4.59), (7.94, 10)], [(-9.19, -8.58),(-4.24, -2.21),(2.01, 4.07),(8."+\
    #    "26, 10.1)]], [[(4.74,7.87),(-1.54,1.71),(-8.05,-4.89)], [(-6.96,-5.74),(-0.951, 1.04),(4.2,7.88)]], -1, 'V' )"
    
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

def fit_resonator_spectrography_vs_coupler_bias(
    ):
    ''' TODO This function exists, but has not been worked on.
    '''
    raise NotImplementedError("Halted! Fitting resonator-vs-coupler-bias plots is not yet supported.")
    
    
def fit_two_tone_spectroscopy_vs_coupler_bias(
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
            assert 1 == 0, "HALTED!!" # TODO DEBUG
            # Grab fitted values. The x0 gives the resonator dip, hopefully.
            resonator_peak = optimal_vals[1]
            fit_error      = fit_err[1]
            
            # Print result.
            print("herb aderba blurgh dorp dorp: " + str(resonator_peak) + " ±" + str(fit_error/2))
            
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
    
    
def fit_coupler_bias_curve_from_avoided_two_level_crossings(
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

def qubit_dispersive_shift_vs_coupler_flux_function(
    x,
    x_scalar,
    x_offset,
    f_01,
    g_something,
    anharm_something
    ):
    ''' Function to be fitted against.
        
        The formulas here follow (145) in Krantz2019,
        https://aip.scitation.org/doi/10.1063/1.5089550
    '''
    ## TODO DEBUG chi_disp_shift = -1 * (g_01 ** 2) / Delta * (1/(1+(Delta/anharmonicity)))
    
    ## TODO tunable bus: return scalar * 6.28318530718 * freq_01 * (abs( np.cos( phi / phi_0 ) )) ** (1/2)
    return f_01 + -(((g_something/6.28318530718) ** 2) / ((-x*x_scalar+x_offset)/(6.28318530718))) * (1/(1+(((-x*x_scalar+x_offset)/(6.28318530718))/(anharm_something/(6.28318530718)))))