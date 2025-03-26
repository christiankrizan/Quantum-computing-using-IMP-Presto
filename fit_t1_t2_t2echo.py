#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import Labber
import h5py
import math
import array
import re
import os
import time
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.stats import moment
from datetime import datetime

def fit_triple_decoherence_data_run(
    filepath,
    select_resonator,
    select_T1_T2_T2e = 'T1',
    maximum_tolerable_error_in_percent = 15,
    use_this_colour = 'firebrick',
    set_transparent = True,
    histogram_legend_location = None,
    known_T1_distribution = [],
    discard = [],
    no_entries = None,
    i_renamed_the_delay_arr_to = '',
    overwrite_q_label = ''
    ):
    ''' Takes a Labber-formatted log file (.hdf5),
        or a list of files taken using the .h5 format from Indecorum,
        where data runs have been done in sets of three: T₁, T₂*, T₂-echo.
        I.e. the measurement consists of N repetitions of three measurement
        traces, the first one of which will contain T₁ data, the second
        will contain T₂* data, and the third one will contain T₂-echo data.
        
        select_resonator: in a file containing multiplexed readout data,
        select your qubit (and resonator) here. Counting from 0.
        
        select_T1_T2_T2e: select whether to fit T₁, T₂* or T₂-echo.
        Legal inputs for select_T1_T2_T2e is 'T1', 'T2' and 'T2e'.
        
        maximum_tolerable_error_in_percent: cut-off for the scipy fitter.
        If the fit's error has a square root of the covariance matrix diagonals
        that is 15 % (or other value), then discard the entry. Lower is better,
        but then you depend on having really good (not noisy) input data.
        
        use_this_colour: matplotlib entry, set data in plots to this colour.
        set_transparent: matplotlib entry, make background of plot transparent.
        histogram_legend_location: matplotlib histogram legend locator.
        
        known_T1_distribution: provided this T₁ data set, remove
        dephasing time (T₂, T₂-echo) datapoints that invalidate T₂ < 2·T₁.
        
        discard: remove this iteration's data from the set.
        
        no_entries: if None, use data file length (number of trace entries)
        to judge how many entries are in the file. Else, override to
        some user-provided argument.
        
        i_renamed_the_delay_arr_to: if using the .h5 data format list option,
        then the user might have provided some string that renames the
        default "delay_arr" argument, i.e., the data for the X axis might
        have a different name in the data file itself.
        
        Returns:
        Numpy array containing the result of the fits.
        
        Originally built using a code basis provided by Janka Biznárová,
        find her at https://orcid.org/0000-0002-8887-8816.
    '''
    
    # User syntax checking.
    if select_T1_T2_T2e == 'T2echo':
        select_T1_T2_T2e = 'T2e'
    if not ((select_T1_T2_T2e == 'T1') or (select_T1_T2_T2e == 'T2') or (select_T1_T2_T2e == 'T2e')):
        raise AttributeError("Error! The user did not provide a legal syntax for the argument 'select_T1_T2_T2e'. Legal values are 'T1', 'T2', or 'T2e'")
    
    # Numpy formatting
    known_T1_distribution = np.array( known_T1_distribution )
    
    # Various functions to be fitted to, for instance.
    def prettify_decoherence_string(input_string):
        return (input_string.replace('1','₁')).replace('2','₂')
    
    def T1_func(x, A, T1, offset):
        return A * np.exp(-x/T1) + offset
    
    def T2_func(x, A, T2_asterisk, offset, f, phase):
        return A * np.exp(-x / T2_asterisk) * np.cos(2.0*np.pi * f * x + phase) + offset
    
    def T2echo_func(x, A, T2_echo, offset):
        return A * np.exp(-x/T2_echo) + offset
    
    def extract_file_creation_time(filename):
        ''' Given an .h5 file, extract the file creation time.
        '''
        # Regular expression to match the date and time pattern
        match = re.search(r'(\d{2})-([A-Za-z]{3})-(\d{4})_\((\d{2})_(\d{2})_(\d{2})\)', filename)
        if not match:
            raise ValueError("Could not extract file creation time from its filename: "+str(filename))
        
        # Collect the results of the regular expression.
        # Then, convert the month abbreviation into a month number.
        day, month_str, year, hour, minute, second = match.groups()
        month = time.strptime(month_str, "%b").tm_mon
        
        # Create a datetime object.
        # Finally, return its UNIX timestamp
        ## Unfortunately, we have to return this value as a numpy float, to
        ## fit in with the Labber Log Browser philosophy of doubles-only.
        dt = datetime(int(year), month, int(day), int(hour), int(minute), int(second))
        return np.float64( dt.timestamp() )
    
    
    #   Log file to be analysed.
    ##  Or, list of .h5 log files.
    if not isinstance(filepath, list):
        f = Labber.LogFile(filepath)
        
        # Acquire the embedded timestamp of the Labber log file.
        # From this, we create a time axis that is valid for the entire file.
        with h5py.File(filepath, 'r') as h5f:
            # Delve into a horrid shittiness of a Labber log file data structure...
            
            ## Grab first timestamp and creation time.
            ## NOTE: the Log Browser API does not enable the user to do this.
            ##       Thank me later. I had to reverse-engineer these attributes.
            creation_time  = h5f.attrs["creation_time"]
            timestamp_list = np.array(h5f["Data"]["Time stamp"][:], dtype=np.float64) + creation_time
            
            ## Grab remaining timestamps.
            for ii in list(h5f.keys()):
                if ii.startswith('Log_'):
                    timestamp_list = np.append( \
                        timestamp_list, \
                        np.array(h5f[ ii ]["Data"]["Time stamp"][:], dtype=np.float64) + h5f[ ii ].attrs["creation_time"] \
                    )
    else:
        # Then we are operating on lists of .h5 filepaths.
        creation_time = extract_file_creation_time(filepath[0])
        ## TODO There is a thing here. The .h5 data files currently do not
        ##      contain information regarding when the individual data points
        ##      were taken. As such, there is a time offset equal to that
        ##      of the duration of the first T₁ measurement, when comparing
        ##      to the Labber data **I think**. Thus, the difference here
        ##      is that the timestamp list is essentially the time at which
        ##      the fit for that datapoint was performed.
        
        #  Get the remaining timestamps for the other datapoints.
        ## Compared to the Labber alternative above, we do not have to add
        ## the original timestamp as an offset to this data.
        timestamp_list = []
        for current_item in filepath:
            timestamp_list.append( extract_file_creation_time( current_item ) )
    
    # Catch user override for the number of entries. The number of entries is
    # simply the total number of T₁, T₂* and T₂e measurements.
    if type(no_entries) == type(None):
        if not isinstance(filepath, list):
            no_entries = f.getNumberOfEntries()
        else:
            no_entries = len(filepath)
    
    #  Acquire assumed x-axis length of the data traces.
    ## The main purpose of this, later, is to spot whether a measurement
    ## was aborted in Labber. For the Indecorum backend, as of yet,
    ## no trace is collected for an aborted setup. And, the data
    ## may very well have different numbers of datapoints in them.
    if not isinstance(filepath, list):
        (a1,b1) = f.getTraceXY(0,0,0) # Note, a1 is in units of seconds.
        points = len(np.real(b1))
    else:
        # For the .h5 data structure, we have no limitation, and
        # no check for points have to be done, strictly speaking.
        pass
    
    #  Catch whether the file(path) provided contains only one trace.
    #  Then, see whether its information can be used for comparison
    #  between traces.
    ## For the .h5 data structure, simply look at the provided list.
    ## Also, there is no limitation that all entries have to have the same
    ## length. So skip that check.
    file_contains_only_one_trace = True
    if not isinstance(filepath, list):
        try:
            (c1,d1) = f.getTraceXY(0,0,1)
            # This operation succeeded, we know that the file contains more
            # than a single trace.
            file_contains_only_one_trace = False
        except:
            ## Note that Labber is shittily coded and does not return
            ## any usable exception class for .getTraceXY( illegal value ),
            ## just 'Error', hence this try-catch all exceptions. Ew.
            # This exception triggered, the file contains only a single trace.
            pass
        
        # Check whether the length of the first trace can be used
        # to compare to the length of all other traces.
        if not file_contains_only_one_trace:
            if not (len(np.real(d1)) == points):
                raise AttributeError("Error! The length of the very first trace and the second trace in the datafile differs. Check your input data. All traces have to be of the same x-axis length as the first trace.")
    else:
        pass
        ##if len(filepath) != 1:
        ##    file_contains_only_one_trace = False
    
    # Declare arrays for storing decoherence times. Also, we must keep track
    # of files that have the same length as the original input file.
    T_dec       = []
    error_T_dec = []
    T_dec_full       = []
    error_T_dec_full = []
    
    # Declare iteration counter.
    counter = 0
    
    # Declare array for holding the particular
    # time where the measurement was performed.
    measurement_time = []
    
    # For all entries:
    if select_T1_T2_T2e == 'T1':
        entry_offset = 0
    elif select_T1_T2_T2e == 'T2':
        entry_offset = 1
    elif select_T1_T2_T2e == 'T2e':
        entry_offset = 2
    else:
        raise AttributeError("Error! Somehow, the argument 'select_T1_T2_T2e' became illegal. Good job. The argument was: "+str(select_T1_T2_T2e))
    
    # For all traces stored in the input file, stepped in steps of 3:
    for i in range(0 + entry_offset, no_entries, 3):
        
        # Put 'NaN' into the T_dec_full entry for this fit, until
        # further notice. Note that T_dec_full will be len(file) / 3 long!
        ## Ideally, it is possible that Indecorum was stopped early,
        ## and may thus be of some other length, non-divisible by 3.
        T_dec_full.append( np.nan )
        error_T_dec_full.append( np.nan )
        
        # Fetch trace!
        if not isinstance(filepath, list):
            (delay,complex_datapoint) = f.getTraceXY(select_resonator,0,i)
        else:
            ## A bit more complicated for the .h5 structure.
            ## Open the file, get the data.
            with h5py.File(os.path.abspath( filepath[i] ), 'r') as h5f:
                complex_datapoint = (h5f["processed_data"][()])[select_resonator][0]
                
                if i_renamed_the_delay_arr_to == '':
                    delay = h5f[ "delay_arr" ][()]
                else:
                    delay = h5f[i_renamed_the_delay_arr_to][()]
                
                # To get around the non-restriction for the .h5 data
                # list format thingy, we write the "points" variable here.
                points = len(complex_datapoint)
                
                # Also, given that the measurements can look quite differently
                # in the .h5 data format thingy from one measurement to
                # another, we should also get the a1 parameter here.
                a1 = delay
            
        # Extract absolute value, store in plot handle.
        readoutY = np.abs(complex_datapoint)
        plt.plot(delay, readoutY)
        
        # Does the user request to remove fits that are knowingly wrong?
        if not (i in np.array(discard)):
            
            # Verify that the measurement was not aborted during
            # the taking of the current trace.
            ## Note: this assumes that every trace will have the
            ## same number of points along the x-axis as the first trace.
            if not (len(readoutY) == points):
                print("The trace of iteration "+str(i)+" has a different datalength than expected, skipped under presumption of measurement abortion.")
            else:
                
                # Catch otherwise shitty traces.
                try:
                    
                    # Get initial amplitude and offset guesses.
                    initial_guess_A = readoutY[0] - readoutY[-1]
                    initial_guess_offset = readoutY[-1]
                    
                    # Get index of the time axis that is closest to the amplitude that most likey corresponds to the T_dec value.
                    pop_likely_for_T_dec = 0.36787944117 * initial_guess_A + initial_guess_offset
                    
                    ## Here, you'd either have an a1 parameter from reading
                    ## the Labber file. Or, some other time trace thing
                    ## from reading the .h5 list format things.
                    initial_guess_T_dec = a1[np.abs(readoutY - pop_likely_for_T_dec).argmin()]
                    
                    # Fit data given p0 as initial guessing values.
                    fit_failure = False
                    if select_T1_T2_T2e == 'T1':
                        try:
                            p, p_cov = optimize.curve_fit(T1_func, a1, readoutY, p0 = [initial_guess_A, initial_guess_T_dec, initial_guess_offset])
                        except RuntimeError:
                            # Fit failure.
                            print("WARNING: Fit failure at iteration "+str(i)+", skipping.")
                            fit_failure = True
                    elif select_T1_T2_T2e == 'T2':
                        
                        # Guess the phase of the Ramsey oscillations.
                        begins_at = (readoutY[0] - initial_guess_offset) / initial_guess_A
                        if begins_at > 1:
                            starting_phase = np.arccos(1)
                        elif begins_at < -1:
                            starting_phase = np.arccos(-1)
                        else:
                            starting_phase = np.arccos(begins_at)
                        
                        # Guess the frequency of the Ramsey oscillations.
                        fq_axis   = np.fft.rfftfreq(len(a1), (a1[1]-a1[0]))
                        fft       = np.fft.rfft(readoutY)
                        frequency_guess = fq_axis[1 + np.argmax( np.abs(fft[1:] ))]
                        
                        # Fit!
                        try:
                            p, p_cov = optimize.curve_fit(T2_func, a1, readoutY, p0 = [initial_guess_A, initial_guess_T_dec, initial_guess_offset, frequency_guess, starting_phase])
                        except RuntimeError:
                            # Fit failure.
                            print("WARNING: Fit failure at iteration "+str(i)+", skipping.")
                            fit_failure = True
                    elif select_T1_T2_T2e == 'T2e':
                        try:
                            p, p_cov = optimize.curve_fit(T2echo_func, a1, readoutY, p0 = [initial_guess_A, initial_guess_T_dec, initial_guess_offset])
                        except RuntimeError:
                            # Fit failure.
                            print("WARNING: Fit failure at iteration "+str(i)+", skipping.")
                            fit_failure = True
                    else:
                        raise AttributeError("Error! Somehow, the argument 'select_T1_T2_T2e' became illegal. Good job. The argument was: "+str(select_T1_T2_T2e))
                    
                    # Calculate fit error.
                    if not fit_failure:
                        p_err = np.sqrt(np.diag(p_cov))
                        T_dec_error_perc = p_err[1]/p[1] * 100
                    else:
                        # In case there was no T_dec_error_perc declared,
                        # make one.
                        T_dec_error_perc = 100
                    
                    # Append datapoint?
                    if (abs(T_dec_error_perc) <= maximum_tolerable_error_in_percent) and (not fit_failure):
                        
                        # Set flag.
                        t2_or_t2e_legal = True
                        
                        # Is T2 > 2 * T1?
                        if (len(known_T1_distribution) > 0) and ((select_T1_T2_T2e == 'T2') or (select_T1_T2_T2e == 'T2e')):
                            if not (np.isnan(known_T1_distribution[counter])):
                                if p[1] > 2*known_T1_distribution[counter]:
                                    print("WARNING! T₂ (or T₂-echo) of iteration "+str(i)+" was larger than 2·T₁ for this entry. Skipping.")
                                    t2_or_t2e_legal = False
                        
                        if t2_or_t2e_legal:
                            # Append succesful fit and report.
                            print('Iteration: ', i, ' '+prettify_decoherence_string(select_T1_T2_T2e)+' value is: ', p[1], ' with error: ', p_err[1], ' and percentage: ', T_dec_error_perc )
                            T_dec.append(p[1])
                            error_T_dec.append(p_err[1])
                            
                            # Also append this fit to the 'total' matrices
                            T_dec_full[counter] = p[1]
                            error_T_dec_full[counter] = p_err[1]
                            
                            # Append current time of measurement, counting from its very start.
                            measurement_time.append( (timestamp_list[i] - timestamp_list[0])/3600 )
                    else:
                        print("Error too large in iteration "+str(i)+", skipping.")
                
                except ValueError:
                    print("Encountered error, skipping iteration "+str(i)+".")
        
        else:
            print("The user requested to skip iteration "+str(i)+".")
        
        # Tick iteration counter.
        counter += 1
    
    # Axis formatting.
    measurement_time = np.array(measurement_time)
    T_dec = np.array(T_dec)
    error_T_dec = np.array(error_T_dec)
    
    # Microsecond formatting:
    T_dec = T_dec*1e6
    error_T_dec = error_T_dec*1e6
    
    # Figure out the qubit number label.
    if overwrite_q_label != '':
        q_label = overwrite_q_label
    else:
        q_label = 'Q'+str(select_resonator+1)
    
    # Axis and figure size dimension management.
    T_dec_plots = plt.figure(figsize = [23.5,10.0])
    ax1 = T_dec_plots.add_subplot(111,  label="1")
    ax1.errorbar(measurement_time, T_dec, error_T_dec, color=use_this_colour, marker = 'o', fmt='o',markersize = 12, capsize = 15, label=q_label)
    ax1.set_ylabel('$T_{'+str(select_T1_T2_T2e.replace('T',''))+'}$ [µs]', fontsize = 37)
    ax1.set_xlabel('Time [h]', fontsize = 37)
    
    # Add legend.
    plt.legend(fontsize=30, loc='upper right')
    
    # Set axis limits and font sizes.
    ##plt.xlim(measurement_time[0]-2,measurement_time[-1]+2) # Might generate different-length x-axes between measurements. Beware.
    # Force x-axis limits. Since different measurements may end at different
    # times, we round the last datapoint to the nearest hour up. And, ±2 hours.
    plt.xlim(-2, np.ceil((timestamp_list[-1] - timestamp_list[0])/3600)+2)
    
    # ... continue as normal.
    plt.xticks(fontsize = 31)
    plt.ylim(0,220)
    plt.yticks(fontsize = 31)
    
    # Set title.
    plt.title('$T_{'+str(select_T1_T2_T2e.replace('T',''))+'}$ time scatter', fontsize=36)
    
    # Get figure name.
    if not isinstance(filepath, list):
        plt.savefig(filepath.replace('.hdf5','')+'_'+q_label+'_'+select_T1_T2_T2e+'_time' + ".png", bbox_inches = 'tight', transparent = set_transparent)
    else:
        # Extract an appropriate file date.
        match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4}_\(\d{2}_\d{2}_\d{2}\))', filepath[-1])
        if match:
            time_name = match.group(0)
        else:
            print("WARNING: could not extract an appropriate date for the exported file. Defaulting to current UNIX time.")
            time_name = str(int(time.time()))
        plt.savefig('decoherence_analysis_'+q_label+'_'+time_name+'_'+select_T1_T2_T2e+'_time' + ".png", bbox_inches = 'tight', transparent = set_transparent)
    print("Total duration for time scatter data: "+str((timestamp_list[-1] - timestamp_list[0])/3600) + " hours. UNIX time for the start is at: "+str(timestamp_list[0]))
    
    ## Decoherence time histogram ##
    
    # Determine the number of bins of the distribution:
    # We are expecting a distribution that is somewhat binomially distributed.
    # Here, Sturge's formula along with Doane's correction factor is used
    # for getting a decent number of bins.
    
    no_entries = len(T_dec)
    mean_T_dec = int(round(np.mean(T_dec),0)) ###round(np.mean(T_dec),2) #np.around(np.mean(T_dec),0)
    std_T_dec  = int(round(np.std(T_dec),0)) ###round(np.std(T_dec),2)  #np.around(np.std(T_dec),0)
    third_moment_skewness_of_ditribution = moment(T_dec, moment = 3) # Get the assymetry of the distribution
    sigma_g1 = np.sqrt( (6*(no_entries - 2))/((no_entries + 1)*(no_entries + 3)) )
    doane_correction_factor_Ke = np.log2(1 + np.abs(third_moment_skewness_of_ditribution)/sigma_g1)
    bins_calculated = int(np.ceil(1 + np.log2( no_entries ) + doane_correction_factor_Ke))
    
    T_dec_histogram_plot = plt.figure(figsize = [10.75,10.0])
    plt.hist(T_dec, bins = bins_calculated, alpha=0.5, color=use_this_colour, label= prettify_decoherence_string(select_T1_T2_T2e)+' = ' + str(mean_T_dec) + ' ±' + str(std_T_dec) + ' µs', rwidth = 0.9)
    
    stringus_dingus = ' $T_{'+str(select_T1_T2_T2e.replace('T',''))+'}$'
    # Add start to T_2^* data?
    if '$T_{2}$' in stringus_dingus:
        stringus_dingus = stringus_dingus.replace('$T_{2}$','${T_{2}}^{*}$')
    plt.title(q_label + stringus_dingus + ' histogram', fontsize=36)
    plt.ylabel('Counts', fontsize = 37)
    
    # Rudimentary check whether the histogram
    # will plot with decimals on the Y axis.
    low_limit_for_y  = round(((plt.gca()).get_ylim())[0],0)
    high_limit_for_y = round(((plt.gca()).get_ylim())[1],0)
    (plt.gca()).set_ylim((low_limit_for_y,high_limit_for_y))
    
    plt.xlabel(stringus_dingus + ' [µs]', fontsize = 37)
    plt.grid(linestyle='--',linewidth=0.5)
    
    plt.xticks(fontsize = 31)
    plt.yticks(fontsize = 31)
    if type(histogram_legend_location) == type(None):
        plt.legend(fontsize = 30)
    else:
        plt.legend(fontsize = 30, loc = histogram_legend_location)
    
    # Export figure things.
    if not isinstance(filepath, list):
        plt.savefig(filepath.replace('.hdf5','') + '_'+q_label+'_'+select_T1_T2_T2e+'_histogram' + ".png", bbox_inches = 'tight')
    else:
        # Extract an appropriate file date.
        match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4}_\(\d{2}_\d{2}_\d{2}\))', filepath[-1])
        if match:
            time_name = match.group(0)
        else:
            print("WARNING: could not extract an appropriate date for the exported file. Defaulting to current UNIX time.")
            time_name = str(int(time.time()))
        plt.savefig('decoherence_analysis_'+q_label+'_'+time_name+'_'+select_T1_T2_T2e+'_histogram' + ".png", bbox_inches = 'tight')
    
    # Plot shits. Return.
    plt.show()
    return (T_dec, T_dec_full)