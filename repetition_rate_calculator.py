#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

def get_repetition_rate_T(
    T_begin,
    T_last,
    repetition_rate,
    repetition_counter,
    repetition_rate_constraint_is_hard_and_may_not_be_broken = False
    ):
    ''' Calculate a timestamp T that can be used for scheduling
        events using the IMP scheduler API, so that these events
        are aligned to multiples of some factor repetition_rate.
        
        repetition_rate_constraint_is_hard_and_may_not_be_broken sets whether
        the repetition_rate may be automatically extended, if the
        constraint is in fact soft, and whether it's more important
        that phase is preserved between measurements.
        
        In other words, whether the most important thing, is simply
        that every new measurement iteration, starts at some
        (common) multiple of the repetition_rate time.
    '''
    
    # How long is the sequence that is currently being scheduled?
    T_needed_initially = T_last - T_begin
    
    # When calling this subroutine, when was the next iteration
    # originally assumed to take place?
    T_next_originally = repetition_rate * repetition_counter
    
    # Check whether the sequence that is currently being scheduled,
    # falls within the time bound set by the user.
    if (T_needed_initially <= repetition_rate):
    
        # All is good. Just report the T that is supposed to happen
        # at that point in the sequence.
        T = repetition_rate * repetition_counter
        repetition_counter += 1
        
    else:
        
        # The sequence that the user is attempting to schedule,
        # fails the time constraint. If this constraint is hard, then halt.
        assert (not repetition_rate_constraint_is_hard_and_may_not_be_broken),\
            "Error! Hard time constraint failure. The user attempted to "    +\
            "schedule a sequence that was at least "+str(T_needed_initially) +\
            " seconds long. But, such a long sequence does not fit within "  +\
            "the requested repetition rate of "+str(repetition_rate)         +\
            " seconds."
        
        # The repetition rate time constraint is not hard. Make another one.
        while (T_last - T_begin) >= repetition_rate:
            T = repetition_rate * repetition_counter
            repetition_counter += 1
            
            # Move reference
            T_begin = T
        
        # Print warning.
        print("WARNING! Repetition rate "+str(repetition_rate)  +\
            " s, was shorter than the least known required sequencer "       +\
            "duration; at least "+str(T_needed_initially)+" s are needed. "  +\
            "One iteration was placed later than its assumed scheduled time "+\
            "T = "+str(T_next_originally)+" s, because the flag" +\
            " \"repetition_rate_constraint_is_hard_and_may_not_be_broken\"" +\
            " was set to False. The event was placed at T = "+str(T)+" s.")
        
    return T, repetition_counter

def illustrate_repetition_rate(
    calculated_repetition_rate,
    array_of_waveform_frequencies = [],
    array_of_waveform_frequencies_that_are_reset_on_every_repetition = []
    ):
    ''' Plot the calculated frequencies, illustrating where they are
        phase commensurate.
    '''
    
    # Perform user input checks.
    if len(array_of_waveform_frequencies) <= 0:
        raise ValueError("Could not understand the frequencies provided.")
    if calculated_repetition_rate <= 0:
        raise ValueError("The provided calculated repetition rate is illegal; the rate cannot be negative nor zero. The provided number was: "+str(calculated_repetition_rate))
    
    # For a simple initial illustration, let's find out where on the time
    # axis that we see a few oscillations of the slowest curve. Let's say 3.
    ## Then, we elongate the time axis to 3x that. We also put vertical lines
    ## on 1x, 2x, and 3x the repetition rate, along the time axis.
    # Now, let's find the slowest frequency requested. And, grab a time value.
    lowest_frequency = np.min(array_of_waveform_frequencies_that_are_reset_on_every_repetition)
    if lowest_frequency > np.min(array_of_waveform_frequencies):
        lowest_frequency = np.min(array_of_waveform_frequencies)
    period_of_lowest_frequency = 1/lowest_frequency
    time_axis = np.linspace(0, 3*period_of_lowest_frequency, 500)
    
    # Stack waves in the plot!
    stacked_waves = 0
    for ii in range(len(array_of_waveform_frequencies)):
        # Here, the ii portion ensures that the curve is offset.
        y_vals = np.cos(2*np.pi * array_of_waveform_frequencies[ii] * time_axis) + ii*2 + 1.6
        plt.plot(time_axis, y_vals, color="#034da3")
        stacked_waves += 1
    
    # Coupler tones too?
    for jj in range(len(array_of_waveform_frequencies_that_are_reset_on_every_repetition)):
        ## Now, we must calculate this segment in parts.
        part_y_axis = []
        for time_item in time_axis:
            part_y_axis.append(
                np.cos(2*np.pi * array_of_waveform_frequencies_that_are_reset_on_every_repetition[jj] * (time_item % calculated_repetition_rate)) + (jj+stacked_waves)*2 + 1.6
            )
        plt.plot(time_axis, part_y_axis, color="#000000")
    
    # Plot repetition rate.
    done = False
    maximum_iterator = 0
    while (not done) and (maximum_iterator <= 10):
        if (calculated_repetition_rate * maximum_iterator) < time_axis[-1]:
            plt.axvline(x=(calculated_repetition_rate * maximum_iterator), color="#ef1620", linestyle='--')
        else:
            done = True
        maximum_iterator += 1
        if maximum_iterator == 10:
            raise ValueError("Halted! You are plotting too many repetition rate lines.")
    
    # Prepare the plot.
    plt.title('Phase error from bad repetition rates', fontsize=22)
    plt.xlabel('Time [s]', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Remove the Y axis.
    plt.yticks([])
    
    # Show the plot!
    plt.show()