#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

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
    if (T_needed_initially < repetition_rate):
    
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
        print("WARNING: the provided repetition rate "+str(repetition_rate)  +\
            " s, was shorter than the least known required sequencer "       +\
            "duration; at least "+str(T_needed_initially)+" s are needed. "  +\
            "One iteration was placed later than its assumed scheduled time "+\
            "T = "+str(T_next_originally)+" s, because the flag" +\
            " \"repetition_rate_constraint_is_hard_and_may_not_be_broken\"" +\
            " was set to False. The event was placed at T = "+str(T)+" s.")
        
    return T, repetition_counter