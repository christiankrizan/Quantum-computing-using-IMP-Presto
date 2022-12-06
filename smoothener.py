#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
from numpy import hanning as von_hann
from math import isnan

def filter_and_interpolate(
    datapoints_to_filter_and_interpolate,
    number_of_times_to_filter_and_interpolate,
    ):
    ''' Grab input data, smoothen this data with itself, and interpolate
        the missing points.
    '''
    
    # The datapoint curve will very likely be very noisy. Clean it up a bit,
    # and interpolate this curve so that the number of datapoints match.
    
    ## FILTER!
    for ii in range(number_of_times_to_filter_and_interpolate):
        filtered_datapoints = []
        if len(datapoints_to_filter_and_interpolate) > 2:
            for curr_datapoint in range(0,len(datapoints_to_filter_and_interpolate),2):
                next_val = (datapoints_to_filter_and_interpolate[curr_datapoint] + datapoints_to_filter_and_interpolate[curr_datapoint+1])/2
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
        if ii != (number_of_times_to_filter_and_interpolate-1):
            datapoints_to_filter_and_interpolate = interpolated_filtered_datapoints
    
    # At this point, we've made the interpolated, filtered curve of the
    # input datapoints. Let's numpy-ify this curve.
    interpolated_filtered_datapoints = np.array(interpolated_filtered_datapoints)
    
    # Return
    return interpolated_filtered_datapoints