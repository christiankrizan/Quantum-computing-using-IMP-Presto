#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
from datetime import datetime

def check_if_integration_window_is_legal(
    sample_rate,
    sampling_duration,
    integration_window_start,
    integration_window_stop
    ):
    ''' Verify that the integration window falls within the scoped data.
        And, that the integration window is longer than one strike of
        the sample clock.
    '''
    # Based on the sample period, increase the integration window stop
    # if it is impossible to scope.
    if (integration_window_stop - integration_window_start) < (1/sample_rate):
        integration_window_stop = integration_window_start + (1/sample_rate)
        print("Warning: an impossible integration window was defined. " + \
        "The window stop was moved to "+str(integration_window_stop)    + \
        " seconds.")
    
    # Construct the expected resulting time trace (the time axis)
    # and verify whether the sampling window fits.
    time_vector = np.linspace( \
        0.0, sampling_duration, int(sampling_duration * sample_rate))
    assert integration_window_start >= time_vector[0], \
        "Error! The requested integration window begins before the first " + \
        "sample of the scoped data."
    assert integration_window_stop  <= time_vector[-1], \
        "Error! The requested integration window stops after the last " + \
        "sample of the scoped data. The integration window stop is at " + \
        str(integration_window_stop)+" seconds, while the last sample " + \
        "of the scoped data is at "+str(time_vector[-1])+" seconds."
    
    # Return the possibly updated integration window stop.
    return integration_window_stop

def show_user_time_remaining(seconds):
    ''' Take some number of seconds remaining for some measurement
        to complete, and print the result in a human-legible form.
    '''
    calc = seconds
    row_of_text_to_print = ""
    if calc < 60.0:
        calc_s = calc
        row_of_text_to_print = str(round(calc_s,2))+" second(s)."
    elif calc < 3600.0:
        calc_m = int(calc // 60)
        calc_s = calc -(calc_m * 60)
        row_of_text_to_print = str(calc_m)+" minute(s), "+str(round(calc_s,2))+" seconds."
    elif calc < 86400.0:
        calc_h = int(calc // 3600)
        calc_m = int((calc -(calc_h * 3600)) // 60)
        calc_s = calc -(calc_h * 3600) -(calc_m * 60)
        row_of_text_to_print = str(calc_h)+" hour(s), "+str(calc_m)+" minutes, "+str(round(calc_s,2))+" seconds."
    elif calc < 604800:
        calc_d = int(calc // 86400)
        calc_h = int((calc -(calc_d * 86400)) // 3600)
        calc_m = int((calc -(calc_d * 86400) -(calc_h * 3600)) // 60)
        calc_s =  calc -(calc_d * 86400) -(calc_h * 3600) -(calc_m * 60)
        row_of_text_to_print = str(calc_d)+" day(s), "+str(calc_h)+" hours, "+str(calc_m)+" minutes, "+str(round(calc_s,2))+" seconds."
    elif calc < 2629743.83:
        calc_w = int(calc // 604800)
        calc_d = int((calc -(calc_w * 604800)) // 86400)
        calc_h = int((calc -(calc_w * 604800) -(calc_d * 86400)) // 3600)
        calc_m = int((calc -(calc_w * 604800) -(calc_d * 86400) -(calc_h * 3600)) // 60)
        calc_s =  calc -(calc_w * 604800) -(calc_d * 86400) -(calc_h * 3600) -(calc_m * 60)
        row_of_text_to_print = str(calc_w)+" week(s), "+str(calc_d)+" days, "+str(calc_h)+" hours, "+str(calc_m)+" minutes, "+str(round(calc_s,2))+" seconds."
    
    # Print message!
    if len(row_of_text_to_print) > 30:
        length_of_string_to_print = len(row_of_text_to_print)
    else:
        length_of_string_to_print = 30
    print("\n\n" + "#" * length_of_string_to_print + "\nEstimated true time remaining:")
    print(row_of_text_to_print)
    print("#" * length_of_string_to_print)

def get_timestamp_string(pretty = False):
    ''' Return an appropriate timestamp string.
    '''
    if (not pretty):
        return (datetime.now()).strftime("%d-%b-%Y_(%H_%M_%S)")
    else:
        return (datetime.now()).strftime("%d-%b-%Y (%H:%M:%S)")