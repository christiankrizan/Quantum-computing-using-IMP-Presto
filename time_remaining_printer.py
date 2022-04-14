#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

def show_user_time_remaining(seconds):
    ''' Take some number of seconds remaining for some measurement
        to complete, and print the result in a human-legible form.
    '''
    calc = seconds
    print("\n\n##############################\nEstimated true time remaining:")
    if calc < 60.0:
        calc_s = calc
        print(str(round(calc_s,2))+" second(s).")
    elif calc < 3600.0:
        calc_m = int(calc // 60)
        calc_s = calc -(calc_m * 60)
        print(str(calc_m)+" minute(s), "+str(round(calc_s,2))+" seconds.")
    elif calc < 86400.0:
        calc_h = int(calc // 3600)
        calc_m = (calc -(calc_h * 3600)) // 60
        calc_s = calc -(calc_h * 3600) -(calc_m * 60)
        print(str(calc_h)+" hour(s), "+str(calc_m)+" minutes, "+str(round(calc_s,2))+" seconds.")
    elif calc < 604800:
        calc_d = int(calc // 86400)
        calc_h = (calc -(calc_d * 86400)) // 3600
        calc_m = (calc -(calc_d * 86400) -(calc_h * 3600)) // 60
        calc_s =  calc -(calc_d * 86400) -(calc_h * 3600) -(calc_m * 60)
        print(str(calc_d)+" day(s), "+str(calc_h)+" hours, "+str(calc_m)+" minutes, "+str(round(calc_s,2))+" seconds.")
    elif calc < 2629743.83:
        calc_w = int(calc // 604800)
        calc_d = (calc -(calc_w * 604800)) // 86400
        calc_h = (calc -(calc_w * 604800) -(calc_d * 86400)) // 3600
        calc_m = (calc -(calc_w * 604800) -(calc_d * 86400) -(calc_h * 3600)) // 60
        calc_s =  calc -(calc_w * 604800) -(calc_d * 86400) -(calc_h * 3600) -(calc_m * 60)
        print(str(calc_w)+" week(s), "+str(calc_d)+" days, "+str(calc_h)+" hours, "+str(calc_m)+" minutes, "+str(round(calc_s,2))+" seconds.")
    print("##############################\n")