#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
from numpy import hanning as von_hann
from math import isnan
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Example conditional- and cross-Ramsey data from the 2024 paper
# by Christian Križan et al. Specifically, Fig. 3., or whichever number
# of the plot that shows the conditional and cross-Ramsey data.
CZ1_P01_post_processed_data = [-0.01005644,-0.00451663,0.01570832,0.08346322,0.15173938,0.21286713,
0.34863982,0.46415354,0.50316574,0.65062938,0.71829164,0.72698084,
0.82704061,0.89276479,0.94675166,0.97476329,0.97733097,0.9091982,
0.90094301,0.82802655,0.72488852,0.64496045,0.54775771,0.45515106,
0.34700856,0.25104969,0.16490552,0.086367,0.01950416,-0.02135357,
-0.02750581]
CZ1_P11_post_processed_data = [0.98057501,0.94569391,0.93248595,0.86679817,0.73684822,0.74884198,
0.67058492,0.54654395,0.45477807,0.36101731,0.23099006,0.11435327,
0.03263705,0.00522372,-0.03999218,-0.04560249,-0.03876028,-0.00259934,
0.04739105,0.14845666,0.22634234,0.40515079,0.49938203,0.61361811,
0.66012964,0.8237349,0.8746499,0.91803652,0.66884165,0.9936905,
0.75359363]
CZ2_P10_post_processed_data = [-0.0024222,-0.00361658,0.02628507,0.0627301,0.11563789,0.22250639,
0.30696278,0.41440996,0.51184705,0.63296035,0.6804796,0.8212281,
0.89631902,0.97410287,1.03274321,0.99233296,1.05186728,0.9491235,
0.88082179,0.84243734,0.7621284,0.69106084,0.61666765,0.49760539,
0.3802999,0.27205965,0.18569945,0.12826084,0.06994152,-0.01196582,
0.03524208]
CZ2_P11_post_processed_data = [0.7145775,0.78547298,1.00504825,0.92561055,0.83503355,0.73169683,
0.65751485,0.52765164,0.38962627,0.29370699,0.16273131,0.09090567,
0.02936981,-0.03979862,-0.05728235,-0.06676103,-0.04638153,0.01580418,
0.04304009,0.11992051,0.20349923,0.31493536,0.39982435,0.54818091,
0.63718184,0.74477318,0.49323495,0.66857716,0.72610843,0.73161532,
0.60270365]
iSWAP1_P10_post_processed_data = [0.62867282,0.83231377,0.9922561,1.03239365,
1.03545558,0.89477493,0.80983717,0.54542092,0.32522569,0.13306179,
0.00495052,-0.02554669,0.04195472,0.17799122,0.4124068]
iSWAP1_P11_post_processed_data = [0.4119812,0.2372616,0.04680503,0.03630252,
0.0983245,0.12112374,0.38738618,0.67813224,0.82684326,0.96261935,
1.03817367,1.08762499,0.99539436,0.85481721,0.61470361]
iSWAP2_P01_post_processed_data = [0.54660501,0.77056781,0.91910785,1.00443943,
0.99017617,0.91757923,0.74666958,0.53805074,0.31951892,0.13717403,
0.01326409,-0.02927062,0.02662306,0.15844817,0.34262769]
iSWAP2_P11_post_processed_data = [0.47358658,0.26036911,0.0952488,0.0369185,
0.00959251,0.13199478,0.28299943,0.4917019,0.72361076,0.90941971,
1.01757679,1.06230023,1.02015774,0.89125994,0.66987237]
SWAP1_P10_post_processed_data = [-0.00641662,0.05197203,0.12880772,0.29646478,
0.57576736,0.76167562,0.86593158,0.89397389,0.75681553,0.66804217,
0.40154822,0.28564834,0.18073173,0.09161637,0.02763258]
SWAP1_P11_post_processed_data = [0.04313859,0.13416431,0.22578824,0.45232721,
0.57508113,0.80592625,0.99146983,0.84837992,0.78483127,0.71954212,
0.62114543,0.36890334,0.19069708,0.14606401,0.03629693]
SWAP2_P01_post_processed_data = [0.05977576,0.14051941,0.23626235,0.37075796,
0.46225366,0.67496267,0.9516245,1.02321066,1.03201483,0.94070706,
0.77281098,0.52847637,0.24221973,0.16003862,0.05397206]
SWAP2_P11_post_processed_data = [0.05628305,0.03689407,0.1387167,0.37555339,
0.56060679,0.8260802,0.93367065,1.06601309,1.04277318,0.85901099,
0.66783281,0.42124768,0.24935897,0.16527482,0.05916419]

## The cross-Ramsey data for fig. 3 was taken in experiments where the phase
## for the iSWAP and SWAP cross-Ramsey was swept from -2 pi to +2 pi radians.
## Thus, creating duplicate experiments in a way. But here are those
## -2 pi / +2 pi cross-Ramsey vectors in case you're curious about them,
## feel free! Just remember below that they will have to be fitted to
## an ideal waveform that does two periods inside the plot window;
## the frequency of the ideal wave is doubled.
iSWAP1_P10_post_processed_data_full2pi = [0.47413015,0.35594626,0.12668214,0.02556512,-0.02410413,0.04676312,
0.18100952,0.45436439,0.62867282,0.83231377,0.9922561,1.03239365,
1.03545558,0.89477493,0.80983717,0.54542092,0.32522569,0.13306179,
0.00495052,-0.02554669,0.04195472,0.17799122,0.4124068,0.61837161,
0.83589122,0.98242357,0.98721949,0.98195232,0.88564226,0.73429002,
0.49380378]
iSWAP1_P11_post_processed_data_full2pi = [0.54131024,0.80709709,0.98782762,1.09053211,1.09987576,1.03602638,
0.86601662,0.64375274,0.4119812,0.2372616,0.04680503,0.03630252,
0.0983245,0.12112374,0.38738618,0.67813224,0.82684326,0.96261935,
1.03817367,1.08762499,0.99539436,0.85481721,0.61470361,0.38261737,
0.16480716,0.03279242,-0.03778708,0.0360156,0.12222725,0.20573944,
0.38978241]
iSWAP2_P01_post_processed_data_full2pi = [0.48925808,0.28565221,0.13272419,0.02086279,-0.016506,0.03150106,
0.15423131,0.33349905,0.54660501,0.77056781,0.91910785,1.00443943,
0.99017617,0.91757923,0.74666958,0.53805074,0.31951892,0.13717403,
0.01326409,-0.02927062,0.02662306,0.15844817,0.34262769,0.56164662,
0.76615586,0.87334057,1.01639014,0.99884703,0.91964751,0.77029446,
0.54313556]
iSWAP2_P11_post_processed_data_full2pi = [0.44130375,0.55193957,0.93417488,1.05591987,1.10147792,1.04426704,
0.88083057,0.70229403,0.47358658,0.26036911,0.0952488,0.0369185,
0.00959251,0.13199478,0.28299943,0.4917019,0.72361076,0.90941971,
1.01757679,1.06230023,1.02015774,0.89125994,0.66987237,0.46555623,
0.23122805,0.08594597,0.0113261,0.03688207,0.1134722,0.25707748,
0.3340285]
SWAP1_P10_post_processed_data_full2pi = [0.80294135,0.74861874,0.55529051,0.46620492,0.29125942,0.20226609,
0.06166923,-0.00749884,-0.00641662,0.05197203,0.12880772,0.29646478,
0.57576736,0.76167562,0.86593158,0.89397389,0.75681553,0.66804217,
0.40154822,0.28564834,0.18073173,0.09161637,0.02763258,0.03137087,
0.11433328,0.27010258,0.42595563,0.57716892,0.63573372,0.76092213,
0.97649214]
SWAP1_P11_post_processed_data_full2pi = [0.87623214,0.69845793,0.69088341,0.65006421,0.4203868,0.24639073,
0.1053249,0.03981902,0.04313859,0.13416431,0.22578824,0.45232721,
0.57508113,0.80592625,0.99146983,0.84837992,0.78483127,0.71954212,
0.62114543,0.36890334,0.19069708,0.14606401,0.03629693,0.02366259,
0.11149788,0.26193286,0.44287849,0.6455607,0.82419926,0.95598176,
1.0591649]
SWAP2_P01_post_processed_data_full2pi = [1.13223505,1.06815563,0.96141375,0.77504374,0.55489797,0.29479198,
0.13492684,0.01159603,0.05977576,0.14051941,0.23626235,0.37075796,
0.46225366,0.67496267,0.9516245,1.02321066,1.03201483,0.94070706,
0.77281098,0.52847637,0.24221973,0.16003862,0.05397206,0.06701597,
0.10095362,0.26096266,0.3624135,0.53052506,0.76297562,0.90763776,
0.81592737]
SWAP2_P11_post_processed_data_full2pi = [0.90021981,0.95379462,0.77056948,0.62370225,0.49651858,0.28124096,
0.17014227,0.05393411,0.05628305,0.03689407,0.1387167,0.37555339,
0.56060679,0.8260802,0.93367065,1.06601309,1.04277318,0.85901099,
0.66783281,0.42124768,0.24935897,0.16527482,0.05916419,0.0296857,
0.08950563,0.26927522,0.35753837,0.54635654,0.67906718,0.80410651,
0.83353718]

def fit_CZ_iSWAP_SWAP_and_acquire_MSE(
    data_to_fit,
    qc_was_prepared_in_1 = False,
    two_qubit_gate = "CZ"
    ):
    ''' 
        qc_was_prepared_in_1: sets whether the input state of the control qubit
        is |0⟩ (False) or |1⟩ (True).
        
        Legal syntax for two_qubit_gate is "CZ", "iSWAP", or "SWAP".
    '''
    
    # Check legal syntax for experiment type.
    if not ( (two_qubit_gate == "CZ") or (two_qubit_gate == "iSWAP") or (two_qubit_gate == "SWAP") ):
        raise ValueError("Halted! Could not understand your input argument for \"two_qubit_gate\". Legal arguments are \"CZ\", \"iSWAP\", or \"SWAP\" only.")
    
    # Guess initial values. These ideal initial values will be identical
    # for all conditional and cross-Ramsey experiments.
    x = np.linspace(-np.pi, np.pi, len(data_to_fit))
    pop_swing = 1.0
    f = 1/(2*np.pi) # The experiment is swept from -π to +π rad Vz phase.
    pop_phase = 0.0
    pop_offset = 0.5
    
    # Fit CZ or SWAP?
    if (two_qubit_gate == "CZ") or (two_qubit_gate == "SWAP"):
        
        # Perform "triggered" fit, or "nontriggered" fit?
        ## i.e. was the control qubit prepared in |0⟩ or |1⟩?
        if (not qc_was_prepared_in_1) or (two_qubit_gate == "SWAP"):
            # Get ideal curve for later calculating the MSE.
            ideal_curve = expected_ideal_CZ_nontriggered_or_SWAP(
                x = x,
                pop_swing = pop_swing,
                f = f,
                pop_phase = pop_phase,
                pop_offset = pop_offset
            )
            # The two-qubit gate did not trigger.
            ## Or, the ideal was the expected outcome of the SWAP experiment.
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = expected_ideal_CZ_nontriggered_or_SWAP,
                xdata = x,
                ydata = data_to_fit,
                p0    = (pop_swing, f, pop_phase, pop_offset)
            )
        else:
            # The two-qubit gate triggered.
            # Get ideal curve for later calculating the MSE.
            ideal_curve = expected_ideal_CZ_triggered(
                x = x,
                pop_swing = pop_swing,
                f = f,
                pop_phase = pop_phase,
                pop_offset = pop_offset
            )
            # The two-qubit gate did trigger.
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = expected_ideal_CZ_triggered,
                xdata = x,
                ydata = data_to_fit,
                p0    = (pop_swing, f, pop_phase, pop_offset)
            )
    
    elif two_qubit_gate == "iSWAP":
        # Perform which fit? This is determined by the input state.
        if not qc_was_prepared_in_1:
            # Get ideal curve for later calculating the MSE.
            ideal_curve = expected_ideal_iSWAP_nontriggered(
                x = x,
                pop_swing = pop_swing,
                f = f,
                pop_phase = pop_phase,
                pop_offset = pop_offset
            )
            # The two-qubit gate did not trigger.
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = expected_ideal_iSWAP_nontriggered,
                xdata = x,
                ydata = data_to_fit,
                p0    = (pop_swing, f, pop_phase, pop_offset)
            )
        else:
            # The two-qubit gate triggered.
            # Get ideal curve for later calculating the MSE.
            ideal_curve = expected_ideal_iSWAP_triggered(
                x = x,
                pop_swing = pop_swing,
                f = f,
                pop_phase = pop_phase,
                pop_offset = pop_offset
            )
            # The two-qubit gate did trigger.
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = expected_ideal_iSWAP_triggered,
                xdata = x,
                ydata = data_to_fit,
                p0    = (pop_swing, f, pop_phase, pop_offset)
            )
        
    else:
        raise ValueError("Could not recognise the selected two-qubit gate type: \""+str(two_qubit_gate)+"\"")
    
    ## Get values and error bars!
    
    # covariance_mtx_of_opt_vals is the covariance matrix of optimal_vals.
    # These values are optimised for minimising (residuals)^2 of
    # fit_function(x, *optimal_vals) -y. Our error bars are found thusly:
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    
    pop_swing     = optimal_vals[0]
    err_pop_swing = fit_err[0]
    print("pop_swing = "+str(pop_swing)+" ±"+str(err_pop_swing))
    
    f     = optimal_vals[1]
    err_f = fit_err[1]
    print("f = "+str(f)+" ±"+str(err_f))
    
    pop_phase     = optimal_vals[2]
    err_pop_phase = fit_err[2]
    DELTA_pop_phase = 0.0 - pop_phase
    print("pop_phase = "+str(pop_phase)+" ±"+str(err_pop_phase)+", Δ is: "+str(DELTA_pop_phase))
    
    pop_offset     = optimal_vals[3]
    err_pop_offset = fit_err[3]
    DELTA_pop_offset = 0.5 - pop_offset
    print("pop_offset = "+str(pop_offset)+" ±"+str(err_pop_offset)+", Δ is: "+str(DELTA_pop_offset))
    
    # Calculate MSE.
    sum_MSE = 0
    for ii in range(len(data_to_fit)):
        observed_val = data_to_fit[ii]
        ideal_val    = ideal_curve[ii]
        sum_MSE += (observed_val - ideal_val)**2
    MSE = sum_MSE / len(data_to_fit)
    print("Mean squared error = "+str(MSE))
    
    # Done!
    return optimal_vals, fit_err, MSE
    

## Define functions to fit against.

def expected_ideal_CZ_nontriggered_or_SWAP(
    x,
    pop_swing,
    f,
    pop_phase,
    pop_offset
    ):
    ''' Function to be fitted against.
    '''
    return (0.5*pop_swing) * np.cos(2*np.pi * f * x + pop_phase) + pop_offset

def expected_ideal_CZ_triggered(
    x,
    pop_swing,
    f,
    pop_phase,
    pop_offset
    ):
    ''' Function to be fitted against.
    '''
    return -1 * (0.5*pop_swing) * np.cos(2*np.pi * f * x + pop_phase) + pop_offset

def expected_ideal_iSWAP_nontriggered(
    x,
    pop_swing,
    f,
    pop_phase,
    pop_offset
    ):
    ''' Function to be fitted against.
    '''
    return -1 * (0.5*pop_swing) * np.sin(2*np.pi * f * x + pop_phase) + pop_offset
    
def expected_ideal_iSWAP_triggered(
    x,
    pop_swing,
    f,
    pop_phase,
    pop_offset
    ):
    ''' Function to be fitted against.
    '''
    return (0.5*pop_swing) * np.sin(2*np.pi * f * x + pop_phase) + pop_offset
