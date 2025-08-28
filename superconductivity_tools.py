#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

from random import randint
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys
import os
import re
import colorsys # Used for generating curve colours.
import time as time_module
from scipy.optimize import curve_fit
from scipy.stats import moment
from scipy.stats import linregress
from scipy.interpolate import interp1d
from datetime import datetime
from collections import deque

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in rgb))

def interpolate_hsv_colours(start_hex, end_hex, n):
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)

    start_hsv = colorsys.rgb_to_hsv(*start_rgb)
    end_hsv = colorsys.rgb_to_hsv(*end_rgb)

    # Handle hue circular interpolation
    h1, h2 = start_hsv[0], end_hsv[0]
    if abs(h1 - h2) > 0.5:
        if h1 > h2:
            h2 += 1
        else:
            h1 += 1

    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        h = (h1 + t * (h2 - h1)) % 1.0
        s = start_hsv[1] + t * (end_hsv[1] - start_hsv[1])
        v = start_hsv[2] + t * (end_hsv[2] - start_hsv[2])
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb_to_hex(rgb))

    return colors

def get_colourise(
    colourised_counter
    ):
    ''' If using patterned colours, return an appropriate colour.
       -2.x:  White
       -1.x:  Regular Black
        0.x:  Kite Blue
        1.x:  Sunny Yellow
        2.x:  Raspberry Red
        3.x:  Frog Green
    '''
    
    # WHITE TONES
    if ((colourised_counter <= -2) and (colourised_counter > -3)):
        if   colourised_counter == -2.1:
            return "#F2F1ED"
        elif colourised_counter == -2.2:
            return "#F2F1ED"
        elif colourised_counter == -2.3:
            return "#F2F1ED"
        elif colourised_counter == -2.4:
            return "#F2F1ED"
        else:
            # Covers the .0 case too.
            return "#F2F1ED"
    
    # REGULAR BLACK TONES
    if ((colourised_counter <= -1) and (colourised_counter > -2)):
        if   colourised_counter == -1.1:
            return "#4D4A49"
        elif colourised_counter == -1.2:
            return "#777373"
        elif colourised_counter == -1.3:
            return "#9F9E9B"
        elif colourised_counter == -1.4:
            return "#C9C7C5"
        else:
            # Covers the .0 case too.
            return "#242021"
    
    # KITE BLUE TONES
    elif ((colourised_counter >= 0) and (colourised_counter < 1)):
        if   colourised_counter == 0.1:
            return "#6EBAE0"
        elif colourised_counter == 0.2:
            return "#90C8E4"
        elif colourised_counter == 0.3:
            return "#B0D6E6"
        elif colourised_counter == 0.4:
            return "#D2E4EA"
        else:
            # Covers the .0 case too.
            return "#4EADDD"
    
    # SUNNY YELLOW TONES
    elif ((colourised_counter >= 1) and (colourised_counter < 2)):
        if   colourised_counter == 1.1:
            return "#F6D330"
        elif colourised_counter == 1.2:
            return "#F5DA60"
        elif colourised_counter == 1.3:
            return "#F4E38E"
        elif colourised_counter == 1.4:
            return "#F3EABE"
        else:
            # Covers the .0 case too.
            return "#F7CC01"
    
    # RASPBERRY RED TONES
    elif ((colourised_counter >= 2) and (colourised_counter < 3)):
        if   colourised_counter == 2.1:
            return "#EE635B"
        elif colourised_counter == 2.2:
            return "#F08680"
        elif colourised_counter == 2.3:
            return "#F0ABA4"
        elif colourised_counter == 2.4:
            return "#F2CEC9"
        else:
            # Covers the .0 case too.
            return "#EE4037"
    
    # FROG GREEN TONES
    elif ((colourised_counter >= 3) and (colourised_counter < 4)):
        if   colourised_counter == 3.1:
            return "#6EBA30"
        elif colourised_counter == 3.2:
            return "#90C860"
        elif colourised_counter == 3.3:
            return "#B0D68E"
        elif colourised_counter == 3.4:
            return "#D2E4BE"
        else:
            # Covers the .0 case too.
            return "#4EAD01"
    
    # ERROR
    else:
        # Default!
        return '#000000'

def calculate_f01_from_RT_resistance(
    room_temperature_resistance,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    verbose = True
    ):
    ''' Given a room temperature resistance, calculate the resulting f_01.
        For difference_between_RT_and_cold_resistance, a value of 1.1385
        means that a cold junction is 13.85 % more resistive than a room
        temperature one. This number is the average of the two junctions
        that were measured in Fig. 2.12 by A. Osman's thesis.
        
        The thesis is OF COURSE not uploaded to Chalmers ODR archive as of
        2025-01-31, that would require somebody to actually know anything
        about archiving practices and university rules at that department's
        division. Lol, ngh. Find the thesis here:
        https://research.chalmers.se/en/publication/543784
        
        E_C is the transmon's charging energy.
        Delta_cold is the superconducting gap at millikelvin temperatures.
        T is the temperature of operation, typically dilution fridge
          temperatures. For instance, 10 mK.
    '''
    
    # Physical constants
    h = 6.62607015e-34       # Planck's constant [J/Hz]
    h_bar = h / (2 * np.pi)  # Reduced Planck's constant [J/Hz]
    e = 1.602176634e-19      # Elementary charge [C]
    k_B = 1.380649e-23       # Boltzmann's constant [J/K]
    
    # User-set values
    ## Calculate the normal state resistance of the S-I-S junction.
    R_N = room_temperature_resistance * difference_between_RT_and_cold_resistance
    Delta_cold = Delta_cold_eV * e # Superconducting gap at mK temperature [J]
    E_C = E_C_in_Hz * h # Charging energy [J]
    
    # Calculate I_c using the Ambegaokar-Baratoff relation
    I_c = (np.pi * Delta_cold)/(2*e*R_N) * np.tanh(Delta_cold / (2 * k_B * T))
    
    # Calculate E_J
    E_J = (h_bar / (2*e)) * I_c
    
    # Print E_C and E_J?
    if verbose:
    
        ## Print E_C
        if (E_C/h) > 1e9:
            print("E_C is [GHz]: "+str((E_C/h)/1e9))
        elif (E_C/h) > 1e6:
            print("E_C is [MHz]: "+str((E_C/h)/1e6))
        elif (E_C/h) > 1e3:
            print("E_C is [kHz]: "+str((E_C/h)/1e3))
        else:
            print("E_C is [Hz]: "+str(E_C/h))
        
        ## Print E_J
        if (E_J/h) > 1e9:
            print("E_J is [GHz]: "+str((E_J/h)/1e9))
        elif (E_J/h) > 1e6:
            print("E_J is [MHz]: "+str((E_J/h)/1e6))
        elif (E_J/h) > 1e3:
            print("E_J is [kHz]: "+str((E_J/h)/1e3))
        else:
            print("E_J is [Hz]: "+str(E_J/h))
        
        ## Print E_J / E_C
        print("E_J / E_C is: "+str(E_J/E_C))
    
    # Calculate f_01
    '''## Koch 2007 equation regarding transmon f_01, precision to second order.
    ## https://doi.org/10.1103/PhysRevB.77.180502
    second_order_correction_factor = -(E_C / 2) * (E_C / (8*E_J))'''
    # Updated calculation from
    xi = np.sqrt(2 * E_C / E_J)
    second_order_correction_factor = (1 + (1/4)*xi + (21/128)*(xi**2))
    f_01 = (np.sqrt(8 * E_J * E_C) - E_C * second_order_correction_factor)/h
    
    # Return value!
    return f_01

def calculate_RT_resistance_from_target_f01(
    target_f_01,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    R_N_initial_guess = 15000,
    acceptable_frequency_offset = 100,
    verbose = True
    ):
    ''' Given a target |0⟩ → |1⟩ transition of a transmon,
        calculate what room-temperature resistance the Josephson junction
        should have.
        
        First, read the description for the function
        "calculate_f01_from_RT_resistance" above.
        
        Now, instead of fighting quartic functions in finding out the
        inverse of the f_01 equation, this function uses a different
        approach. As in, guessing different resistances until a
        match is found.
        
        acceptable_frequency_offset [Hz] is the difference after which
        the function will stop.
        
        R_N_initial_guess [Ω] is the initial resistance guess where we'll
        begin.
    '''
    
    done = False
    r_rt = R_N_initial_guess
    while(not done):
        
        # Try!
        result_freq = calculate_f01_from_RT_resistance(
            room_temperature_resistance = r_rt,
            E_C_in_Hz = E_C_in_Hz,
            Delta_cold_eV = Delta_cold_eV,
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = False
        )
        
        # Find difference.
        ## Positive difference: the target freq is above your calculated value.
        ## Negative difference: the target freq is below your calculated value.
        difference = target_f_01 - result_freq
        
        # Finished?
        if (np.abs(difference) < np.abs(acceptable_frequency_offset)):
            done = True
        else:
            if difference > 0:
                r_rt *= 0.99
            elif difference < 0:
                r_rt *= 1.01
    
    # Print some values.
    if verbose:
        calculate_f01_from_RT_resistance(
            room_temperature_resistance = r_rt,
            E_C_in_Hz = E_C_in_Hz,
            Delta_cold_eV = Delta_cold_eV,
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = True
        )
    
    # Return value!
    return r_rt

def calculate_resistance_to_manipulate_to(
    target_f_01,
    E_C_in_Hz,
    Delta_cold_eV,
    original_resistance_of_junction = 0,
    expected_aging = 0,
    expected_resistance_relaxation = 0,
    verbose = True
    ):
    ''' Given a target frequency, calculate what resistance should be
        manipulated to, including resistance relaxation effects.
        
        The units of expected_aging and expected_resistance_relaxation are linear.
        I.e., 1.05 means that you expect an additional 5 % extra resistance.
    '''
    
    # Get the room temperature resistance that
    # this target frequency corresponds to.
    room_temperature_resistance_to_hit = calculate_RT_resistance_from_target_f01(
        target_f_01 = target_f_01,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        verbose = verbose
    )
    
    # Print to user?
    if verbose:
        # Original junction resistance known?
        if original_resistance_of_junction > 0:
            increase = room_temperature_resistance_to_hit / original_resistance_of_junction
            assert increase >= 1.0, "Error! The resistance can only go up (for now). Unable to tune junction to the target frequency. Needed tuning: "+str((increase-1)*100)+" %"
            print(f"The target room-temperature resistance is: {(room_temperature_resistance_to_hit):.3f} [Ω], which corresponds to {((increase-1)*100):.3f} %")
        else:
            print(f"The target room-temperature resistance is: {(room_temperature_resistance_to_hit):.3f} [Ω]")
    
    # Does the sample have some aging left to do?
    ## TODO
    if expected_aging > 0:
        raise NotImplementedError("Not finished.")
    
    # TODO current, our only knowledge of the relaxation is that it is a fixed offset compared to the initial resistance.
    ## Well, to be picky, we know that the end number is pretty much
    ## a fixed resistance offset.
    if expected_resistance_relaxation >= 0:
        if original_resistance_of_junction == 0:
            raise ValueError("Halted! If assuming relaxation, then the original resistance of the junction must be known. Check your arguments.")
        
        print(f"Expecting {((expected_resistance_relaxation-1)*100):.3f} % worth of resistance relaxation.")
        
        # Figure out the relaxation.
        relaxation_in_ohms = (original_resistance_of_junction * expected_resistance_relaxation - original_resistance_of_junction)
        
        # Subtract!
        room_temperature_resistance_to_hit -= relaxation_in_ohms
        increase = room_temperature_resistance_to_hit / original_resistance_of_junction
        print(f"Excluding resistance relaxation, expect to hit: {(room_temperature_resistance_to_hit):.3f} [Ω], which is {((increase-1)*100):.3f} %")
        
    elif expected_resistance_relaxation < 0:
        raise ValueError("Error! The resistance relaxation is expected to be a positive number; the resistance is expected to increase post-manipulation.")
    
    return room_temperature_resistance_to_hit

def fit_ambegaokar_baratoff_josephson_koch_to_resistance(
    measured_junction_resistances,
    measured_qubit_frequencies,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    colourise = False,
    verbose = True
    ):
    ''' Given measured frequency and resistance values,
        fit the equation that maps frequency to resistance.
    '''
    
    # User argument sanitation:
    if not (len(measured_qubit_frequencies) == len(measured_junction_resistances)):
        raise ValueError("Halted! The list measured_qubit_frequencies must have an equal number of entries as the list measured_junction_resistances.")
    
    # Sort the lists. Sort by the first list, and unzip them
    sorted_pairs = sorted(zip(measured_junction_resistances, measured_qubit_frequencies))
    sorted_resistances, sorted_frequencies = zip(*sorted_pairs)
    measured_junction_resistances = (list(sorted_resistances)).copy()
    measured_qubit_frequencies    = (list(sorted_frequencies)).copy()
    
    # Define the equation to fit to.
    def ambegaokar_baratoff_josephson_koch(
        room_temperature_resistance,
        E_C_in_Hz,
        Delta_cold_eV,
        ##difference_between_RT_and_cold_resistance,
        ##T
        ):
        # Physical constants
        h = 6.62607015e-34       # Planck's constant [J/Hz]
        h_bar = h / (2 * np.pi)  # Reduced Planck's constant [J/Hz]
        e = 1.602176634e-19      # Elementary charge [C]
        k_B = 1.380649e-23       # Boltzmann's constant [J/K]
        
        # User-set values
        ## Calculate the normal state resistance of the S-I-S junction.
        R_N = room_temperature_resistance * difference_between_RT_and_cold_resistance
        Delta_cold = Delta_cold_eV * e # Superconducting gap at mK temperature [J]
        E_C = E_C_in_Hz * h # Charging energy [J]
        
        # Calculate I_c using the Ambegaokar-Baratoff relation
        I_c = (np.pi * Delta_cold)/(2*e*R_N) ## * np.tanh(Delta_cold / (2 * k_B * T))
        
        # Calculate E_J
        E_J = (h_bar / (2*e)) * I_c
        
        # Calculate f_01
        ## Koch 2007 equation regarding transmon f_01, precision to second order.
        ## https://doi.org/10.1103/PhysRevB.77.180502
        second_order_correction_factor = -(E_C / 2) * (E_C / (8*E_J))
        
        # Return answer.
        return (np.sqrt(8 * E_J * E_C) -E_C + second_order_correction_factor)/h
    
    # Create figure for plotting.
    if verbose:
        if colourise:
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=get_colourise(-2))
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the data, and its fitted-to curve,
    ## Get the fit and its data.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = ambegaokar_baratoff_josephson_koch,
        xdata = measured_junction_resistances,
        ydata = measured_qubit_frequencies,
        p0    = (E_C_in_Hz, Delta_cold_eV)
    )
    ###(E_C_in_Hz, Delta_cold_eV, difference_between_RT_and_cold_resistance, T)
    fit_x_values = np.linspace(measured_junction_resistances[0]*0.90, measured_junction_resistances[-1]*1.10, 100)
    fitted_curve = ambegaokar_baratoff_josephson_koch(
        room_temperature_resistance = fit_x_values,
        E_C_in_Hz = optimal_vals[0],
        Delta_cold_eV = optimal_vals[1],
        ##difference_between_RT_and_cold_resistance = optimal_vals[2],
        ##T = optimal_vals[3],
    )
    
    # Get the fit error.
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    err_E_C_in_Hz = fit_err[0]
    err_Delta_cold_eV = fit_err[1]
    ##err_difference_between_RT_and_cold_resistance = fit_err[2]
    ##err_T = fit_err[3]
    
    # Do teh plottings!
    if verbose:
        if colourise:
            plt.scatter(measured_junction_resistances, measured_qubit_frequencies, color=get_colourise(colourise_counter), label=f"Measured data.")
            colourise_counter += 1
            ## Plot the ideal curve.
            plt.plot(fit_x_values, fitted_curve, color=get_colourise(colourise_counter))
            colourise_counter += 1
        else:
            plt.scatter(measured_junction_resistances, measured_qubit_frequencies, color="#34D2D6", label=f"Measured data")
            plt.plot(fit_x_values, fitted_curve, '--', color="#D63834")
    
    # Labels and such.
    if verbose:
        plt.grid()
        if colourise:
            fig.patch.set_alpha(0)
            ax.grid(color=get_colourise(-1))
            ax.set_facecolor(get_colourise(-2))
            ax.spines['bottom'].set_color(get_colourise(-1))
            ax.spines['top'].set_color(get_colourise(-1))
            ax.spines['left'].set_color(get_colourise(-1))
            ax.spines['right'].set_color(get_colourise(-1))
            ax.tick_params(axis='both', colors=get_colourise(-1))
    
        # Bump up the size of the ticks' numbers on the axes.
        ax.tick_params(axis='both', labelsize=23)
    
        # Fancy colours?
        if (not colourise):
            plt.xlabel("Resistance [Ω]", fontsize=33)
            plt.ylabel("Qubit plasma frequency [Hz]", fontsize=33)
            plt.title(f"Qubit frequency vs. resistance", fontsize=38)
        else:
            plt.xlabel("Resistance [Ω]", color=get_colourise(-1), fontsize=33)
            plt.ylabel("Qubit plasma frequency [Hz]", color=get_colourise(-1), fontsize=33)
            plt.title(f"Qubit frequency vs. resistance", color=get_colourise(-1), fontsize=38)
        
        # Show shits.
        plt.legend(fontsize=26)
        plt.show()
    
    # Print shits.
    if verbose:
        print("E_C: "+str(optimal_vals[0])+" ±"+str(fit_err[0])+" Hz")
        print("Delta: "+str(optimal_vals[1])+" ±"+str(fit_err[1])+" eV")
        ##print("Diff. R vs. R_N: "+str(optimal_vals[2])+" ±"+str(fit_err[2])+" %")
        ##print("T: "+str(optimal_vals[3])+" ±"+str(fit_err[3])+" K")
        
    # Calculate frequency differences.
    diff_list = []
    for fif in range(len(measured_qubit_frequencies)):
        current_measured_frequency  = measured_qubit_frequencies[fif]
        current_measured_resistance = measured_junction_resistances[fif]
        
        current_predicted_frequency = calculate_f01_from_RT_resistance(
            room_temperature_resistance = current_measured_resistance,
            E_C_in_Hz = 195e6,#optimal_vals[0],
            Delta_cold_eV = 172.48e-6,#optimal_vals[1],
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = verbose
        )
        
        # Positive = "Your value was above the prediction"
        # Negative = "Your value was below the prediction"
        curr_difference = current_measured_frequency - current_predicted_frequency
        
        # Append to list of stats.
        diff_list.append(curr_difference)
    
    return optimal_vals, fit_err, diff_list

def run_fits_of_ambegaokar_baratoff_josephson_koch(
    measured_junction_resistances,
    measured_qubit_frequencies,
    E_C_in_Hz_guess_list,
    Delta_cold_eV_guess_list,
    ##difference_between_RT_and_cold_resistance_guess_list,
    acceptable_limits = [(100e6, 500e6), (105e-6, 285e-6)]##, (0.95, 1.25)]
    ##T = 0.010,
    ):
    ''' Try different values until the fit works out.
    '''
    
    # Set flags.
    success = False
    attempts = 0
    try:
        total_attempts_to_do = len(E_C_in_Hz_guess_list) * len(Delta_cold_eV_guess_list) * len(difference_between_RT_and_cold_resistance_guess_list)
    except NameError:
        total_attempts_to_do = len(E_C_in_Hz_guess_list) * len(Delta_cold_eV_guess_list)
    
    for E_C_in_Hz_current in E_C_in_Hz_guess_list:
        for Delta_cold_eV_current in Delta_cold_eV_guess_list:
            ##for difference_between_RT_and_cold_resistance_current in difference_between_RT_and_cold_resistance_guess_list:
            try:
                optimal_vals, fit_err = fit_ambegaokar_baratoff_josephson_koch_to_resistance(
                    measured_junction_resistances = measured_junction_resistances,
                    measured_qubit_frequencies = measured_qubit_frequencies,
                    E_C_in_Hz = E_C_in_Hz_current,
                    Delta_cold_eV = Delta_cold_eV_current,
                    ##difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance_current,
                    ##T = T,
                    colourise = False,
                    verbose = False
                )
                
                # Check limits:
                if (optimal_vals[0] >= acceptable_limits[0][0]) and (optimal_vals[0] <= acceptable_limits[0][1]):
                    # E_C was fine!
                    if (optimal_vals[1] >= acceptable_limits[1][0]) and (optimal_vals[1] <= acceptable_limits[1][1]):
                        # Delta was fine!
                        ##if (optimal_vals[2] >= acceptable_limits[2][0]) and (optimal_vals[2] <= acceptable_limits[2][1]):
                        ## # Resistance mapping was fine!
                        
                        # Jolly! Report!
                        print("--------------------------------------------------\nLegal values found!")
                        print("E_C: "+str(optimal_vals[0])+" Hz ±"+str(fit_err[0]))
                        print("Delta: "+str(optimal_vals[1])+" eV ±"+str(fit_err[1]))
                        ##print("R_RT to R_N: "+str(optimal_vals[2])+" ±"+str(fit_err[2]))
                        
                        success = True
            except RuntimeError:
                pass
            
            attempts += 1
            if ((attempts/total_attempts_to_do) % 0.025) == 0:
                print("Attempts made: "+str(attempts)+", "+str(attempts/total_attempts_to_do)+"% done.")
    
    if (not success):
        print("Failed to find working set of parameters.")
    
def plot_fourier_transform_of_resistance_relaxation(
    filepath,
    normalise_resistances = 0,
    normalise_time_to_relaxation_effect = False,
    attempt_to_color_plots_from_file_name = False,
    plot_no_junction_resistance_under_ohm = 0
    ):
    ''' Take the resistance-over-time data, and plot the FFT.
    '''
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Create figure for plotting.
    plt.figure(figsize=(10, 5))
    
    # Create list that will keep track of where the "time = 0" points are
    # in the files.
    zero_points = np.zeros_like(filepath)
    zz = 0
    
    # Go through the files and add them to the plot.
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
    
        # Initialise values.
        zero_points[zz] = 0
        times = []
        resistances = []
        first_time_value_has_been_checked = False
        zero_point_was_found = False
        time_offset_due_to_appended_data = 0
        si_unit_prefix_scaler = 1.0
        resistance_at_relaxation_start = 0
        add_this_time_offset_too = 0
        
        with open(os.path.abspath(filepath_item), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)  # Convert to list for indexing options
            
            # Go through the file.
            for i in range(len(rows)):
                if i % 6 == 3:
                    
                    # Every sixth row +3 contains a resistance value
                    current_resistance = float(rows[i][1])
                    
                    # Get the SI prefix for this data.
                    ## TODO append more options, like MOhm.
                    if '[kOhm]' in str(rows[i][0]):
                        si_unit_prefix_scaler = 1000
                    else:
                        si_unit_prefix_scaler = 1
                    
                    # Scale to Ohm
                    current_resistance *= si_unit_prefix_scaler
                    
                    # Plot junction? (i.e., plot broken junctions?)
                    if current_resistance > plot_no_junction_resistance_under_ohm:
                        # Super, append junction and continue.
                        resistances.append(current_resistance)
                        
                        # Every sixth row +4 contains a time value
                        time_value = float(rows[i+1][1])
                        
                        # Check whether a new measurement appended data onto the old one.
                        if time_value == 0:
                            if not first_time_value_has_been_checked:
                                # The very first entry is time = 0, do not append.
                                first_time_value_has_been_checked = True
                            else:
                                if time_offset_due_to_appended_data > 0:
                                    print("WARNING: file \""+str(filepath_item)+"\" contains multiple \"time = 0\" points.")
                                    add_this_time_offset_too = times[-1]
                                else:
                                    # Set zero point, i.e., start of relaxation.
                                    zero_points[zz] = time_offset_due_to_appended_data
                                    zero_point_was_found = True
                                    
                                    # Log the resistance at the relaxation's start
                                    resistance_at_relaxation_start = current_resistance
                                    
                                # Log time for the appended data.
                                ## I.e., time_value == 0, and it's not the first
                                ## data value in the time series.
                                time_offset_due_to_appended_data = times[-1]
                                
                        # Append time value.
                        times.append( time_value + time_offset_due_to_appended_data + add_this_time_offset_too )
        
        # Ensure lists are the same length
        min_length = min(len(times), len(resistances))
        times = times[:min_length]
        resistances = resistances[:min_length]
        
        # Normalise resistance axis?
        resistances = np.array(resistances, dtype=np.float64)
        if normalise_resistances == 2:
            # Check whether the start of the relaxation was found.
            if resistance_at_relaxation_start > 0:
                resistances = (resistances / resistance_at_relaxation_start) - 1
                plt.ylabel("Resistance normalised to relaxation effect start [-]")
            else:
                # In that case, just take the last value and normalise to that.
                resistances = (resistances / resistances[-1]) - 1
        elif normalise_resistances == 1:
            resistances = (resistances / resistances[0]) - 1
            plt.ylabel("Resistance normalised to starting value [-]")
        else:
            plt.ylabel("Resistance [Ω]")
        
        # Normalise time axis?
        if normalise_time_to_relaxation_effect:
            # Then scale the time axis accordingly.
            times = np.array(times, dtype=np.float64)
            times = times - time_offset_due_to_appended_data
            if not zero_point_was_found:
                # In that case, the measurement was likely aborted.
                # Subtract the highest time value.
                times -= times[-1]
        
        # Iterate zero_points index counter.
        zz += 1
        
        # Add item to plot!
        ## Get the file label name.
        file_label = str(os.path.splitext(os.path.basename(filepath_item))[0])
        
        ## Attempt to find the chip identity and the junction position.
        ## Extract channel number (ChXX)
        ch_match = re.search(r'Ch(\d+)_', filepath_item)
        chip_number = ch_match.group(1) if ch_match else None

        ## Extract TR/BL prefix (TRX or BLX)
        tr_bl_match = re.search(r'_(tr|bl)\d+', filepath_item.lower())
        tr_bl = tr_bl_match.group(1) if tr_bl_match else None
        
        # Determine color for trace?
        if ((chip_number is not None) and (tr_bl is not None)) and (attempt_to_color_plots_from_file_name):
            raise NotImplementedError("Halted. Not implemented.")
            '''hex_color_string = '#'
            
            # Red
            ##hex_color_string += f"{int(((int(chip_number) - 1) / 26) * 255):02X}"
            if int(chip_number)-1 <= 13:
                hex_color_string += f"{randint(0,30):02X}"
            else:
                hex_color_string += f"{randint(204,225):02X}"
            
            # Green
            if tr_bl == "tr":
                hex_color_string += f"{randint(0,30):02X}"
            elif tr_bl == "bl":
                hex_color_string += f"{randint(204,225):02X}"
            else:
                raise ValueError("ERROR: Could not determine TL/BR for this measurement data file, even though the file name seemed to still match expectations.")
            
            # Blue
            hex_color_string += f"{randint(10,225):02X}"
            
            # Plot!
            plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=hex_color_string)'''
        else:
            # Just plot from a map.
            num_items_to_colour = len(filepath)
            colors = plt.cm.get_cmap('tab20', num_items_to_colour)
            
            ## Perform Fourier transform things.
            # Define zero-padding factor
            zero_padding_factor = 64  # Increase this for even finer resolution

            # Compute the next power of two for zero-padding
            n_fft = len(resistances) * zero_padding_factor  

            # Compute FFT with zero-padding
            fft_values = np.fft.fft(resistances, n=n_fft)  
            freqs = np.fft.fftfreq(n_fft, d=(times[1] - times[0]))  # Proper frequency scaling

            # Plot the magnitude spectrum
            plt.plot(freqs[:n_fft // 2], np.abs(fft_values[:n_fft // 2]))  # Keep positive frequencies

            
            # Plot the magnitude spectrum
            ###plt.figure(figsize=(8, 4))
            ###plt.plot(freqs, np.abs(fft_values))
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.title("FFT: resistances over time")
            plt.grid()
            plt.show()
            
            ##plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colors[jj])## TODO!! colors(jj))
        
    ##plt.xlabel("Duration [s]")
    ##plt.title("Resistance vs. Time")
    ##plt.grid()
    ##plt.legend()
    ##plt.show()
    
def plot_josephson_junction_resistance_manipulation_and_relaxation(
    filepath,
    normalise_resistances = 2,
    normalise_time = 0,
    attempt_to_color_plots_from_file_name = False,
    plot_no_junction_resistance_under_ohm = 0,
    colourise = False,
    savepath = '',
    ):
    ''' Plot the data from the resistance manipulation and ensueing
        resistance relaxation.
        
        normalise_resistances will:
            0:  do not normalise resistances,
            1:  set the first measured resistance value as the initial
                resistance of the device; all subsequent values
                will be reported as a percentage of this initial value.
            2:  same as 1, but the resistance relaxation's datapoint_0
                will be the resistance that is normalised to.
        
        normalise_time will:
            0: plot the UNIX timestamp on the x axis.
            1: normalise to the relaxation effect.
            2: normalise to the very beginning of the measurement.
    '''
    
    # Colourise counter, keeping track of the colour formatting.
    colourised_counter = 0
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Create figure for plotting.
    if colourise:
        fig, ax = plt.subplots(figsize=(12.8, 13), facecolor=get_colourise(-2))
        #plt.figure(figsize=(10, 5), facecolor=get_colourise(-2))
    else:
        fig, ax = plt.subplots(figsize=(12.8, 9.75))
        ##fig, ax = plt.subplots(figsize=(12.8, 13))
    
    # Create list that will keep track of where the "time = 0" points are
    # in the files.
    zero_points = np.zeros_like(filepath)
    zz = 0
    
    # Go through the files and add them to the plot.
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
        
        # Initialise values.
        zero_points[zz] = 0
        times = []
        resistances = []
        first_time_value_has_been_checked = False
        zero_point_was_found = False
        time_offset_due_to_appended_data = 0
        first_time_value_detected = -1.0
        si_unit_prefix_scaler = 1.0
        resistance_at_relaxation_start = 0
        add_this_time_offset_too = 0
        obvious_short = 100 # [Ω]  --  Define a resistance that defines "a short."
        lowest_non_short_resistance_in_set = 1000000000
        
        with open(os.path.abspath(filepath_item), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)  # Convert to list for indexing options
            
            # Go through the file.
            for i in range(len(rows)):
                if i % 6 == 3:
                    
                    # Every sixth row +3 contains a resistance value
                    current_resistance = float(rows[i][1])
                    
                    # Get the SI prefix for this data.
                    ## TODO append more options, like MOhm.
                    if '[kOhm]' in str(rows[i][0]):
                        si_unit_prefix_scaler = 1000
                    else:
                        si_unit_prefix_scaler = 1
                    
                    # Scale to Ohm
                    current_resistance *= si_unit_prefix_scaler
                    
                    # Update the lowest resistance found!
                    if (current_resistance < lowest_non_short_resistance_in_set) and (current_resistance > obvious_short):
                        lowest_non_short_resistance_in_set = current_resistance
                    
                    # Plot junction? (i.e., plot broken junctions?)
                    if current_resistance > plot_no_junction_resistance_under_ohm:
                        # Super, append junction and continue.
                        resistances.append(current_resistance)
                        
                        # Every sixth row +4 contains a time value
                        time_value = float(rows[i+1][1])
                        
                        ## Was this a UNIX timestamp that we should offset for?
                        if first_time_value_detected == -1.0:
                            first_time_value_detected = time_value
                        
                        # Check whether a new measurement appended data onto the old one.
                        if time_value == 0:
                            if not first_time_value_has_been_checked:
                                # The very first entry is time = 0, do not append.
                                first_time_value_has_been_checked = True
                            else:
                                if time_offset_due_to_appended_data > 0:
                                    print("WARNING: file \""+str(filepath_item)+"\" contains multiple \"time = 0\" points.")
                                    add_this_time_offset_too = times[-1]
                                else:
                                    # Set zero point, i.e., start of relaxation.
                                    zero_points[zz] = time_offset_due_to_appended_data
                                    zero_point_was_found = True
                                    
                                    # Log the resistance at the relaxation's start
                                    resistance_at_relaxation_start = current_resistance
                                    
                                # Log time for the appended data.
                                ## I.e., time_value == 0, and it's not the first
                                ## data value in the time series.
                                time_offset_due_to_appended_data = times[-1]
                                
                        # Append time value.
                        times.append( time_value + time_offset_due_to_appended_data + add_this_time_offset_too )
        
        # Ensure lists are the same length
        min_length = min(len(times), len(resistances))
        times = times[:min_length]
        resistances = resistances[:min_length]
        
        # Normalise resistance axis?
        resistances = np.array(resistances, dtype=np.float64)
        
        if normalise_resistances == 3:
            # In this case, use both y axes of normalise_resistances 1 and 2.
            resistances_ohm = resistances.copy()
            resistances_pct = ((resistances / resistances[0]) - 1) * 100
        else:
            if normalise_resistances == 2:
                # Check whether the start of the relaxation was found.
                if resistance_at_relaxation_start > 0:
                    resistances = (resistances / resistance_at_relaxation_start) - 1
                    y_label_text = "Resistance normalised to relaxation effect start [-]"
                else:
                    # In that case, just take the last value and normalise to that.
                    resistances = (resistances / resistances[-1]) - 1
            elif normalise_resistances == 1:
                resistances = ((resistances / resistances[0]) - 1) * 100
                y_label_text = "Resistance increase [%]"
            else:
                y_label_text = "Resistance [Ω]"
        
        # Normalise time axis to relaxation effect?
        if (normalise_time == 1):
            # Then scale the time axis accordingly.
            times = np.array(times, dtype=np.float64)
            times = times - time_offset_due_to_appended_data
            if not zero_point_was_found:
                # In that case, the measurement was likely aborted.
                # Subtract the highest time value.
                times -= times[-1]
        elif (normalise_time == 2):
            # Then cut away the UNIX timestamp taken at datapoint 0.
            times = np.array(times, dtype=np.float64)
            times -= first_time_value_detected
        
        # Iterate zero_points index counter.
        zz += 1
        
        # Add item to plot!
        ## Get the file label name.
        file_label = str(os.path.splitext(os.path.basename(filepath_item))[0])
        
        ## Attempt to find the chip identity and the junction position.
        ## Extract channel number (ChXX)
        ch_match = re.search(r'Ch(\d+)_', filepath_item)
        chip_number = ch_match.group(1) if ch_match else None

        ## Extract TR/BL prefix (TRX or BLX)
        tr_bl_match = re.search(r'_(tr|bl)\d+', filepath_item.lower())
        tr_bl = tr_bl_match.group(1) if tr_bl_match else None
        
        # Determine color for trace?
        if (not colourise):
            if ((chip_number is not None) and (tr_bl is not None)) and (attempt_to_color_plots_from_file_name):
                hex_color_string = '#'
                
                # Red
                ##hex_color_string += f"{int(((int(chip_number) - 1) / 26) * 255):02X}"
                if int(chip_number)-1 <= 13:
                    hex_color_string += f"{randint(0,30):02X}"
                else:
                    hex_color_string += f"{randint(204,225):02X}"
                
                # Green
                if tr_bl == "tr":
                    hex_color_string += f"{randint(0,30):02X}"
                elif tr_bl == "bl":
                    hex_color_string += f"{randint(204,225):02X}"
                else:
                    raise ValueError("ERROR: Could not determine TL/BR for this measurement data file, even though the file name seemed to still match expectations.")
                
                # Blue
                hex_color_string += f"{randint(10,225):02X}"
                
                # Plot!
                plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=hex_color_string)
            else:
                # Just plot from a map.
                num_items_to_colour = len(filepath)
                colors = plt.cm.get_cmap('tab20', num_items_to_colour)
                if savepath == '':
                    ## TODO remove this try-catch bs.
                    try:
                        plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colors[jj])## TODO!! colors(jj))
                    except TypeError:
                        plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colors(jj)) ## TODO!!
                else:
                    #dot_color = "#C4EE1C" ## TODO!
                    dot_color = "#EE1C1C" ## TODO!
                    if normalise_resistances == 3:
                        ##ax.plot(times, resistances_ohm, marker='o', linestyle='-', label=file_label + " [Ω]", color=dot_color)
                        print("TODO: dividing the time axis by 3600 to put it in hours instead of seconds.")
                        #ax.plot(times/3600, resistances_ohm, marker='o', linestyle='-', label="200x200 nm sparse-oxide", color=dot_color)
                        #ax.plot(times/3600, resistances_ohm, marker='o', linestyle='-', label="350x350 nm dense-oxide", color=dot_color)
                        ax.plot(times/3600, resistances_ohm, marker='o', linestyle='-', label="318x318 nm compact-oxide", color=dot_color)
                        
                        if 'ax2' not in locals():
                            ax2 = ax.twinx()
                            ax2.set_ylabel("Resistance increase [%]", fontsize=33)
                            
                            # Set ylim limits!
                            ###bottom_percent = -5
                            ###top_percent = 280
                            bottom_percent = -1
                            top_percent = 6
                            
                            # Begin with the percent axis. Simple, just set the numbers.
                            ax2.set_ylim(bottom_percent,top_percent)
                            
                            ## Ok, but for resistances, then this gymnastics
                            ## right here is a bit tricky. Say that
                            ## bottom_percent = -5, meaning, -5% is the bottom.
                            ## And, say that top_percent = 280, i.e.,
                            ## that the resist. axis top corresponds to +280%.
                            ## The solution is to count backwards.
                            ## (res/res[0] - 1)*100 = -5  # [%]
                            ## ⇒ res = (((-5)/100)+1)*res[0]  # [%]
                            ## ... where you replace -5% with bottom_percent
                            ##     or top_percent, as needed. res then goes
                            ##     into .ylim for the resistance axis in Ω.
                            
                            # Set the ordinary resistance axis!
                            ax.set_ylim(((bottom_percent/100)+1)*resistances_ohm[0], ((top_percent/100)+1)*resistances_ohm[0])
                        
                        print("TODO: the percent-x axis has been divided into hours, instead of seconds.")
                        ax2.plot(times/3600, resistances_pct, marker='o', linestyle='--', color=dot_color)
                        
                    else:
                        plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=dot_color)
        else:
            # Use patterned colour.
            plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=get_colourise((colourised_counter // 4) + ((colourised_counter % 4) + 1) / 10))
            colourised_counter += 1
        
    
    # Set axes' colour? Title colour? And so on.
    if colourise:
        fig.patch.set_alpha(0)
        ax.grid(color=get_colourise(-1))
        ax.set_facecolor(get_colourise(-2))
        ax.spines['bottom'].set_color(get_colourise(-1))
        ax.spines['top'].set_color(get_colourise(-1))
        ax.spines['left'].set_color(get_colourise(-1))
        ax.spines['right'].set_color(get_colourise(-1))
        #ax.set_xlabel('X Label', color=get_colourise(-1))
        #ax.set_ylabel('Y Label', color=get_colourise(-1))
        ax.tick_params(axis='both', colors=get_colourise(-1))
        plt.grid()
    
    
    if normalise_resistances == 3:
        ax.set_ylabel("Resistance [Ω]", fontsize=33)
        ax.grid()
        if colourise:
            ax2.tick_params(axis='y', colors=get_colourise(-1), labelsize=23)
            ax2.spines['right'].set_color(get_colourise(-1))
            ax2.yaxis.label.set_color(get_colourise(-1))
        ax.tick_params(axis='y', labelsize=23)
        if 'ax2' in locals():
            ax2.tick_params(axis='y', labelsize=23)
    else:
        ax.set_ylabel(y_label_text, fontsize=33)
        ax.grid()
    
    # Set x-axis text.
    if normalise_time == 0:
        ax.set_xlabel("Time [s since epoch]", fontsize=33)
    elif normalise_time == 1:
        ax.set_xlabel("Time since resistance relaxation onset [s]", fontsize=33)
    elif normalise_time == 2:
        print("TODO: the x axis may have been divided into hours, instead of seconds. Check the unit in the plot if unsure.")
        ax.set_xlabel("Duration [h]", fontsize=33)
    else:
        ax.set_xlabel("Time [s]", fontsize=33)
    
    # Bump up the size of the ticks' numbers on the axes.
    ax.tick_params(axis='both', labelsize=23)
    
    plt.xlabel("Duration [s]", color=get_colourise(-1), fontsize=33)
    ##plt.ylabel(y_label_text, color=get_colourise(-1), fontsize=33)
    if savepath == '':
        plt.title("Resistance vs. Time", color=get_colourise(-1), fontsize=38)
    
    # Legend and layout.
    if normalise_resistances == 3:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=24)
    else:
        ax.legend()
    
    # Tight layout.
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
    
        # Fix path name?
        thin = ('JJAnneal01C' in filepath[0]) and ('stepped' in filepath[0])
        thick = ('MPW005A' in filepath[0]) and ('stepped' in filepath[0])
        base, _ = os.path.splitext(savepath)
        if thin:
            new_savepath = f"{base}_soft_stepped.png"
        elif thick:
            new_savepath = f"{base}_hard_stepped.png"
        else:
            new_savepath = f"{base}.png"
        plt.savefig(new_savepath, dpi=164, bbox_inches='tight')
    
    # Show shits.
    plt.show()

def simulate_frequency_accuracy_of_model_from_RT_resistance(
    no_junctions,
    resistance,
    resistance_measurement_error_std_deviation,
    E_C_mean_in_Hz,
    E_C_error_std_deviation_in_Hz,
    Delta_mean_eV,
    Delta_error_std_deviation_eV,
    temperature_mean,
    temperature_std_deviation,
    difference_between_RT_and_cold_resistance_mean,
    difference_between_RT_and_cold_resistance_std_dev,
    plot = True
    ):
    ''' Virtually manufactures no_junctions worth of junctions,
        and produces a distribution, showing what the frequency
        accuracy actually is.
        
        resistance_measurement_error_mean is the error bar
        of the resistance measurement itself.
    '''
    
    # Given a chip's RT resistance, what is the accuracy of the model?
    ## First, we have a resistance measurement error.
    ## Create some distribution around a mean, which is our measured value.
    resistance_with_meas_error = np.random.normal(resistance, resistance_measurement_error_std_deviation, no_junctions)
    
    # Given some known E_C, and some known deviation onto it,
    # randomly sample some E_C values.
    E_C_with_error = np.random.normal(E_C_mean_in_Hz, E_C_error_std_deviation_in_Hz, no_junctions)
    
    # Given some known superconducting energy gap, and its error, get values.
    Delta_with_error_eV = np.random.normal(Delta_mean_eV, Delta_error_std_deviation_eV, no_junctions)
    error_Delta_in_text = str(f"{(Delta_error_std_deviation_eV / Delta_mean_eV)*100:.3f}")
    
    # Given some known temperature, and its error, get values.
    temperature_with_error = np.random.normal(temperature_mean, temperature_std_deviation, no_junctions)
    ##error_temperature_in_text = str(f"{(temperature_std_deviation / temperature_mean)*100:.3f}")
    error_temperature_in_text = str(f"{(temperature_std_deviation*1000):.1f}")
    
    # Given some known difference between room temperature resistance
    # and the normal state resistance at mK, get values.
    diff_rt_R_with_error = np.random.normal(difference_between_RT_and_cold_resistance_mean, difference_between_RT_and_cold_resistance_std_dev, no_junctions)
    error_diff_R_in_text = str(f"{(difference_between_RT_and_cold_resistance_std_dev / difference_between_RT_and_cold_resistance_mean)*100:.3f}")
    
    # What frequency does those resistances correspond to?
    frequencies_calculated = []
    for jj in range(no_junctions):
        frequencies_calculated.append(
            calculate_f01_from_RT_resistance(
                room_temperature_resistance = resistance_with_meas_error[jj],
                E_C_in_Hz = E_C_with_error[jj],
                Delta_cold_eV = Delta_with_error_eV[jj],
                difference_between_RT_and_cold_resistance = diff_rt_R_with_error[jj],
                T = temperature_with_error[jj],
                verbose = False
            )
        )
    
    # Get the standard deviation of the calculated frequencies.
    frequencies_calculated_standard_deviation = np.std(frequencies_calculated, ddof=0)
    ## Here, I am not using the sample standard deviation.
    ## Isn't it so that I know the full population? TODO
    
    # Plot the expected normal distribution curve!
    ''' This code snippet was implemented from Christian Križan's
        research work in https://arxiv.org/abs/2412.15022 '''
    
    ## Here, Sturge's formula along with Doane's correction factor is used
    ## for getting a decent number of bins.
    no_entries = len(frequencies_calculated)
    third_moment_skewness_of_distribution = moment(frequencies_calculated, moment = 3) # Get the assymetry of the distribution
    sigma_g1 = np.sqrt( (6*(no_entries - 2))/((no_entries + 1)*(no_entries + 3)) )
    doane_correction_factor_Ke = np.log2(1 + np.abs(third_moment_skewness_of_distribution)/sigma_g1)
    bins_calculated = int(np.ceil(1 + np.log2( no_entries ) + doane_correction_factor_Ke))
    
    # Plot histogram of the calculated frequency values
    if plot:
        plt.figure(figsize=(8,5))
        plt.hist(frequencies_calculated, bins=bins_calculated, density=True, alpha=0.6, color='b', edgecolor='black', rwidth = 0.9)
        num_sigmas_in_expected_pdf = 5
        
        # Create trace for the expected probability distribution
        x = np.linspace(
            np.mean(frequencies_calculated) - float(num_sigmas_in_expected_pdf*frequencies_calculated_standard_deviation),
            np.mean(frequencies_calculated) + float(num_sigmas_in_expected_pdf*frequencies_calculated_standard_deviation),
            100
        )
        pdf = (1 / (frequencies_calculated_standard_deviation * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((x - np.mean(frequencies_calculated)) / frequencies_calculated_standard_deviation) ** 2)
        plt.plot(x, pdf, 'r-', label="Expected normal distribution")
        
        # Labels and title
        plt.xlabel("Calculated frequencies [Hz]")
        plt.ylabel("Probability density")
        plt.title("Distribution about frequency target:\n±"+str(resistance_measurement_error_std_deviation)+" Ω measurement error, ±"+str(E_C_error_std_deviation_in_Hz/1e6)+" MHz E_C,\n±"+str(error_Delta_in_text)+"% Δ, ±"+str(error_temperature_in_text)+" mK T, ±"+str(error_diff_R_in_text)+"% R vs R_T")
        plt.legend()
        plt.show()
    
    # Print some things.
    print("Mean frequency is: "+str(np.mean(frequencies_calculated))+" [Hz]")
    print("Standard deviation for the frequency is: "+str(frequencies_calculated_standard_deviation)+" [Hz]")
    
    # Get values for the return, and return.
    final_mean = np.mean(frequencies_calculated)
    final_std = frequencies_calculated_standard_deviation
    return (final_mean, final_std)

def plot_trend_for_changing_superconducting_gap(
    list_of_doubles_of_Delta_eV_and_Delta_std_eV,
    no_junctions,
    resistance,
    resistance_measurement_error_std_deviation,
    E_C_mean_in_Hz,
    E_C_error_std_deviation_in_Hz,
    Delta_mean_eV,
    Delta_error_std_deviation_eV,
    temperature_mean,
    temperature_std_deviation,
    difference_between_RT_and_cold_resistance_mean,
    difference_between_RT_and_cold_resistance_std_dev,
    ):
    ''' Given a list of doubles, containing (Delta_mean, Delta_std),
        calculate the resulting frequencies and their standard deviation.
        Finally, plot.
    '''
    
    # Get calculated frequencies.
    output_values = []
    inserted_std_values = []
    for ii in range(len(list_of_doubles_of_Delta_eV_and_Delta_std_eV)):
        inserted_std_values.append( list_of_doubles_of_Delta_eV_and_Delta_std_eV[ii][1] )
        output_values.append(
            simulate_frequency_accuracy_of_model_from_RT_resistance(
                no_junctions = no_junctions,
                resistance = resistance,
                resistance_measurement_error_std_deviation = resistance_measurement_error_std_deviation,
                E_C_mean_in_Hz = E_C_mean_in_Hz,
                E_C_error_std_deviation_in_Hz = E_C_error_std_deviation_in_Hz,
                Delta_mean_eV = list_of_doublers_of_Delta_eV_and_Delta_std_eV[ii][0],
                Delta_error_std_deviation_eV = inserted_std_values[ii],
                temperature_mean = temperature_mean,
                temperature_std_deviation = temperature_std_deviation,
                difference_between_RT_and_cold_resistance_mean = difference_between_RT_and_cold_resistance_mean,
                difference_between_RT_and_cold_resistance_std_dev = difference_between_RT_and_cold_resistance_std_dev,
                plot = False
            )
        )
    
    # Unpack data.
    means, error_bars = zip(*output_values)
    
    # Plot!
    plt.figure(figsize=(8, 5))
    #plt.errorbar(range(len(means)), means, yerr=error_bars, fmt='o', linestyle='-', color='orange', label='Simulated frequency')
    plt.plot(inserted_std_values, error_bars, marker='o', linestyle='-', color='orange')
    plt.xlabel("std_dev of Δ [eV]")
    plt.ylabel("Simulated frequency std_dev [Hz]")
    plt.title("Improving accuracy of Δ")
    #plt.legend()
    plt.grid(True)
    plt.show()

def plot_manipulation_plan(
    expected_resistance_relaxation = 1.0229,
    E_C_in_Hz = 195e6,
    Delta_cold_eV = 174.28e-6,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    verbose = False
    ):
    ''' Illustrate how to manipulate qubits on a chip.
    '''
    
    # TODO user-settable    
    original_resistances = [
        5.749e3, 6.045e3, 6.334e3, 6.653e3,
        7.411e3, 7.479e3, 7.541e3, 7.979e3
    ]
    original_frequencies = []
    for jj in range(len(original_resistances)):
        resistance_item = original_resistances[jj]
        original_frequencies.append(
            calculate_f01_from_RT_resistance(
                room_temperature_resistance = resistance_item,
                E_C_in_Hz = E_C_in_Hz,
                Delta_cold_eV = Delta_cold_eV,
                difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
                T = T,
                verbose = False
            )
        )
    
    # Target frequencies.
    '''## Find the top and bottom values first.
    top_freq = calculate_f01_from_RT_resistance(
        room_temperature_resistance = original_resistances[0] * 1.143, # Statistically survivable
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = False
    )
    bot_freq = calculate_f01_from_RT_resistance(
        room_temperature_resistance = original_resistances[-1] * 1.0143, # Statistically survivable
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = False
    )'''
    target_frequencies = np.linspace(original_frequencies[0], original_frequencies[-1], len(original_frequencies))
    
    res_of_m = original_resistances[jj]
    min_res_increased_for_m = res_of_m * 1.0238 # 75% survival for n=8
    new_m = calculate_f01_from_RT_resistance(
        room_temperature_resistance = min_res_increased_for_m,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = True
    )
    target_frequencies[4] = new_m
    slope_of_old_line = (original_frequencies[-1] - original_frequencies[0]) / len(original_frequencies)
    
    target_frequencies[5] = slope_of_old_line * 1 + new_m
    target_frequencies[6] = slope_of_old_line * 2 + new_m
    target_frequencies[7] = slope_of_old_line * 3 + new_m
    
    target_frequencies[3] = slope_of_old_line * -1 + new_m
    target_frequencies[2] = slope_of_old_line * -2 + new_m
    target_frequencies[1] = slope_of_old_line * -3 + new_m
    target_frequencies[0] = slope_of_old_line * -4 + new_m
    
    calcualted_res_to_manip = []
    for jj in range(len(target_frequencies)):
        frequency_item = target_frequencies[jj]
        calcualted_res_to_manip.append(
            calculate_resistance_to_manipulate_to(
                target_f_01 = frequency_item,
                E_C_in_Hz = E_C_in_Hz,
                Delta_cold_eV = Delta_cold_eV,
                original_resistance_of_junction = original_resistances[jj],
                expected_aging = 0,
                expected_resistance_relaxation = expected_resistance_relaxation,
                verbose = True
            )
        )
    
    ##data_points = [
    ##    5.43108914, 5.31263521, 5.19418128, 5.07572735,
    ##    4.95727342, 4.83881949, 4.72036556, 4.60191163
    ##]
    
    # Plot frequencies
    plt.figure(figsize=(8, 5))
    qubit_axis = np.linspace(1, len(target_frequencies), len(target_frequencies))
    plt.plot(qubit_axis, original_frequencies, marker='s', linestyle='--', color='r', label='Original frequencies')
    plt.plot(qubit_axis, target_frequencies, marker='o', linestyle='-', color='b', label='Target frequencies')
    
    # Labels and title
    plt.xlabel('Qubit')
    plt.ylabel('Frequency [Hz]')
    plt.title('Frequency manipulation plan')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def plot_active_manipulation(
    filepath,
    normalise_resistances = 0,
    normalise_time = True,
    plot_no_junction_resistance_under_ohm = 0,
    fitter = 'none',
    skip_initial_dip = False,
    plot_fit_parameters_in_legend = False,
    colourise = False,
    title_label = None,
    enable_mask = False,
    savepath = ''
    ):
    ''' Plot soledly only the active manipulation.
        
        normalise_resistances will:
            0:  do not normalise resistances,
            1:  set the first measured resistance value as the initial
                resistance of the device; all subsequent values
                will be reported as a percentage of this initial value.
        
        normalise_time:
            True: the x-axis will be in seconds after the measurement started.
            False: the x-axis will be given in UNIX time.
        
        fitter:
            'none':         Perform no fitting.
            'second_order': Attempt fit to R(t) = R_0 + alpha·t + beta·t^2
            'third_order':  Attempt fit to R(t) = R_0 + alpha·t + beta·t^2 + delta·t^3
            'exponential':  Attempt fit to R(t) = R_0 + epsilon·e( t/t_0 · gamma)
            'power':        Attempt fit to R(t) = R_0 + A·t^B
        
        skip_initial_dip:
            If false, look for the string "START_MANIPULATION" in column 3
            of the .csv format.
    '''
    
    ## Already up here, let's define a few lists to be used for the
    ## return statement.
    list_of_traces_in_plot = []
    list_of_error_bars_of_traces_in_plot = []
    list_of_fit_parameter_labels = []
    
    # Initially, let's define some functions for the fitting.
    '''def second_order_func(t, t_0, R_0, alpha, beta):
        return R_0 + (alpha * (t-t_0)) + (beta * (t-t_0)**2)'''
    def second_order_func(t, alpha, beta):
        return (alpha * t) + (beta * t**2)
    
    '''def third_order_func(t, t_0, R_0, alpha, beta, delta):
        return R_0 + (alpha * (t-t_0)) + (beta * (t-t_0)**2) + (delta * (t-t_0)**3)'''
    def third_order_func(t, alpha, beta, delta):
        return (alpha * t) + (beta * t**2) + (delta * t**3)
    
    '''def exponential_func(t, t_0, R_0, epsilon, gamma, tau):
        ##return R_0 + epsilon * (np.e)**((t-t_0) * gamma)
        return R_0 + epsilon * (1 - (np.e)**((t-t_0)/tau * -gamma))'''
    def exponential_func(t, epsilon, gamma):
        return epsilon * (1 - (np.e)**(t * -gamma))
    
    '''def power_func(t, t_0, R_0, A, B):
        return R_0 + A * ((t-t_0)**B)'''
    def power_func(t, A, B):
        return A * (t**B)
    
    
    def active_increase_fitter(
        resistances,
        time,
        fitter
        ):
        ''' fitter:
            'second_order': Attempt fit to R(t) = R_0 + alpha·t + beta·t^2
            'third_order':  Attempt fit to R(t) = R_0 + alpha·t + beta·t^2 + delta·t^3
            'exponential':  Attempt fit to R(t) = R_0 + epsilon·e( t/t_0 · gamma)
            'power':        Attempt fit to R(t) = R_0 + A·t^B
        '''
        ## Let's guess initial guessing values.
        ## To guess the polynomial values, let's do something a bit
        ## interesting, and run np.polyfit, to fit a 2nd and 3rd order
        ## polynomial.
        ## Then, fit.
        
        t_0_guess = time[0]
        R_0_guess = resistances[0] # Should be R_0, will be overwritten if better numbers exist below.
        if fitter == 'second_order':
            coeffs_2nd = np.polyfit(time, resistances, 2) # [beta, alpha, R_0]
            R_0_guess, alpha_guess, beta_guess = coeffs_2nd[::-1]  ## Reverse order for readability
        elif fitter == 'third_order':
            coeffs_3rd = np.polyfit(time, resistances, 3) # [delta, beta, alpha, R_0]
            R_0_guess, alpha_guess, beta_guess, delta_guess = coeffs_3rd[::-1]
        elif fitter == 'exponential':
            gamma_guess   = 0.010  # Exponential rise (so, minus +gamma_guess). 1/100 s is a rough guess.
            epsilon_guess = 0.10   # Expect flattening off at ~10% increase.
            tau_guess     = 30     # Rough estimate of one experimental exp-slope I saw.
        elif fitter == 'power':
            A_guess = 0.007 # Pro-tip: check t=1 for this value, since A*(t=1)^borp = A
            B_guess = 0.9   # Must be between 0 and 1 for the expected flattening behaviour, right?
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Fit!
        if fitter == 'second_order':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = second_order_func,
                xdata = time,
                ydata = resistances,
                p0    = (alpha_guess, beta_guess)
            )
            """p0    = (t_0_guess, R_0_guess, alpha_guess, beta_guess)"""
        elif fitter == 'third_order':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = third_order_func,
                xdata = time,
                ydata = resistances,
                p0    = (alpha_guess, beta_guess, delta_guess)
            )
            """p0    = (t_0_guess, R_0_guess, alpha_guess, beta_guess, delta_guess)"""
        elif fitter == 'exponential':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = exponential_func,
                xdata = time,
                ydata = resistances,
                p0    = (epsilon_guess, gamma_guess)
            )
            """p0    = (t_0_guess, R_0_guess, epsilon_guess, gamma_guess, tau_guess)"""
        elif fitter == 'power':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = power_func,
                xdata = time,
                ydata = resistances,
                p0    = (A_guess, B_guess)
            )
            """p0    = (t_0_guess, R_0_guess, A_guess, B_guess)"""
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Extract parameters.
        """optimal_t_0    = optimal_vals[0]
        optimal_R_0    = optimal_vals[1]
        if fitter == 'second_order':
            optimal_alpha = optimal_vals[2]
            optimal_beta  = optimal_vals[3]
        elif fitter == 'third_order':
            optimal_alpha = optimal_vals[2]
            optimal_beta  = optimal_vals[3]
            optimal_delta = optimal_vals[4]
        elif fitter == 'exponential':
            optimal_epsilon = optimal_vals[2]
            optimal_gamma   = optimal_vals[3]
            optimal_tau     = optimal_vals[4]
        elif fitter == 'power':
            optimal_A = optimal_vals[2]
            optimal_B = optimal_vals[3]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))"""
        
        # Updated version without R_0 and without t_0, to try and force
        # the error bar down on the fits. because, it's larger than
        # the fitted parameter, which is a sign that the number of parameters
        # is too big.
        if fitter == 'second_order':
            optimal_alpha = optimal_vals[0]
            optimal_beta  = optimal_vals[1]
        elif fitter == 'third_order':
            optimal_alpha = optimal_vals[0]
            optimal_beta  = optimal_vals[1]
            optimal_delta = optimal_vals[2]
        elif fitter == 'exponential':
            optimal_epsilon = optimal_vals[0]
            optimal_gamma   = optimal_vals[1]
            optimal_tau     = optimal_vals[2]
        elif fitter == 'power':
            optimal_A = optimal_vals[0]
            optimal_B = optimal_vals[1]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Get the fit errors.
        fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
        """err_t_0 = fit_err[0]
        err_R_0 = fit_err[1]
        if fitter == 'second_order':
            err_alpha = fit_err[2]
            err_beta  = fit_err[3]
        elif fitter == 'third_order':
            err_alpha = fit_err[2]
            err_beta  = fit_err[3]
            err_delta = fit_err[4]
        elif fitter == 'exponential':
            err_epsilon = fit_err[2]
            err_gamma   = fit_err[3]
            err_tau     = fit_err[4]
        elif fitter == 'power':
            err_A = fit_err[2]
            err_B = fit_err[3]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))"""
        # See comment above regarding the error bars being larger than
        # the fitted parameters if R_0 and t_0 is included.
        if fitter == 'second_order':
            err_alpha = fit_err[0]
            err_beta  = fit_err[1]
        elif fitter == 'third_order':
            err_alpha = fit_err[0]
            err_beta  = fit_err[1]
            err_delta = fit_err[2]
        elif fitter == 'exponential':
            err_epsilon = fit_err[0]
            err_gamma   = fit_err[1]
            err_tau     = fit_err[2]
        elif fitter == 'power':
            err_A = fit_err[0]
            err_B = fit_err[1]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Get a fit curve!
        if fitter == 'second_order':
            fitted_curve = second_order_func(
                t     = times,
                alpha = optimal_alpha,
                beta  = optimal_beta
            )
            """t_0   = optimal_t_0,
                R_0   = optimal_R_0,"""
        elif fitter == 'third_order':
            fitted_curve = third_order_func(
                t     = times,
                alpha = optimal_alpha,
                beta  = optimal_beta,
                delta = optimal_delta
            )
            """t_0   = optimal_t_0,
                R_0   = optimal_R_0,"""
        elif fitter == 'exponential':
            fitted_curve = exponential_func(
                t       = times,
                epsilon = optimal_epsilon,
                gamma   = optimal_gamma,
                tau     = optimal_tau
            )
            """t_0     = optimal_t_0,
                R_0     = optimal_R_0,"""
        elif fitter == 'power':
            fitted_curve = exponential_func(
                t   = times,
                A   = optimal_A,
                B   = optimal_B,
            )
            """t_0 = optimal_t_0,
                R_0 = optimal_R_0,"""
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Return!
        return fitted_curve, optimal_vals, fit_err
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Create figure for plotting.
    if colourise:
        fig1, ax1 = plt.subplots(figsize=(12.8, 10), facecolor=get_colourise(-2))
        fig2, ax2 = plt.subplots(figsize=(9, 11), facecolor=get_colourise(-2))
    else:
        ##fig1, ax1 = plt.subplots(figsize=(12.8, 10))
        fig1, ax1 = plt.subplots(figsize=(12.8, 9.14))
        fig2, ax2 = plt.subplots(figsize=(9, 11))
    
    # Go through the files and add them to the plot.
    ## Also, store fit values for the return statement.
    fitted_values_to_be_returned = []
    fitted_errors_to_be_returned = []
    lowest_non_short_resistance_of_all = 1000000000
    highest_number_on_the_y_axis = 0.0
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
        
        # Set initial parameters.
        times = []
        resistances = []
        obvious_short = 160 # [Ω]  --  Define a resistance that defines "a short."
        lowest_non_short_resistance_in_set = 1000000000
        
        ## The new time format assumes that all data is set with reference
        ## to some initial time.
        time_0 = -1 # [s] -- Which would be 23:59:59 on December 31st 1969
        time_at_start = 0.0
        
        # Check whether to begin to store data from this file, i.e., whether
        # the user is requesting to skip the initial resistance dip.
        do_not_save_data_yet = skip_initial_dip
        
        # Open file.
        with open(os.path.abspath(filepath_item), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)  # Convert to list for indexing options
            
            ## In case of old data files, we need to catch whether there is
            ## no tag telling us where the manipulation sequence started.
            ## So, we actually go through the file once first, looking
            ## for tags.
            old_file = True
            for i in range(len(rows)):
                try:
                    if 'START_MANIPULATION' in str(rows[i+1][2]):
                        old_file = False
                except IndexError:
                    # In this case, we didn't find the tag either.
                    pass
            if old_file:
                # Thus, we (TODO: currently) have no reliable way of
                # determining where the initial dip ends, if this happens.
                print("Data file '"+str(filepath_item)+"' could not be used to identify where the initial dip ends. Ignoring argument.")
                do_not_save_data_yet = False
            
            # Old file or not, we take the timestamp in cell [4,1] (0 indx.)
            # in order to calculate how much initial time was removed.
            time_value_at_first_entry = float(rows[4][1])
            time_at_start = -1.0
            
            # Go through the file.
            for i in range(len(rows)):
                if i % 6 == 3:
                    
                    if do_not_save_data_yet:
                        ## This portion will create an IndexError in case
                        ## the datafile is old and does not contain
                        ## any 'START_MANIPULATION' tag.
                        ## This error case was handled above.
                        if 'START_MANIPULATION' in str(rows[i+1][2]):
                            # Then signal that we may commence.
                            do_not_save_data_yet = False
                            
                            # Report how much time passed before this tag
                            # appeared.
                            time_at_start = float(rows[i+1][1])
                            print(filepath_item)
                            print("Detected that the initial dip was "+str(time_at_start - time_value_at_first_entry)+" [s] long.")
                    
                    # The reason this if-if case is written this way,
                    # is to catch the data in the same data storage event
                    # in the file, that also contained the START_* keyword.
                    # Which, could have been the zeroth data storage event.
                    
                    if (not do_not_save_data_yet):
                        ## In that case, continue!
                        
                        # Every sixth row +3 contains a resistance value
                        current_resistance = float(rows[i][1])
                        
                        # Get the SI prefix for this data.
                        ## TODO append more options, like MOhm.
                        if '[kOhm]' in str(rows[i][0]):
                            si_unit_prefix_scaler = 1000
                        else:
                            si_unit_prefix_scaler = 1
                        
                        # Scale to Ohm
                        current_resistance *= si_unit_prefix_scaler
                        
                        # Update the lowest resistance found!
                        if (current_resistance < lowest_non_short_resistance_in_set) and (current_resistance > obvious_short):
                            lowest_non_short_resistance_in_set = current_resistance
                            if lowest_non_short_resistance_in_set < lowest_non_short_resistance_of_all:
                                lowest_non_short_resistance_of_all = lowest_non_short_resistance_in_set
                        
                        # Plot junction? (i.e., plot broken junctions?)
                        if current_resistance > plot_no_junction_resistance_under_ohm:
                            
                            # Super, append junction and continue.
                            resistances.append(current_resistance)
                            
                            # Every sixth row +4 contains a time value
                            time_value = float(rows[i+1][1])
                            ## Did we define the starting time?
                            if time_0 == -1:
                                time_0 = time_value
                            
                            # Calculate what number to be put as the time_value.
                            ## UNIX time or seconds relative to start?
                            if normalise_time:
                                time_value -= time_0
                            times.append(time_value)
                            
                            ## The new format assumes that any and all times
                            ## report the UNIX timestamp of the data itself.
                            ## This way, there is less b/s here regarding
                            ## relative offsets and hatmatilka.
        
        # Ensure lists are the same length
        min_length = min(len(times), len(resistances))
        times = times[:min_length]
        resistances = resistances[:min_length]
        
        # At this point we may just as well numpy-ify the lists.
        times = np.array(times, dtype=np.float64)
        resistances = np.array(resistances, dtype=np.float64)
        
        # Report detected R(t=0).
        print("R₀ of trace: "+str(resistances[0]))
        
        # Normalise resistance axis?
        if normalise_resistances == 1:
            resistances = ((resistances / resistances[0]) - 1) * 100
            y_label_text = "Resistance increase [%]"
        else:
            y_label_text = "Resistance [Ω]"
        
        # At this point, we can identify how much initial dip was removed,
        # vs. the maximum resistance change.
        if normalise_resistances == 1:
            print("Dip removed: "+str(time_at_start - time_value_at_first_entry)+", max change: "+str(resistances[-1]))
        
        # Add item to plot!
        ## First, let's try to fit the data too.
        fit_results = None
        if (fitter != 'none'):
            # The user has requested a fit.
            fit_results = active_increase_fitter(
                resistances = resistances, ## Note that this axis has been normalised by now, if so was requested by the user.
                time = times,
                fitter = fitter
            )
        
        ## Get the file label name.
        file_label = str(os.path.splitext(os.path.basename(filepath_item))[0])
        
        # Determine color and marker for trace?
        ## Select marker symbol. Also, this marker symbol is used later, too.
        if   (jj % 5) == 0:
            marker_symbol = 'o'
        elif (jj % 5) == 1:
            marker_symbol = 's'
        elif (jj % 5) == 2:
            marker_symbol = '^'
        elif (jj % 5) == 3:
            marker_symbol = '*'
        elif (jj % 5) == 4:
            marker_symbol = 'D'
        if (not colourise):
            
            # Just plot from a map.
            if 'thin' in file_label.lower():
                colour_label = 'custom_thin'
                colors = ['#1cee70', '#1cd89c', '#1cc1c3', '#1ca5da', '#1c88e8', '#1c70ee']
                
                num_items_to_colour = len(filepath)
                if num_items_to_colour < 6:
                    colors = colors[:num_items_to_colour]
                elif num_items_to_colour == 6:
                    colors = colors
                else:
                    raise NotImplementedError("Colouring error.")
                
                # Get the last number from the filename
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None) + ' mV'
            
            elif 'thick' in file_label.lower():
                colour_label = 'custom_thick'
                colors = ['#ee1c1c', '#f47e1c', '#f6c11c', '#e6df1c', '#d1eb1c', '#c4ee1c']
                
                num_items_to_colour = len(filepath)
                if num_items_to_colour < 6:
                    colors = colors[:num_items_to_colour]
                elif num_items_to_colour == 6:
                    colors = colors
                else:
                    raise NotImplementedError("Colouring error.")
                
                # Get the last number from the filename
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None) + ' mV'
            
            elif 'thinLEGACY' in file_label.lower():
                ## Catch legacy thin.
                colour_label = 'winter'
                num_items_to_colour = len(filepath)
                cmap = plt.cm.get_cmap(colour_label, num_items_to_colour)
                colors = [cmap(i) for i in range(num_items_to_colour)]
                
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
            
            elif 'thickLEGACY' in file_label.lower():
                ## Catch legacy thick.
                colour_label = 'autumn'
                num_items_to_colour = len(filepath)
                cmap = plt.cm.get_cmap(colour_label, num_items_to_colour)
                colors = [cmap(i) for i in range(num_items_to_colour)]
                
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
            
            else:
                # Fallback to default colormap if not 'thin' or 'thick'
                colour_label = 'tab20'
                num_items_to_colour = len(filepath)
                cmap = plt.cm.get_cmap(colour_label, num_items_to_colour)
                colors = [cmap(i) for i in range(num_items_to_colour)]
            
            # Mask the scatter plot?
            if enable_mask:
                mask = (times >= 0) & (times <= 300)
                times_to_plot = np.array(times)[mask]
                resistances_to_plot = np.array(resistances)[mask]
            else:
                times_to_plot = times.copy()
                resistances_to_plot = resistances.copy()
            
            plt.figure(1) # Set figure 1 as active.
            plt.scatter(times_to_plot, resistances_to_plot, marker=marker_symbol, label=file_label, color=colors[jj])## TODO!! colors[jj])## TODO!! colors(jj))
            
            # Update the largest value present on the y-axis?
            if np.max(resistances) > highest_number_on_the_y_axis:
                highest_number_on_the_y_axis = np.max(resistances)
            
            ##plt.figure(3) # Set figure 3 as active.
            ##plt.scatter(times_to_plot, np.log10(resistances_to_plot), marker=marker_symbol, label=file_label, color=colors[jj])## TODO!! colors(jj))
        else:
            
            # Get pseudolegacy filename labels?
            if   'thin'  in file_label.lower():
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
                
            elif 'thick' in file_label.lower():
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
            
            # Mask the scatter plot?
            if enable_mask:
                mask = (times >= 0) & (times <= 300)
                times_to_plot = np.array(times)[mask]
                resistances_to_plot = np.array(resistances)[mask]
            else:
                times_to_plot = times.copy()
                resistances_to_plot = resistances.copy()
        
            # Then follow the schema.
            plt.figure(1) # Set figure 1 as active.
            plt.scatter(times_to_plot, resistances_to_plot, marker=marker_symbol, label=file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
            
            ##plt.figure(3) # Set figure 3 as active.
            ##plt.plot(times_to_plot, np.log10(resistances_to_plot), marker=marker_symbol, label=file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
            
            # Update the largest value present on the y-axis?
            if np.max(resistances) > highest_number_on_the_y_axis:
                highest_number_on_the_y_axis = np.max(resistances)
        
        # Plot the fit curve.
        fit_label = ''
        
        ## At this point, store the fit values and errors, for the return
        ## statement that happens later.
        if fitter != 'none':
            fitted_values_to_be_returned.append(fit_results[1])
            fitted_errors_to_be_returned.append(fit_results[2])
        
            for kk in range(len(fit_results[1])):
                fitted_values = (fit_results[1])[kk]
                fitted_errors = (fit_results[2])[kk]
                prefix = '?'
                
                # See comment above regarding the error bars being larger
                # than the fitted values if R_0 and t_0 are included.
                """if   kk == 0:
                    prefix = 'R₀'
                    fit_label += prefix+': '+(f"{fitted_values:.3f} ±{fitted_errors:.3f}")+'\n'
                elif kk == 1:
                    prefix = 't₀'
                    fit_label += prefix+': '+(f"{fitted_values:.3f} ±{fitted_errors:.3f}")+'\n'
                else:"""
                if fitter == 'second_order':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'α'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'β'
                elif fitter == 'third_order':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'α'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'β'
                        """elif kk == 4:"""
                    elif kk == 2:
                        prefix = 'δ'
                elif fitter == 'exponential':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'ε'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'γ'
                        """elif kk == 4:"""
                    elif kk == 2:
                        prefix = 'τ'
                elif fitter == 'power':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'A'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'B'
                else:
                    raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
                
                # Find a proper exponent of the number.
                exponent       = np.floor(np.log10(np.abs( fitted_values )))
                error_exponent = np.floor(np.log10(np.abs( fitted_errors )))
                fit_label += prefix+': '+(f"{(fitted_values * (10**(-exponent))):.3f}·10^{exponent} ±{(fitted_errors * (10**(-error_exponent))):.3f}·10^{error_exponent}")+'\n'
            
            # Mask the fit pot?
            if enable_mask:
                mask = (times >= 0) & (times <= 300)
                times_to_plot = np.array(times)[mask]
                fit_to_plot   = np.array(fit_results[0])[mask]
            else:
                times_to_plot = times
                fit_to_plot   = fit_results[0]
            
            if (not colourise):
                plt.figure(1) # Set figure 1 as active.
                if plot_fit_parameters_in_legend:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', label='Fit '+str(jj)+': '+fit_label, color=colors[jj])## TODO!! colors(jj))
                else:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', color=colors[jj])## TODO!! colors(jj))
            else:
                plt.figure(1) # Set figure 1 as active.
                if plot_fit_parameters_in_legend:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', label='Fit '+str(jj)+': '+fit_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
                else:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
        else:
            # Fitter == 'none'.
            fitted_values_to_be_returned.append(None)
            fitted_errors_to_be_returned.append(None)
        
        # Get residuals plot?
        if fitter != 'none':
            ## fit_results[0]: the fit curve.
            ##    resistances: the actual numbers measured.
            ##       residual: actual_y - predicted_y
            residuals = resistances - fit_results[0]
            
            if (not colourise):
                plt.figure(2) # Set figure 2 as active.
                plt.scatter(times, residuals, marker=marker_symbol, label='Residuals, '+file_label, color=colors[jj])## TODO!! colors(jj))
            else:
                plt.figure(2) # Set figure 2 as active.
                plt.scatter(times, residuals, marker=marker_symbol, label='Residuals, '+file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
    
    # Set axes' colour? Title colour? And so on.
    for mm in range(2):
        plt.figure(mm+1)
        plt.grid()
    
    if colourise:
        fig1.patch.set_alpha(0)
        fig2.patch.set_alpha(0)
        
        ax1.grid(color=get_colourise(-1))
        ax1.set_facecolor(get_colourise(-2))
        ax1.spines['bottom'].set_color(get_colourise(-1))
        ax1.spines['top'].set_color(get_colourise(-1))
        ax1.spines['left'].set_color(get_colourise(-1))
        ax1.spines['right'].set_color(get_colourise(-1))
        ax1.tick_params(axis='both', colors=get_colourise(-1))
        
        ax2.grid(color=get_colourise(-1))
        ax2.set_facecolor(get_colourise(-2))
        ax2.spines['bottom'].set_color(get_colourise(-1))
        ax2.spines['top'].set_color(get_colourise(-1))
        ax2.spines['left'].set_color(get_colourise(-1))
        ax2.spines['right'].set_color(get_colourise(-1))
        ax2.tick_params(axis='both', colors=get_colourise(-1))
    
    # Bump up the size of the ticks' numbers on the axes.
    ax1.tick_params(axis='both', labelsize=23)
    ax2.tick_params(axis='both', labelsize=23)
    
    # Extend axes to include the origin?
    ## Do not extend the x-axis if trying to plot the UNIX time.
    if (np.all(times > 0)) and (normalise_time):
        ax1.set_xlim(xmin=0)
    #if np.all(lowest_non_short_resistance_of_all >= -5):
    #    ax1.set_ylim(ymin=-5)
    
    # Other figure formatting.
    ## Figure out the label padding.
    if (normalise_resistances == 1):
        ax1.set_xlim(-10,310)
        if highest_number_on_the_y_axis < 8:
            ax1.set_ylim(-1,9)
            label_padding = 68
        elif highest_number_on_the_y_axis < 30:
            ax1.set_ylim(-1,25)
            label_padding = 49
        else:
            ax1.set_ylim(-8,105)
            label_padding = 30
    else:
        label_padding = 30
    
    
    if (not colourise):
        plt.figure(1) # Set figure 1 as active.
        plt.xlabel("Duration [s]", fontsize=33)
        plt.ylabel(y_label_text, fontsize=33, labelpad=label_padding)
        
        # Set title for this figure.
        plt.title(title_label, fontsize=38)
        
        plt.figure(2) # Set figure 2 as active.
        plt.xlabel("Duration [s]", fontsize=33)
        plt.ylabel(y_label_text, fontsize=33)
        if fitter != 'none':
            if fitter == 'second_order':
                plt.title("Residuals, 2nd-ord-polyn.", fontsize=38)
            elif fitter == 'third_order':
                plt.title("Residuals, 3rd-ord-polyn.", fontsize=38)
            elif fitter == 'exponential':
                plt.title("Residuals, exponential func.", fontsize=38)
            elif fitter == 'power':
                plt.title("Residuals, power-law", fontsize=38)
    else:
        plt.figure(1) # Set figure 1 as active.
        plt.xlabel("Duration [s]", color=get_colourise(-1), fontsize=33)
        plt.ylabel(y_label_text, color=get_colourise(-1), fontsize=33)
        plt.title("Resistance vs. Time", color=get_colourise(-1), fontsize=38)
    
    # Show shits.
    for ll in range(2):
        plt.figure(ll+1)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        if (not plot_fit_parameters_in_legend):
            plt.legend(fontsize=26)
        else:
            plt.legend()
        # Save plot?
        if savepath != '':
            plt.tight_layout()
            fig1.savefig(savepath, dpi=164, bbox_inches='tight')
    plt.show()
    
    # Return stuffs.
    return fitted_values_to_be_returned, fitted_errors_to_be_returned
    
def analyse_fitted_polynomial_factors(
    filepath,
    voltage_list_mV = ['auto'],
    normalise_resistances = 0,
    normalise_time = True,
    plot_no_junction_resistance_under_ohm = 0,
    fitter = 'second_order',
    skip_initial_dip = False,
    plot_fit_parameters_in_legend = False,
    colourise = False,
    savepath = '',
    ):
    ''' For a list of files, given as a list of filepath (strings),
        perform a fit for the whole file, and get the fit values
        back.
        
        Then, plot these fitted values versus a user-supplied voltage list.
        The value 'auto' vill analyse the file name in order to try to find
        XpYY, which defines X.YY volt.
    '''
    
    # Create a list, that will be filled with traces and their properties.
    # This list will be returned as the function ends.
    list_of_traces_in_plot = []
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Try to make voltage list?
    if voltage_list_mV[0] == 'auto':
        
        # User said yes.
        ## Clear out the list.
        voltage_list_mV = []
        for mm in range(len(filepath)):
            voltage_list_mV.append(0)
        
        # Go through files.
        for ii in range(len(filepath)):
            item = filepath[ii]
            
            # Pattern to match "_XpYY_" format, [V]
            match_volts = re.search(r'(\d+)p(\d+)', item)
            if match_volts:
                volts = float(f"{match_volts.group(1)}.{match_volts.group(2)}")
                voltage_list_mV[ii] = int(volts*1000)
            
            else:
                # Alternative: pattern to match the last number before ".csv"
                # Which, is in units of [mV]
                match_millivolts = re.search(r'(\d+)(?=\.csv$)', item)
                if match_millivolts:
                    voltage_list_mV[ii] = int(match_millivolts.group(1))
                
                else:
                    # At this point, the filepath could not be used to determine the voltage.
                    raise ValueError("Halted! Could not determine voltage used automatically from the file: '"+str(item)+"'")
    
    ## At this point, the voltage list is known.
    
    # Perform fits.
    assert fitter != 'none', "Halted! This function requires a fitter to be active, i.e., fitter != 'none'."
    (fitted_values, fitted_errors) = plot_active_manipulation(
        filepath = filepath,
        normalise_resistances = normalise_resistances,
        normalise_time = normalise_time,
        plot_no_junction_resistance_under_ohm = plot_no_junction_resistance_under_ohm,
        fitter = fitter,
        skip_initial_dip = skip_initial_dip,
        plot_fit_parameters_in_legend = plot_fit_parameters_in_legend,
        colourise = colourise,
        savepath = savepath
    )
    
    ## The data format here is weird:
    ## Each new ROW of fitted_values, contains information about the next
    ## datapoint on the Y axis. Each COLUMN of fitted values, contains
    ## this datapoint for a new TRACE in the plot. And, each value
    ## in fitted_errors, is the error bar.
    
    num_traces = max(len(arr) for arr in fitted_values)  # Max number of parameters
    num_points = len(fitted_values)  # Number of voltage points
    
    # Organise data by parameter index
    y_values = [[] for _ in range(num_traces)]
    y_errors = [[] for _ in range(num_traces)]
    for i in range(num_points):
        for j in range(len(fitted_values[i])):  # Iterate over parameters in each fit
            y_values[j].append(fitted_values[i][j])
            y_errors[j].append(fitted_errors[i][j])
    
    # Prepare labels for the plot.
    if fitter == 'second_order':
        fit_label_list = ['α', 'β']
    elif fitter == 'third_order':
        fit_label_list = ['α', 'β', 'δ']
    elif fitter == 'exponential':
        fit_label_list = ['γ', 'τ']
    elif fitter == 'power':
        fit_label_list = ['A', 'B']
    else:
        raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
    
    ## At this point, voltage_list_mV is the X axis.
    ## Similarly, y_values[i] is the Y axis.
    
    # If fitter is either 'second_order' or 'third order', then the first
    # parameter reveals the dependency on applied voltage.
    # Let's try to fit this dependency.    
    polynomial_fit_successful = False
    if (fitter == 'second_order') or (fitter == 'third_order'):
        polynomial_fit_successful = True
        try:
            ''' Plotting it as a log-lin diagram, reveals a straight line for the
            alpha fit parameter, so we have reason to suspect that the first-order
            resistance dependency to the applied voltage is exponential.
            // Christian 2025-03-09'''
            
            def exponential_func_for_alpha(v_mV, alpha_0, gamma):#, v_mott_mV):
                return alpha_0 * ((np.e)**(v_mV * gamma))
                #return alpha_0 * ((np.e)**((v_mV - v_mott_mV) * gamma))
            
            # Grab the α values and β values.
            alpha_values = y_values[0]
            beta_values  = y_values[1]
            
            # Here, sort the alpha_values versus the applied voltages.
            ##if voltage_list_mV == sorted(voltage_list_mV):
            ##    # The list is ordered, do nothing.
            ##    pass
            ##elif
            if voltage_list_mV == sorted(voltage_list_mV, reverse = True):
                # The list is reversed, sort it.
                ## IMPORTANT: remember that the beta_values
                ##            also must be reversed here.
                voltage_list_mV.reverse()
                alpha_values.reverse()
                beta_values.reverse()
            else:
                # The list is a mess.
                sorted_triples = sorted(zip(voltage_list_mV, alpha_values, beta_values))
                voltage_list_mV, alpha_values, beta_values = zip(*sorted_triples)
                voltage_list_mV = list(voltage_list_mV)
                alpha_values = list(alpha_values)
                beta_values = list(beta_values)
            
            ## I don't really know how to make a good guess for the scalar alpha_0.
            alpha_0_guess = 1.0
            
            ## For the gamma guess, take the slope of the ln curve.
            gamma_guess_vector_y = np.log(alpha_values)
            gamma_guess_alpha_fit = (gamma_guess_vector_y[-1] - gamma_guess_vector_y[0])/(voltage_list_mV[-1] - voltage_list_mV[0])
            
            ## For the V_mott guess, it's about 0.5 V for aliminium.
            v_mott_guess_mV = 500 # mV
            
            # Fit!
            optimal_vals_alpha_fit, covariance_mtx_of_opt_vals_alpha_fit = curve_fit(
                f     = exponential_func_for_alpha,
                xdata = voltage_list_mV,
                ydata = alpha_values,
                p0    = (alpha_0_guess, gamma_guess_alpha_fit)#, v_mott_guess_mV)
            )
            # Get fit errors.
            fit_err_alphas = np.sqrt(np.diag(covariance_mtx_of_opt_vals_alpha_fit))
            # Get fit curve for later.
            fit_curve_x_mV = np.linspace(0, np.max(voltage_list_mV), 100)
            fitted_curve_alphas = exponential_func_for_alpha(
                v_mV = fit_curve_x_mV,
                alpha_0 = optimal_vals_alpha_fit[0],
                gamma = optimal_vals_alpha_fit[1],
                ##v_mott_mV = optimal_vals_alpha_fit[2]
            )
        except RuntimeError:
            # Signal failed fit.
            polynomial_fit_successful = False
    
    # Plot each parameter trace.
    ## Create figure for plotting. Note that this is the plot with
    ## α₀ and γ fitted in it, that is, not the α(V) fit.
    if colourise:
        fig1, ax1 = plt.subplots(figsize=(12.8, 12.8), facecolor=get_colourise(-2))
    else:
        fig1, ax1 = plt.subplots(figsize=(12.8, 12.8))
    
    for i in range(num_traces):
        if y_values[i]:
            label_string = f'Parameter {fit_label_list[i]}'
            if colourise:
                plt.errorbar(voltage_list_mV, y_values[i], yerr=y_errors[i], marker='o', linestyle='-', capsize=3, label=label_string, color=get_colourise(i))
            else:
                plt.errorbar(voltage_list_mV, y_values[i], yerr=y_errors[i], marker='o', linestyle='-', capsize=3, label=label_string)
            
            # Append to list of traces to be returned.
            list_of_traces_in_plot += [[voltage_list_mV, y_values[i], label_string, y_errors[i]]]
    
    # Plot fit of fit?
    ## polynomial_fit_successful will be False if fitter
    ## was not set to 'second_order' or 'third_order'.
    if polynomial_fit_successful:
        fit_of_fit_label = 'f(V) = α₀·e^(V · γ)\n'
        for item in range(len(optimal_vals_alpha_fit)):
            # Find a proper exponent of the number.
            exponent       = np.floor(np.log10(np.abs( optimal_vals_alpha_fit[item] )))
            error_exponent = np.floor(np.log10(np.abs( fit_err_alphas[item] )))
            if item == 0:
                prefix = 'α₀'
            else:
                prefix = ' γ'
            fit_of_fit_label += prefix+': '+(f"{(optimal_vals_alpha_fit[item] * (10**(-exponent))):.3f}·10^{exponent} ±{(fit_err_alphas[item] * (10**(-error_exponent))):.3f}·10^{error_exponent}")
            if item != (len(optimal_vals_alpha_fit)-1):
                fit_of_fit_label += '\n'
        plt.plot(fit_curve_x_mV, fitted_curve_alphas, label = fit_of_fit_label)
        
        # Append to list that will be returned.
        list_of_traces_in_plot += [[fit_curve_x_mV, fitted_curve_alphas, fit_of_fit_label, None]]
    
    if colourise:
        plt.xlabel("Voltage [mV]", fontsize=33, color=get_colourise(-1))
        plt.ylabel("Fit parameters", fontsize=33, color=get_colourise(-1))
        plt.title("Fit parameter trends vs. voltage", fontsize=38, color=get_colourise(-1))
    else:
        plt.xlabel("Voltage [mV]", fontsize=33)
        plt.ylabel("Fit parameters", fontsize=33)
        plt.title("Fit parameter trends vs. voltage", fontsize=38)
    
    # Colourise axes, set axis limits, and such?
    plt.grid()
    ax1.set_xlim(xmin=0.0, xmax=1100)   # Include the zero for the voltage.
    ax1.set_ylim(ymax=0.6, ymin=-0.050)
    if colourise:
        fig1.patch.set_alpha(0)
        
        ax1.grid(color=get_colourise(-1))
        ax1.set_facecolor(get_colourise(-2))
        ax1.spines['bottom'].set_color(get_colourise(-1))
        ax1.spines['top'].set_color(get_colourise(-1))
        ax1.spines['left'].set_color(get_colourise(-1))
        ax1.spines['right'].set_color(get_colourise(-1))
        ax1.tick_params(axis='both', colors=get_colourise(-1))
    
    # Show and save shits.
    plt.legend(fontsize=26)
    plt.show()
    
    # Return a whole bunch of stuff, so that this function
    # can be used in the meta-function that calls it several times.
    return list_of_traces_in_plot

def analyse_multiple_sets_of_fitted_polynomial_factors(
    list_of_filepath_lists,
    voltage_list_mV = ['auto'],
    normalise_resistances = 0,
    normalise_time = True,
    plot_no_junction_resistance_under_ohm = 0,
    fitter = 'second_order',
    skip_initial_dip = False,
    plot_fit_parameters_in_legend = False,
    set_labels = [],
    colourise = False,
    savepath = '',
    ):
    ''' Analyse multiple sets of active resistance manipulation data.
        
        The list_of_filepath_lists argument should be a list, containing lists
        of filepaths. These filepaths correspond to resistance-vs-time
        manipulations done on Josephson junctions.
    '''
    
    # User argument sanitation.
    if fitter.lower() == 'none':
        raise ValueError("Halted! This function requires a fitter to be active; you provided 'none' as an argument here.")

    # Did the user label the sets?
    if (len(set_labels) != 0) and (not (len(list_of_filepath_lists) == len(set_labels))):
        raise ValueError("Halted! Unable to determine set labels, ensure that you did provide a set label for each filepath in your arguments.") 
    
    def insert_counter_before_extension(filepath, counter, default_ext=".png"):
        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)

        if not ext:
            ext = default_ext

        new_filename = f"{name}{counter}{ext}"
        return os.path.join(directory, new_filename)
    
    # Collect all of the data.
    results_of_sets = []
    repeat_counter = 1
    for i in range(len(list_of_filepath_lists)):
        current_set = list_of_filepath_lists[i]
        
        # Fit!
        results_of_sets.append(
            analyse_fitted_polynomial_factors(
                filepath = current_set,
                voltage_list_mV = voltage_list_mV,
                normalise_resistances = normalise_resistances,
                normalise_time = normalise_time,
                plot_no_junction_resistance_under_ohm = plot_no_junction_resistance_under_ohm,
                fitter = fitter,
                skip_initial_dip = skip_initial_dip,
                plot_fit_parameters_in_legend = plot_fit_parameters_in_legend,
                colourise = colourise,
                savepath = insert_counter_before_extension(savepath, i),
            )
        )
    
    # Plot the parameters in separate plots.
    ## How many parameters do we expect to see?
    if   fitter == 'second_order':
        expected_number_of_fit_parameters = 2
    elif fitter == 'third_order':
        expected_number_of_fit_parameters = 3
    elif fitter == 'exponential':
        expected_number_of_fit_parameters = 2
    elif fitter == 'power':
        expected_number_of_fit_parameters = 2
    
    # Make plots.
    ## Create figures for plotting.
    if colourise:
        fig1, ax1 = plt.subplots(figsize=(15, 11), facecolor=get_colourise(-2))
        fig2, ax2 = plt.subplots(figsize=(15, 11), facecolor=get_colourise(-2))
        if expected_number_of_fit_parameters == 3:
            fig3, ax3 = plt.subplots(figsize=(15, 11), facecolor=get_colourise(-2))
    else:
        fig1, ax1 = plt.subplots(figsize=(14.6, 9))
        fig2, ax2 = plt.subplots(figsize=(12.5, 9))
        if expected_number_of_fit_parameters == 3:
            fig3, ax3 = plt.subplots(figsize=(15, 11))
    
    # Figure out colours.
    if not colourise:
        num_items_to_colour = len(list_of_filepath_lists)
        if num_items_to_colour != 5:
            colour_label = 'tab20'
            colours = plt.cm.get_cmap(colour_label, num_items_to_colour)
        else:
            colours = ["#EE1C1C", "#C4EE1C", "#1CEE70", "#1C70EE", "#C41CEE"]
    
    # Loop through the data and create plotty things.
    for curr_figure in range(expected_number_of_fit_parameters):
        plt.figure( curr_figure+1 ) # Set figure "curr_figure" as active.
        
        # Go through the data.
        for sets in range(len(list_of_filepath_lists)):
            
            # Get the fitted parameter's voltages, data, label, and error bars.
            voltage_list_mV = results_of_sets[sets][curr_figure][0]
            parameter_data = results_of_sets[sets][curr_figure][1]
            
            if len(set_labels) > 0:
                ##label_string = results_of_sets[sets][curr_figure][2] + ', set '+str(set_labels[sets])
                label_string = ''+str(set_labels[sets])
            else:
                ##label_string = results_of_sets[sets][curr_figure][2] + ', set '+str(sets+1)
                label_string = ''+str(sets+1)
            errors = results_of_sets[sets][curr_figure][3]
            # Plot!
            if curr_figure == 0:
                linestyle = ''
            else:
                linestyle = '-'
            if colourise:
                plt.errorbar(voltage_list_mV, parameter_data, yerr=errors, marker='o', linestyle=linestyle, capsize=3, label=label_string, color=get_colourise(sets))
            else:
                if num_items_to_colour != 5:
                    plt.errorbar(voltage_list_mV, parameter_data, yerr=errors, marker='o', linestyle=linestyle, capsize=3, label=label_string, color=colours(sets))
                else:
                    plt.errorbar(voltage_list_mV, parameter_data, yerr=errors, marker='o', linestyle=linestyle, capsize=3, label=label_string, color=colours[sets])
            
            # Plot fit trace in the alpha plot?
            if curr_figure == 0:
                # Note that curr_figure here is always 0. The fit curve is
                # stored at index expected_number_of_fit_parameters, since the
                # first indices contain the fit parameter data.
                ## Do note that a failed fit results in no fit data being
                ## stored. Hence, a try-catch statement.
                try:
                    fit_x_axis = results_of_sets[sets][curr_figure+expected_number_of_fit_parameters][0]
                    fit_y_axis = results_of_sets[sets][curr_figure+expected_number_of_fit_parameters][1]
                    if plot_fit_parameters_in_legend:
                        fit_label  = results_of_sets[sets][curr_figure+expected_number_of_fit_parameters][2]
                    else:
                        fit_label = None
                    if not colourise:
                        if num_items_to_colour != 5:
                            plt.plot(fit_x_axis, fit_y_axis, label = fit_label, color = colours(sets))
                        else:
                            plt.plot(fit_x_axis, fit_y_axis, label = fit_label, color = colours[sets])
                    else:
                        plt.plot(fit_x_axis, fit_y_axis, label = fit_label, color = get_colourise(sets+0.1))
                except IndexError:
                    print("No fit data found in set "+str(sets)+".")
    
    # Include the origin and such.
    ax1.set_xlim(xmin=0.0, xmax=1100)
    
    # Make title.
    for ll in range(expected_number_of_fit_parameters):
        plt.figure(ll+1) # Set figure ll+1 as active.
        plt.grid()
        if (not colourise):
            title_colour = "#000000"
        else:
            title_colour = get_colourise(-1)
        
        plt.xlabel("Voltage [mV]", fontsize=33)
        if fitter == 'second_order':
            if (ll+1) == 1:
                plt.ylabel("Parameter α", fontsize=33)
            else:
                plt.ylabel("Parameter β", fontsize=33)
        else:
            print("WARNING: Unable to select appropriate Y-axis labels at this time.")
            plt.ylabel("Parameter value", fontsize=33)
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if fitter != 'none':
            if savepath == '':
                if fitter == 'second_order':
                    plt.title("Fit parameters, 2nd order polynomial", color=title_colour, fontsize=38)
                elif fitter == 'third_order':
                    plt.title("Fit parameters, 3rd order polynomial", color=title_colour, fontsize=38)
                elif fitter == 'exponential':
                    plt.title("Fit parameters, exponential function", color=title_colour, fontsize=38)
                elif fitter == 'power':
                    plt.title("Fit parameters, power-law", color=title_colour, fontsize=38)
            else:
                plt.title("", color=title_colour, fontsize=38)
        else:
            raise ValueError("Error! Could not understand argument provided to 'fitter': "+str(fitter))
        
        # Show legends.
        plt.legend(fontsize=26)
    
    # Grid colourisation?
    if colourise:
        fig1.patch.set_alpha(0)
        ax1.grid(color=get_colourise(-1))
        ax1.set_facecolor(get_colourise(-2))
        ax1.spines['bottom'].set_color(get_colourise(-1))
        ax1.spines['top'].set_color(get_colourise(-1))
        ax1.spines['left'].set_color(get_colourise(-1))
        ax1.spines['right'].set_color(get_colourise(-1))
        ax1.tick_params(axis='both', colors=get_colourise(-1))
    
        fig2.patch.set_alpha(0)
        ax2.grid(color=get_colourise(-1))
        ax2.set_facecolor(get_colourise(-2))
        ax2.spines['bottom'].set_color(get_colourise(-1))
        ax2.spines['top'].set_color(get_colourise(-1))
        ax2.spines['left'].set_color(get_colourise(-1))
        ax2.spines['right'].set_color(get_colourise(-1))
        ax2.tick_params(axis='both', colors=get_colourise(-1))
    
    # Save plots?
    if savepath != '':
        plt.tight_layout()
        fig1.savefig(insert_counter_before_extension(savepath, 10000), dpi=164, bbox_inches='tight')
        fig2.savefig(insert_counter_before_extension(savepath, 20000), dpi=164, bbox_inches='tight')
    
    # Show shits.        
    plt.show()
    
    return results_of_sets

def calculate_delta_f01(
    initial_resistance,
    final_resistance,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    verbose = True
    ):
    ''' Return the difference in frequency that the qubit has shifted,
        given the before and after resistances of the junction.
    '''
    
    # Get final frequency.
    final_frequency = calculate_f01_from_RT_resistance(
        room_temperature_resistance = final_resistance,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = verbose
    )
    
    # Get initial frequency.
    initial_frequency = calculate_f01_from_RT_resistance(
        room_temperature_resistance = initial_resistance,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = verbose
    )
    
    # Return difference.
    return (final_frequency - initial_frequency)

def acquire_relaxation_data_from_folder(
    folder_path,
    take_relaxation_data_at_this_time_s,
    filename_tags = [],
    verbose = True,
    ):
    ''' From a supplied folder path, acquire the resistance relaxation of the
        files held within.
    '''
    
    # Make lists to store the active and active+passive resistance increase.
    active_gain_percent  = []
    total_gain_percent   = []
    
    # User input sanitisation.
    if not os.path.isdir(folder_path):
        raise ValueError("Halted! Invalid path: "+str(folder_path))
    
    # Process all files yo.
    for filename in os.listdir(folder_path):
        # Ensure that all filenames are present.
        if all(keyword in filename for keyword in filename_tags):
            full_path = os.path.join(folder_path, filename)
            # Catch whether there are nasty subfolders.
            if os.path.isfile(full_path):
                
                ## TODO: Below, there is no catch to see whether
                ##       the user simply inserted a very very long time
                ##       for relaxation, and then simply let the sample run
                ##       for this very very long time. A "STOP_CREEP" is
                ##       inserted by the measurement apparatur.
                ##       Then, typically there is a new resistance
                ##       manipulation that triggers. So the file gets
                ##       'corrupted' into containing two manipulations.
                ##       In one case, such a file was discovered.
                
                # We only want .csv files.
                if filename.endswith(".csv"):
                    ## Set some process flags.
                    reference_resistance = 0.0
                    resistance_at_manipulation_finished = 0.0
                    relaxation_began_at_time = -1.0
                    relaxation_analysed_at_this_time = -1.0
                    resistance_at_relaxation_point = 0.0
                    shorted = False
                    
                    # Open file, do the thing.                    
                    with open(os.path.abspath(full_path), newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';')
                        rows = list(reader)  # Convert to list for indexing options
                        
                        # Go through the file.
                        for i in range(len(rows)):
                            # Every sixth row +3 contains a resistance value.
                            if (i % 6 == 3) and (not shorted):
                                
                                # We will constantly be on the lookout for
                                # broken junctions in the measurement.
                                try:
                                    if 'SHORTED' in rows[i+1][2]:
                                        # Broken junction. If we got our value,
                                        # so be it. Otherwise, just abort anyhow.
                                        shorted = True
                                except:
                                    # Then there is no such cell.
                                    pass
                                
                                # Grab some resistance as the starting value
                                # for the resistance manipulation?
                                if reference_resistance == 0.0:
                                    try:
                                        # Was this the start of the
                                        # resistance manipulation?
                                        if 'START_MANIPULATION' in rows[i+1][2]:
                                            # Oh shit, it was. Grab resistance.
                                            reference_resistance = float(rows[i][1])
                                            
                                            # Get the SI prefix for this data.
                                            ## TODO append more options, like MOhm.
                                            if '[kOhm]' in str(rows[i][0]):
                                                si_unit_prefix_scaler = 1000
                                            else:
                                                si_unit_prefix_scaler = 1
                                            
                                            # Scale to Ohm
                                            reference_resistance *= si_unit_prefix_scaler
                                            
                                    except IndexError:
                                        # In this case, there is simply nothing
                                        # written in such a cell.
                                        pass
                                
                                else:
                                    ## At this point, there is a reference
                                    ## resistance that we can work from.
                                    ## We want to know whether the upcoming
                                    ## resistance value is the one that the
                                    ## user wants.
                                    
                                    # Look for the tag signalling
                                    # that the manipulation is done.
                                    ## There are situations where
                                    ## this tag never appears.
                                    ## If that happens, simply assume
                                    ## that the measurement before
                                    ## the START_CREEP tag is the end
                                    ## of the manipulation.
                                    if 'STOP_MANIPULATION' in rows[i+1][2]:
                                        # Manipulation finished!
                                        # Grab the current resistance.
                                        resistance_at_manipulation_finished = float(rows[i][1])
                                        
                                        # Get the SI prefix for this data.
                                        ## TODO append more options, like MOhm.
                                        if '[kOhm]' in str(rows[i][0]):
                                            si_unit_prefix_scaler = 1000
                                        else:
                                            si_unit_prefix_scaler = 1
                                        
                                        # Scale to Ohm
                                        resistance_at_manipulation_finished *= si_unit_prefix_scaler
                                        
                                        # Grab the time at which this happened.
                                        # Every sixth row +4 contains a time value
                                        relaxation_began_at_time = float(rows[i+1][1])
                                    
                                    elif 'START_CREEP' in rows[i+1][2]:
                                        # Check whether this is a merged file,
                                        # that is, the resistance manipulation
                                        # simply stopped, and the operator
                                        # stopped the manipulation portion.
                                        if resistance_at_manipulation_finished == 0:
                                            # This happened, meaning that
                                            # the manipulation didn't reach
                                            # its target. Select the previous
                                            # resistance as the 'end datapoint'
                                            resistance_at_manipulation_finished = float(rows[i-6][1])
                                        
                                            # Get the SI prefix for this data.
                                            ## TODO append more options, like MOhm.
                                            if '[kOhm]' in str(rows[i-6][0]):
                                                si_unit_prefix_scaler = 1000
                                            else:
                                                si_unit_prefix_scaler = 1
                                            
                                            # Scale to Ohm
                                            resistance_at_manipulation_finished *= si_unit_prefix_scaler
                                            
                                            # Grab the time at which the relaxation started.
                                            try:
                                                relaxation_began_at_time = float(rows[i+1-6][1])
                                            except IndexError:
                                                relaxation_began_at_time = float(rows[i+1][1])
                                        
                                        # In either case, we continue
                                        # by looking for the user-set
                                        # time at which the resistance
                                        # of interest is located.
                                        relaxation_analysed_at_this_time = \
                                            relaxation_began_at_time + \
                                            take_relaxation_data_at_this_time_s
                                    
                                    # Now, we are merely waiting for the
                                    # resistance point taken at the time that
                                    # the user is interested in.
                                    if relaxation_analysed_at_this_time != -1:
                                        
                                        # Then, let's look for times.
                                        current_time = float(rows[i+1][1])
                                        if (current_time >= relaxation_analysed_at_this_time) and (resistance_at_relaxation_point == 0.0):
                                            
                                            ## Here, resistance_at_relaxation_point == 0.0 ensures that
                                            ## we only perform this part if the sample
                                            ## has not had its relaxation point resistance identified.
                                            
                                            # Fantastic, this resistance should probably be our final datapoint.
                                            # Let's just check whether the latest datapoint were
                                            # closer in time to that data, first.
                                            previous_time = float(rows[i+1-6][1])
                                            diff_current  = relaxation_began_at_time - current_time
                                            diff_previous = relaxation_began_at_time - previous_time
                                            if diff_current < diff_previous:
                                                # The datapoint that passed the
                                                # timestamp where the user
                                                # would have wanted the data,
                                                # is closer.
                                                ## VERIFY that the sample
                                                ## didn't die at precisely
                                                ## this point in time!
                                                try:
                                                    if 'SHORTED' in rows[i+1][2]:
                                                        # Junction dead at the
                                                        # finish line.
                                                        shorted = True
                                                except:
                                                    # Then there is no such cell.
                                                    pass
                                                
                                                if not shorted:
                                                    # Success!
                                                    resistance_at_relaxation_point = float(rows[i][1])
                                            else:
                                                # The previous datapoint was
                                                # closer in time to the user-
                                                # requested time.
                                                ## Here, we do not have to
                                                ## verify whether this
                                                ## resistance is taken at a
                                                ## point that is a short.
                                                ## Since, this fact would
                                                ## have been discovered by now.
                                                resistance_at_relaxation_point = float(rows[i-6][1])
                                            
                                            # Double-check whether we shorted
                                            # at the finish line.
                                            if not shorted:
                                                # Get the SI prefix for this data.
                                                ## TODO append more options, like MOhm.
                                                if '[kOhm]' in str(rows[i-6][0]):
                                                    si_unit_prefix_scaler = 1000
                                                else:
                                                    si_unit_prefix_scaler = 1
                                                
                                                # Scale to Ohm
                                                resistance_at_relaxation_point *= si_unit_prefix_scaler
                    
                    # At this point, we are done with the file.
                    # Collect values.
                    if reference_resistance != 0.0:
                        if resistance_at_manipulation_finished != 0.0:
                            if resistance_at_relaxation_point != 0.0:
                                if verbose:
                                    print("Ref res: "+str(reference_resistance)+", Res at manip finished: "+str(resistance_at_manipulation_finished)+", Res total: "+str(resistance_at_relaxation_point))
                                active_gain_percent.append((resistance_at_manipulation_finished / reference_resistance -1)*100)
                                total_gain_percent.append((resistance_at_relaxation_point / reference_resistance -1)*100)
                                
                                ## Removing this for now; there is no guarantee that the resistance hasn't
                                ## crept down below zero.
                                ##assert (resistance_at_relaxation_point / reference_resistance -1) > 0, "ERROR: "+str(filename)
                                
                                if (shorted) and (verbose):
                                    print(">> Shorted during relaxation, but after the sought-for data was found: "+str(filename))
                            else:
                                if verbose:
                                    print(">> Failed during relaxation: "+str(filename))
                        else:
                            if verbose:
                                print(">> Failed during manipulation: "+str(filename))
                    else:
                        if verbose:
                            print(">> Manipulation failed to start: "+str(filename))
                else:
                    print(">> Can't read file '"+str(filename)+"'")
            else:
                print(">> Can't process '"+str(filename)+"'")
    
    # Return things.
    if len(active_gain_percent) != len(total_gain_percent):
        raise RuntimeError("Error! The number of entries for the active manipulations does not match the number of successful manipulations. This is a bug. No. manipulations was: "+str(active_gain_percent)+", No. successes was: "+str(total_gain_percent))
    return (active_gain_percent, total_gain_percent)
    

def plot_trend_active_vs_total_resistance_gain(
    title_voltage_V,
    title_junction_size_nm,
    folder_path,
    take_relaxation_data_at_this_time_interval_s,
    filename_tags = [],
    outlier_threshold_in_std_devs = 2.0,
    consider_outliers = True,
    plot_ideal_curve = False,
    colourise = False,
    savepath = '',
    plot = True,
    enforce_n = -1,
    ):
    ''' For an interval of times to be observed, get the slope and offset
        data for the linear fits of the active-vs-total-manipulation
        experiments.
        
        enforce_n:  Set the minimum number of samples taken at some datapoint
                    to quality as fittable.
    '''
    list_of_slopes          = []
    list_of_slopes_err      = []
    list_of_offsets         = []
    list_of_offsets_err     = []
    list_of_n_samples       = []
    list_of_rms_deviations  = []
    latest_progress = 0.0
    last_progress_printed = -100.0
    
    # Keep last 5 timing intervals between progress prints
    progress_timestamps = deque(maxlen=10)
    
    for ii in range(len(take_relaxation_data_at_this_time_interval_s)):
        # Get progress.
        latest_progress = int(np.round((100*(ii/len(take_relaxation_data_at_this_time_interval_s)))))
        if not (latest_progress == last_progress_printed):
            ## #print(f"Progress: {100*(ii/len(take_relaxation_data_at_this_time_interval_s)):.1f} % done.")
            ## print("Progress: "+str(latest_progress)+" % done.")
            ## last_progress_printed = latest_progress
            
            now = time_module.time()
            
            # Calculate estimated remaining time if we have enough data
            eta_str = ""
            if last_progress_printed != -100.0:
                progress_timestamps.append(now)
                if len(progress_timestamps) >= 2:
                    # Compute average seconds per progress step
                    intervals = [
                        progress_timestamps[i] - progress_timestamps[i-1]
                        for i in range(1, len(progress_timestamps))
                    ]
                    avg_interval = sum(intervals) / len(intervals)
                    
                    remaining_steps = 100 - latest_progress
                    remaining_seconds = int(avg_interval * remaining_steps)
                    
                    # Convert to h, m, s
                    h, rem = divmod(remaining_seconds, 3600)
                    m, s = divmod(rem, 60)
                    eta_str = f", {h}h {m}' {s}\" remaining."
            
            print(f"{latest_progress} %{eta_str}")
            last_progress_printed = latest_progress
        
        # Get time!
        curr_investigated_time = take_relaxation_data_at_this_time_interval_s[ii]
        
        # Get data!
        curr_k, curr_m, curr_k_err, curr_m_err, n_samples, rms_deviation = plot_active_vs_total_resistance_gain(
            title_voltage_V = title_voltage_V,
            title_junction_size_nm = title_junction_size_nm,
            folder_path = folder_path,
            take_relaxation_data_at_this_time_s = curr_investigated_time,
            filename_tags = filename_tags,
            outlier_threshold_in_std_devs = outlier_threshold_in_std_devs,
            consider_outliers = consider_outliers,
            plot_ideal_curve = plot_ideal_curve,
            colourise = colourise,
            savepath = '',
            plot = False,
        )
        
        # Append to lists!
        list_of_slopes.append(curr_k)
        list_of_offsets.append(curr_m)
        list_of_slopes_err.append(curr_k_err)
        list_of_offsets_err.append(curr_m_err)
        list_of_n_samples.append(n_samples)
        list_of_rms_deviations.append(rms_deviation)
    
    # Conversions into numpy.
    k       = np.array(list_of_slopes)
    m       = np.array(list_of_offsets)
    k_err   = np.array(list_of_slopes_err)
    m_err   = np.array(list_of_offsets_err)
    n       = np.array(list_of_n_samples)
    rms_dev = np.array(list_of_rms_deviations)
    
    ## Plot!
    
    # Get time axis, i.e., x-axis.
    time = np.array(take_relaxation_data_at_this_time_interval_s)
    
    # How many datapoints must we have in the fitted point to qualify
    # as a reliable metric? → enforce_n
    if enforce_n != -1:
        # Check descending order
        for i in range(1, len(n)):
            if n[i] > n[i - 1]:
                print(f"Warning: 'n' is not in descending order at index {i}: "
                      f"n[{i-1}] = {n[i-1]}, n[{i}] = {n[i]}")
    # Enforce n-threshold for reported data accuracy.
    for i in range(len(n)):
        if n[i] < enforce_n:
            k[i] = None
            m[i] = None
            k_err[i] = None
            m_err[i] = None
            time[i]  = None
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25.31, 10), sharex=True)

    # Top plot: slope
    ax1.errorbar(time/3600, k, yerr=k_err, fmt='o', markersize=4, ecolor='gray', capsize=2, label='Slope', color="#EE1C1C")
    ax1.set_xlabel('Relaxation time [h]', fontsize=33)
    ax1.set_ylabel('Linear fit slope [-]', fontsize=33)
    #ax1.set_title('Slope and Offset vs Time')
    ax1.tick_params(axis='both', labelsize=23)
    ax1.set_ylim(0,1.21)
    ax1.grid(True)
    ax1.legend(fontsize=26, loc='lower right')

    # Bottom plot: offset
    ax2.errorbar(time/3600, m, yerr=m_err, fmt='o', markersize=4, ecolor='gray', capsize=2, label='Offset', color="#1CEE70")
    ax2.tick_params(axis='both', labelsize=23)
    ax2.set_ylim(0,8.5)
    ax2.set_xlabel('Relaxation time [h]', fontsize=33)
    ax2.set_ylabel('Linear fit offset [%]', fontsize=33)
    ax2.grid(True)
    ax2.legend(fontsize=26, loc='lower right')
    
    # Third plot: RMS deviation
    ax3.plot(time/3600, rms_dev, 'o-', color="#1C70EE", label='RMS deviation')
    ax3.tick_params(axis='both', labelsize=23)
    ax3.set_xlabel('Relaxation time [h]', fontsize=33)
    ax3.set_ylabel('Deviation from fit [%]', fontsize=33)
    ax3.set_ylim(0,8.5)
    ax3.grid(True)
    ax3.legend(fontsize=26, loc='lower right')
    
    ## # Insert plot.
    ## axins = ax2.inset_axes([0.55, 0.55, 0.4, 0.4])  # [x0, y0, width, height] in axes fraction
    ## axins.plot(time/3600, rms_dev, 'o-', color="#1C70EE", markersize=3)
    ## axins.set_title("RMS dev.", fontsize=26)
    ## axins.tick_params(axis='both', labelsize=24)
    ## axins.grid(True)
    
    # Go through the list of samples n, and append vertical lines to where this makes sense.
    print("TODO: add verical lines showing n of samples left.")
    
    # Tight layout!
    plt.tight_layout()
    
    # Save plot?
    if savepath != '':
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show!
    plt.show()
    
    # Return!
    return list_of_slopes, list_of_offsets, list_of_slopes_err, list_of_offsets_err, list_of_n_samples, list_of_rms_deviations


def plot_active_vs_total_resistance_gain(
    title_voltage_V,
    title_junction_size_nm,
    folder_path,
    take_relaxation_data_at_this_time_s,
    filename_tags = [],
    outlier_threshold_in_std_devs = 2.0,
    consider_outliers = True,
    plot_ideal_curve = False,
    colourise = False,
    savepath = '',
    plot = True,
    ):
    ''' Given some set of data, plot the active gain on the X axis in percent,
        and the total gain on the Y axis in percent relative to the
        initial resistance point. Also, make a fit.
        
        folder_path: path to folder containing measurement files.
        
        take_relaxation_data_at_this_time_s: time in seconds where the datapoint
        will be taken for the resistance manipulation. The point closest
        in time to the user-supplied value will be chosen.
        
        filename_tags: defines substrings that must be found in the filename
        for the file to be considered for data inclusion. If blank, then
        consider all files in the folder.
        
        Datapoints that fall outside of outlier_threshold_in_std_devs are
        considered outliers, this limit was set to 2 standard deviations
        from the mean.
    '''
    
    # Counter to keep track of colourised patterns.
    colourise_counter = 0
    
    # Detect special junction sizes for colourisation?
    dot_colour = "#D63834"
    if title_junction_size_nm == 200:
        dot_colour = "#EE1C1C"
    elif title_junction_size_nm == 350:
        dot_colour = "#1CEE70"
    elif title_junction_size_nm == 318:
        dot_colour = "#1C70EE"
    elif title_junction_size_nm == 354:
        dot_colour = "#C41CEE"
    
    # Acquire dataset.
    ## Rework voltage into something that will be written in the title.
    ## Append this number to the filename_tags to look for.
    filename_tags.append(str(title_voltage_V).replace('.',"p"))
    (active_gain_percent, total_gain_percent) = acquire_relaxation_data_from_folder(
        folder_path = folder_path,
        take_relaxation_data_at_this_time_s = take_relaxation_data_at_this_time_s,
        filename_tags = filename_tags,
        verbose = plot,
    )
    
    ##  Here are the two datasets for R01 and R02, i.e., the 354x354nm^2  ##
    ##  and 318x318nm^2 junction sizes from the JJTest100W3 wafer.        ##
    ##  That is, 354 is 'Compact 2', and 318 is 'Compact 1'.
    #active_gain_percent  = [2.45,  0.71,  10.05,  1.79,  5.02, 0.61, 3.52, 0.77, 1.01,  8.26, 2.39, 4.22, 4.15,  9.02,  7.71, 2.58, 1.77, 16.51, 5.13, 6.07, 11.03, 7.01, 0.28, 8.11, 2.75, 12.53, 13.16, 5.42, 1.60, 1.94, 4.97522988,  6.17057683, 7.01116346] ## THICK354
    #total_gain_percent   = [3.587, 2.180, 11.969, 2.999, 7.30, 2.76, 6.76, 2.17, 3.06, 10.55, 4.61, 5.70, 6.10, 10.85, 11.78, 6.45, 5.91, 19.07, 6.61, 8.23, 13.13, 8.93, 1.77, 9.74, 5.66, 14.68, 15.84, 7.13, 2.75, 3.36, 6.99326582, 10.37398798, 9.57568379] ## THICK354
    #active_gain_percent = [7.0942, 10.0738, 11.8119, 5.0343, 15.0296, 18.0029, 20.0640, 6.0434,  9.0186,  7.6048, 4.040, 10.007, 4.060, 6.020, 21.129, 3.577, 20.741, 3.043, 1.017, 0.78152682, 1.08007167, 2.1641953,  3.4114392 ] ## THICK318
    #total_gain_percent  = [9.7949, 12.2327, 14.0164, 6.9078, 17.4735, 20.3925, 22.1454, 8.2411, 11.1680, 10.2548, 5.986, 12.063, 5.804, 7.871, 23.058, 5.378, 23.186, 4.797, 2.859, 2.51228217, 4.02749426, 4.87928191, 6.21816964] ## THICK318
    
    # Sort lists together based on the active gain list.
    sorted_active, sorted_total = zip(*sorted(zip(active_gain_percent, total_gain_percent)))
    
    # Convert to numpy arrays!
    sorted_active = np.array(sorted_active, dtype=np.float64)
    sorted_total  = np.array(sorted_total, dtype=np.float64)
    
    # Report!
    if plot:
        print("Sorted active:", sorted_active)
        print("Sorted total:",  sorted_total)
    
    # Ensure the data sets are of identical length.
    if (len(sorted_active) != len(sorted_total)):
        raise ValueError("Error! The provided data sets do not have matching lengths; len(x) != len(y)")
        
    # Initial linear regression
    slope, intercept, r_value, p_value, std_err = linregress(sorted_active, sorted_total)
    
    # Compute residuals
    y_predicted = slope * sorted_active + intercept
    residuals   = sorted_total - y_predicted
    
    # Set threshold for outliers.
    threshold = outlier_threshold_in_std_devs * np.std(residuals)
    mask = np.abs(residuals) < threshold
    
    # Filter out outliers?
    active_filtered = sorted_active[mask]
    total_filtered  = sorted_total[mask]
    
    # Create figure for plotting.
    if plot:
        if colourise:
            fig, ax = plt.subplots(figsize=(12.8, 10), facecolor=get_colourise(-2))
            #plt.figure(figsize=(10, 5), facecolor=get_colourise(-2))
        else:
            fig, ax = plt.subplots(figsize=(12.5, 9.14))
            #plt.figure(figsize=(10, 5))
    
    # Let's fit the data and see what we get.
    def linear_func(x, k, m):
        return k * x + m
    
    def linear_fitter(
        sorted_active,
        sorted_total
        ):
        ## Let's guess initial guessing values.
        k_guess = (sorted_total[-1] - sorted_total[0]) / (sorted_active[-1] - sorted_active[0])
        m_guess = sorted_total[0]
        ## Fit!
        optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
            f     = linear_func,
            xdata = sorted_active,
            ydata = sorted_total,
            p0    = (k_guess, m_guess)
        )
        
        # Extract parameters.
        optimal_k = optimal_vals[0]
        optimal_m = optimal_vals[1]
        
        # Get the fit error.
        fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
        err_k = fit_err[0]
        err_m = fit_err[1]
        
        # Get a fit curve! Extend to zero, i.e., insert zero.
        fitted_curve = linear_func(
            x = sorted_active,
            k = optimal_k,
            m = optimal_m
        )
        
        # Return!
        return fitted_curve, optimal_k, optimal_m, err_k, err_m
    
    # Plot the fit curve!
    ## Extend to zero!
    original_fitted_curve, optimal_k, optimal_m, err_k, err_m = linear_fitter(sorted_active, sorted_total)
    if plot:
        if colourise:
            plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color=get_colourise(colourise_counter), label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
            colourise_counter += 1
        else:
            if savepath == '':
                plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color="#34D2D6", label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
            else:
                plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color="#222222", label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
        
    if consider_outliers:
        # Do the same with the filtered data.
        ## Remember to extend to 0.
        filtered_fitted_curve, filtered_k, filtered_m, filtered_err_k, filtered_err_m = linear_fitter(active_filtered, total_filtered)
        if plot:
            if colourise:
                plt.plot(np.insert(active_filtered, 0, 0), np.insert(filtered_fitted_curve, 0, filtered_m), color=get_colourise(colourise_counter), label=f"Trimmed fit: {filtered_k:.2f} · x + {filtered_m:.2f} [%]\nSlope error: ±{filtered_err_k:.2f}\nOffset error: ±{filtered_err_m:.2f}")
                colourise_counter += 1
            else:
                plt.plot(active_filtered, filtered_fitted_curve, color="#81D634", label=f"Trimmed fit: {filtered_k:.2f} · x + {filtered_m:.2f} [%]\nSlope error: ±{filtered_err_k:.2f}\nOffset error: ±{filtered_err_m:.2f}")
        
    # Plot an ideal curve?
    if plot:
        if plot_ideal_curve:
            if consider_outliers:
                plt.plot(np.insert(sorted_active, 0, 0), 1.00 * np.insert(sorted_active, 0, filtered_m) + filtered_m, color="#000000", label=f"Ideal trend: 1.0000 · x + {optimal_m:.2f} [%]")
            else:
                plt.plot(np.insert(sorted_active, 0, 0), 1.00 * np.insert(sorted_active, 0, optimal_m) + optimal_m, color="#000000", label=f"Ideal trend: 1.0000 · x + {optimal_m:.2f} [%]")
        
    '''
    ## Now, let's redo that with the outlier-filtered data and the original data.
    plt.plot(sorted_active, slope * sorted_active + intercept, color="#34D2D6", label=f"Fit with outliers: y = {slope:.2f}x + {intercept:.2f}\n±{std_err:.2f}")
    
    slope_filtered, intercept_filtered, r_value_filtered, p_value_filtered, std_err_filtered = linregress(active_filtered, total_filtered)
    plt.plot(active_filtered, slope_filtered * active_filtered + intercept_filtered, color="#34D2D6", label=f"Fit sans outliers: y = {slope_filtered:.2f}x + {intercept_filtered:.2f}\n{std_err_filtered:.2f}")
    '''
    
    # Insert datapoints!
    if plot:
        if (not colourise):
            if consider_outliers:
                plt.scatter(sorted_active, sorted_total, color="#8934D6", label=str(outlier_threshold_in_std_devs)+"σ outliers")
                ## This is a weird way to do this plotting, but it ensures that
                ## only the outliers are highlighted in the plot legend.
                ## Even though this dataset is the original! → Result = Good legend!
                plt.scatter(active_filtered, total_filtered, color="#D63834")
            else:
                plt.scatter(sorted_active, sorted_total, color=dot_colour)
        else:
            if consider_outliers:
                plt.scatter(sorted_active, sorted_total, color=get_colourise(3), label=str(outlier_threshold_in_std_devs)+"σ outliers")
                ## This is a weird way to do this plotting, but it ensures that
                ## only the outliers are highlighted in the plot legend.
                ## Even though this dataset is the original! → Result = Good legend!
                plt.scatter(active_filtered, total_filtered, color=get_colourise(2))
            else:
                plt.scatter(sorted_active, sorted_total, color=get_colourise(2))
        
        # Labels and such.
        plt.grid()
        if colourise:
            fig.patch.set_alpha(0)
            ax.grid(color=get_colourise(-1))
            ax.set_facecolor(get_colourise(-2))
            ax.spines['bottom'].set_color(get_colourise(-1))
            ax.spines['top'].set_color(get_colourise(-1))
            ax.spines['left'].set_color(get_colourise(-1))
            ax.spines['right'].set_color(get_colourise(-1))
            ax.tick_params(axis='both', colors=get_colourise(-1))
        
        # Bump up the size of the ticks' numbers on the axes.
        ax.tick_params(axis='both', labelsize=23)
        
        # Extend axes to include the origin?
        if np.all(sorted_active >= 0):
            ax.set_xlim(xmin=0, xmax=26.0)
        if np.all(sorted_total >= 0):
            ax.set_ylim(ymin=0, ymax=36.0)
        
        # Fancy colours?
        if (not colourise):
            plt.xlabel("Active manipulation [%]", fontsize=33)
            plt.ylabel("Total manipulation [%]", fontsize=33)
            #plt.title(f"Active vs. total manipulation\n30 minutes after stopping\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", fontsize=38)
            if savepath == '':
                plt.title(f"Active vs. total manipulation\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", fontsize=38)
        else:
            plt.xlabel("Active manipulation [%]", color=get_colourise(-1), fontsize=33)
            plt.ylabel("Total manipulation [%]",  color=get_colourise(-1), fontsize=33)
            print("WARNING: CHANGE BACK")
            if savepath == '':
                plt.title(f"Active vs. total manipulation\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", color=get_colourise(-1), fontsize=38)
        
        # Bump up legend.
        plt.legend(fontsize=26, loc='lower right')
        
        # Save plots?
        if savepath != '':
            plt.tight_layout()
            # Fix path name.
            base, _ = os.path.splitext(savepath)
            new_savepath = f"{base}_{title_junction_size_nm}nm.png"
            plt.savefig(new_savepath, dpi=164, bbox_inches='tight')
            
        # Show shits.
        plt.show()
    
    # How many samples were in the set?
    n_samples = len(sorted_active)
    
    # Let's calculate the RMS deviation.
    ## TODO: filtered_xx currently not supported here.
    
    ## Non-weighted RMS version here.
    residuals = sorted_total - (optimal_k * sorted_active + optimal_m)
    rms_deviation = np.sqrt(np.mean(residuals**2))
    return optimal_k, optimal_m, err_k, err_m, n_samples, rms_deviation
    ## Weighted RMS version here.
    ##residuals = sorted_total - (optimal_k * sorted_active + optimal_m)
    ##weighted_rms = np.sqrt(np.sum(n_samples * residuals**2) / np.sum(n_samples))
    ##return optimal_k, optimal_m, err_k, err_m, n_samples, weighted_rms
    
    
def plot_ac_voltage_for_biased_junction(
    dc_current,
    ac_current,
    ac_freq,
    critical_current
    ):
    
    time_axis = np.linspace(-25e-9, 25e-9, 20000)
    
    def ac_voltage_for_biased_junction(
        t,
        dc_current,
        ac_current,
        ac_freq,
        critical_current
        ):
        ''' Calculated by Christian Križan. '''
        
        omega = 2*np.pi*ac_freq
        
        mfq = (2.067833848e-15)/(2*np.pi)
        
        top_part = omega * (ac_current / critical_current) * np.cos(omega * t)
        
        inner_part_of_bottom = (dc_current + ac_current*np.sin( omega * t )) / critical_current
        
        bottom_part = np.sqrt( 1 - (inner_part_of_bottom)**2 )
        
        return mfq * (top_part / bottom_part)
    
    y_axis = ac_voltage_for_biased_junction(
        t = time_axis,
        dc_current = dc_current,
        ac_current = ac_current,
        ac_freq = ac_freq,
        critical_current = critical_current
        )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.grid()
    
    plt.xlabel("Time [µs]", fontsize=33)
    plt.ylabel("Voltage [nV]", fontsize=33)
    plt.title(f"Voltage over junction", fontsize=38)
    
    ax.tick_params(axis='both', labelsize=23)
    
    print(np.max(y_axis))
    
    plt.plot(time_axis*(1e6), y_axis*(1e9), color='green')
    plt.show()
    
def plot_critical_current_of_double_ScS_junction(  ):
    
    phi_phi0_axis = np.linspace(-5.5, 5.5, 20000)
    
    def function_cos(
        phi_phi0,
        ):
        ''' Calculated by Christian Križan. '''
        
        return np.abs(np.cos(np.pi * phi_phi0))
    
    y_axis = 2 * function_cos( phi_phi0_axis )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.grid()
    
    plt.xlabel("Φ/Φ₀ [-]", fontsize=33)
    plt.ylabel("I_c / I_s [-]", fontsize=33)
    plt.title(f"Critical current dependency on B", fontsize=38)
    
    ax.tick_params(axis='both', labelsize=23)
    
    print(np.max(y_axis))
    
    plt.plot(phi_phi0_axis, y_axis, color='purple')
    plt.show()
    
def plot_critical_current_of_triple_ScS_junction(  ):
    
    raise NotImplementedError("Halted! There is an error in the calculation below, do not proceed.")
    
    phi_phi0_axis = np.linspace(-5.5, 5.5, 20000)
    
    def function_triple_cos(
        phi_phi0,
        ):
        ''' Calculated by Christian Križan. '''
        
        return 2*np.abs(np.cos(np.pi * phi_phi0))+1
    
    y_axis = function_triple_cos( phi_phi0_axis )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.grid()
    
    plt.xlabel("Φ/Φ₀ [-]", fontsize=33)
    plt.ylabel("I_c / I_s [-]", fontsize=33)
    plt.title(f"Critical current dependency on B, 3 shorts", fontsize=38)
    
    ax.tick_params(axis='both', labelsize=23)
    
    print(np.max(y_axis))
    
    plt.plot(phi_phi0_axis, y_axis, color='orange')
    plt.show()

def plot_barplot_comparing_qubit_quality_factors(
    qubit_identifiers = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    ):
    raise NotImplementedError("Halted! Not done.")

    # Redefine the data since execution state was reset
    quarters = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]
    ch3_values_updated = [1676534, 1190290, 2487886, 1500422, 1014434, 1123330, 1186250, 1122872]
    ch5_values_updated = [840215, None, 1337651, None, 1297704, None, 1212784, None]
    ch3_values_twice_updated = [1302120, None, 1375136, None, None, 365889, 1397712, 1116143]
    ch5_values_twice_updated = [None, None, 1341931, None, 1860994, None, 163470, None]

    # Create two subplots: one for Ch3 comparison and one for Ch5 comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # First subplot: Ch3 (Reference vs Manipulated Reference)
    axes[0].bar(x - 0.2, reference, width=0.4, label="Reference", color="blue", alpha=0.7)
    axes[0].bar(x + 0.2, [v if v is not None else 0 for v in manipulated_reference], width=0.4, label="Manipulated reference", color="green", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(quarters)
    axes[0].set_title("Ch3 Comparison")
    axes[0].set_ylabel("Qubit quality factor")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].legend()
    for i in range(len(quarters)):
        if manipulated_reference[i] is None:
            axes[0].text(x[i] + 0.2, 500000, "N/A", ha="center", fontsize=9, color="green")

    # Second subplot: Ch5 (Manipulated vs Twice Manipulated)
    axes[1].bar(x - 0.2, [v if v is not None else 0 for v in manipulated], width=0.4, label="Manipulated", color="red", alpha=0.7)
    axes[1].bar(x + 0.2, [v if v is not None else 0 for v in twice_manipulated], width=0.4, label="Twice manipulated", color="orange", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(quarters)
    axes[1].set_title("Ch5 Comparison")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend()
    for i in range(len(quarters)):
        if manipulated[i] is None:
            axes[1].text(x[i] - 0.2, 500000, "N/A", ha="center", fontsize=9, color="red")
        if twice_manipulated[i] is None:
            axes[1].text(x[i] + 0.2, 500000, "N/A", ha="center", fontsize=9, color="orange")

    # Adjust layout and display
    plt.suptitle("Side-by-Side Comparisons: Ch3 and Ch5 Variants")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_quality_factor_vs_manipulation(
    savepath = '',
    plot_difference_plot = False
    ):
    ''' Given a set of quality factors, plot these against how much the
        qubits were manipulated.
    '''
    qubits = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    
    ch3_quality_reference = [1676534, 1190290, 2487886, 1500422, 1014434, 1123330, 1186250, 1122872]
    ch5_quality_single    = [840215,  None,    1337651, None,    1297704, None,    1212784, None   ]
    ch3_quality_single    = [1302120, None,    1375136, None,    None,    365889,  1397712, 1116143]
    ch5_quality_twice     = [None,    None,    1341931, None,    1860994, None,    163470,  None   ]
    
    ch3_manipulated_first_percent  = [0, 0, 0, 0, 0, 0, 0, 0]
    ch5_manipulated_first_percent  = [-8.004, 7.679, 9.126, 1.780, 6.760, 6.682, 6.677, -0.510]
    ch3_manipulated_second_percent = [3.912, 0.145, 1.765, 2.207, 11.123, 9.152, 4.108, 4.010]
    ch5_manipulated_second_percent = [0, 0, 1.315, 0, 0.545, 0, 0.517, 0]
    
    # Ch5 round 1:
    # Q1: 12-02-2025_17-01_JJTest100W3_Ch5_Q1_0p70V_MANIPULATE_7855p085_KriK.csv
    # Q2: 14-02-2025_14-03_JJTest100W3_Ch5_Q2_0p85V_MANIPULATE_AND_CREEP_9754p432_KriK.csv
    # Q3: 14-02-2025_17-26_JJTest100W3_Ch5_Q3_0p85V_MANIPULATE_AND_CREEP_8494p915_KriK.csv
    # Q4: 13-02-2025_17-32_JJTest100W3_Ch5_Q4_0p85V_MANIPULATE_AND_CREEP_KriK.csv
    # Q5: 16-02-2025_17-53_JJTest100W3_Ch5_Q5_0p85V_MANIPULATE_AND_CREEP_6692p460_KriK.csv
    # Q6: 17-02-2025_10-55_JJTest100W3_Ch5_Q8_0p85V_MANIPULATE_AND_CREEP_6836p235_KriK.csv
    # Q7: 17-02-2025_11-40_JJTest100W3_Ch5_Q7_0p85V_MANIPULATE_AND_CREEP_7341p045_KriK.csv
    # Q8: 17-02-2025_10-55_JJTest100W3_Ch5_Q8_0p85V_MANIPULATE_AND_CREEP_6836p235_KriK.csv
    
    # Ch5 round 2:
    # Q1: 2025-04-02_17-01_Ch5_Q1_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q2: -
    # Q3: 2025-04-02_18-17_Ch5_Q3_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q4: -
    # Q5: 2025-04-02_19-56_Ch5_Q5_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q6: -
    # Q7: 2025-04-02_20-15_Ch5_Q7_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q8: -
    
    # Ch3 round 2:
    # Q1: 2025-04-02_21-00_Ch3_Q1_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q2: 2025-04-03_18-47_Ch3_Q2_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q3: 2025-04-03_12-52_Ch3_Q3_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q4: 2025-04-03_18-09_Ch3_Q4_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q5: 2025-04-03_15-36_Ch3_Q5_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q6: 2025-04-03_17-52_Ch3_Q6_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q7: 2025-04-03_16-24_Ch3_Q7_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q8: 2025-04-03_16-58_Ch3_Q8_ROUND2_0p85V_MANIPULATE_KriK.csv
    
    # Convert all data to numpy arrays, replacing None with np.nan
    def to_np_array(data):
        return np.array([np.nan if x is None else x for x in data])
    
    ref_y = to_np_array(ch3_quality_reference)
    single_ch5_y = to_np_array(ch5_quality_single)
    single_ch3_y = to_np_array(ch3_quality_single)
    twice_ch5_y = to_np_array(ch5_quality_twice)
    
    ref_x = np.array(ch3_manipulated_first_percent)
    single_ch5_x = np.array(ch5_manipulated_first_percent)
    single_ch3_x = np.array(ch3_manipulated_second_percent)
    twice_ch5_x = np.array(ch3_manipulated_second_percent)
    
    # Compute qubit quality factor difference plots
    ch3_difference_quality = single_ch3_y - ref_y
    ch5_difference_quality = twice_ch5_y - single_ch5_y
    
    # Start figure with two subplots side by side
    if plot_difference_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 11), sharey=False)
    else:
        fig, ax1 = plt.subplots(1, figsize=(12.8, 11), sharey=False)
    
    symbol_list = ['s', '^', 'o', 'v', 'd', '*', 'x', 'p']
    for ii in range(len(qubits)):
        if ii == 0:
            label_to_use_1 = "Quantum device 1, unmanipulated"
            label_to_use_2 = "Quantum device 1, once manipulated"
            label_to_use_3 = "Quantum device 2, once manipulated"
            label_to_use_4 = "Quantum device 2, twice manipulated"
        else:
            label_to_use_1 = None
            label_to_use_2 = None
            label_to_use_3 = None
            label_to_use_4 = None
        
        if (ii == 6):
            bump_up_size = 140
        else:
            bump_up_size = 0
        
        ax1.scatter(ref_x[ii],        ref_y[ii]/1e6,        s=130+bump_up_size, label=label_to_use_1,  marker=symbol_list[ii], color="#EE1C1C")
        ax1.scatter(single_ch3_x[ii], single_ch3_y[ii]/1e6, s=130+bump_up_size, label=label_to_use_2,  marker=symbol_list[ii], color="#1CEE70")
        ax1.scatter(single_ch5_x[ii], single_ch5_y[ii]/1e6, s=130+bump_up_size, label=label_to_use_3,  marker=symbol_list[ii], color="#1C70EE")
        ax1.scatter(twice_ch5_x[ii],  twice_ch5_y[ii]/1e6,  s=130+bump_up_size, label=label_to_use_4,  marker=symbol_list[ii], color="#C41CEE")
    
    # Bump up the size of the ticks' numbers on the axes.
    ax1.tick_params(axis='both', labelsize=23)
    
    # Set up additional plot stuff.
    ## There is one datapoint that underwent negative active maipulation,
    ## that is, I knowingly performed the resistance manipulation such that
    ## it had a lower resistance by the time I stopped the manipulation.
    ## My examiner said that we should not put this datapoint in the upcoming
    ## paper, because we do not delve into the mechanism behind the negative
    ## resistance manipulation, that we do not understand what is going on,
    ## and putting this data in the paper is asking for trouble.
    #ax1.set_xlim(xmin=-13.0, xmax=13.0)
    
    ax1.set_xlim(xmin=-0.5, xmax=13.0)
    ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Active manipulation [%]", fontsize=33)
    ax1.set_ylabel("Qubit quality factor [10⁶]", fontsize=33)
    ax1.legend(fontsize=26)
    ax1.grid(True)
    
    # --- Difference Plot (right) ---
    if plot_difference_plot:
        ax2.scatter(single_ch3_x, ch3_difference_quality/1e6, s=120, label="Sample 1: once manip. - no manip.",    marker='o', color="#EE1C1C")
        ax2.scatter(twice_ch5_x,  ch5_difference_quality/1e6, s=120, label="Sample 2: twice manip. - once manip.", marker='s', color="#1CEE70")
        
        ax2.axhline(0, color='gray', linewidth=1.2, linestyle='--')  # Zero-line
        ax2.tick_params(axis='both', labelsize=23)
        ax2.set_xlim(xmin=-13.0, xmax=13.0)
        ax2.set_xlabel("Active manipulation [%]", fontsize=33)
        ax2.set_ylabel("Δ Qubit quality factor [10⁶]", fontsize=33)
        ax2.legend(fontsize=26)
        ax2.grid(True)
    
    # Tight layout.    
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show shit.
    plt.show()
    

def perform_stepped_manipulation_analysis(
    filepath_list,
    normalise_time = 1,
    study_relaxation_here_idx = 30,
    savepath = ''
    ):
    ''' Take a stepped manipulation measurement, and perform various
        analyses on it.
        
        normalise_time:
        0 = X axis will be in UNIX timestamp.
        1 = X axis is now a duration, with t₀ set at the timestamp of the
            very first datapoint.
    '''
    
    def unit_multiplier(unit_str):
        ''' Analyse the unit prefix for each datapoint.
        '''
        # Begin with the most common (and expected) resistance unit prefix.
        # Then, test a few other not-impossible alternatives.
        if 'kOhm' in unit_str:
            return 1e3
        elif 'MOhm' in unit_str:
            return 1e6
        elif 'GOhm' in unit_str:
            return 1e9
        elif 'TOhm' in unit_str:
            return 1e12
        elif 'TOhm' in unit_str:
            return 1e12
        
        elif 'hOhm' in unit_str:
            return 1e2
        elif 'daOhm' in unit_str:
            return 1e1
        elif 'dOhm' in unit_str:
            return 1e-1
        elif 'cOhm' in unit_str:
            return 1e-2
        elif 'mOhm' in unit_str:
            return 1e-3
        elif (('uOhm' in unit_str) or ('µOhm' in unit_str)):
            return 1e-6
        elif 'nOhm' in unit_str:
            return 1e-9
        
        elif 'Ohm' in unit_str:
            return 1
        else:
            raise ValueError("Halted! Cannot identify resistance prefix from datapoint: "+str(unit_str))
    
    def dump_csv(filepath):
        ''' For some .csv file containing resistance manipulation data,
            analyse this file and dump its contents.
        '''
        
        # Get CSV data.
        df = pd.read_csv(filepath, sep=';', header=None, dtype=str)
        
        resistances = []
        errorbars = []
        timestamps = []
        resistance_added = []
        annotations = {
            "START_MANIPULATION": [],
            "STOP_MANIPULATION": [],
            "START_CREEP": [],
            "STOP_CREEP": [],
            "SHORTED": []
        }
        
        for start_row in range(3, len(df), 6):  # 0-indexed, so row 4 is index 3
            # Extract unit and apply multiplier
            unit_cell = str(df.iat[start_row, 0])  # A4, A10, A16, etc.
            multiplier = unit_multiplier(unit_cell)
            
            # Extract resistance value and apply multiplier
            try:
                resistance_value = float(df.iat[start_row, 1]) * multiplier
            except (ValueError, TypeError):
                print("WARNING! Failed reading resistance value at data row "+str(start_row)+" (0-indexed).")
                resistance_value = np.nan  # Set to NaN
            resistances.append(resistance_value)
            
            # Extract error bar and apply multiplier
            try:
                errorbar_value = float(df.iat[start_row, 2]) * multiplier
            except (ValueError, TypeError):
                print("WARNING! Failed reading resistance error bar value at data row "+str(start_row)+" (0-indexed).")
                errorbar_value = np.nan  # Set to NaN
            errorbars.append(errorbar_value)
            
            # Extract timestamp
            try:
                timestamp_value = float(df.iat[start_row + 1, 1])  # B5, B11, etc.
            except (ValueError, TypeError):
                timestamp_value = np.nan
            timestamps.append(timestamp_value)
            
            # Check annotation string
            try:
                annotation = str(df.iat[start_row + 1, 2])  # C5, C11, etc.
            except IndexError:
                annotation = ""
            for keyword in annotations:
                if keyword in annotation:
                    annotations[keyword].append(len(resistances) - 1)
                    ##
                    ## Note: there is no guarantee that the target resistance
                    ##       was the resistance that was reached. Hence,
                    ##       this code snippet has been commented out.
                    ##
                    ##if keyword == 'START_MANIPULATION':
                    ##    try:
                    ##        suffix = annotation.strip().split('_')[-1]
                    ##        addition = float(suffix)
                    ##    except (IndexError, ValueError):
                    ##        print(f"WARNING! Failed to parse resistance addition from annotation: '{annotation}'")
                    ##        addition = np.nan
                    ##    resistance_added.append(addition)
        
        dumped_resistances = np.array(resistances)
        dumped_resistance_errorbars = np.array(errorbars)
        dump_timestamps = np.array(timestamps)
        ##resistance_added = np.array(resistance_added)
        
        return dumped_resistances, dumped_resistance_errorbars, dump_timestamps, annotations
    
    # Go through the entries.
    ## Let's instantiate the figure already at this point.
    if len( filepath_list ) == 2:
        fig1, axs1 = plt.subplots(1, 2, figsize=(31.1, 13))
        fig2, axs2 = plt.subplots(2, 2, figsize=(31.1, 13))
        fig3, axs3 = plt.subplots(1, 2, figsize=(25.4, 13))
    else:
        raise NotImplementedError("Todo!")
        fig1, axs1 = plt.subplots(2, 2, figsize=(12.8, 13))
    
    # During the process, we will try to figure out the y limit for plots
    # showing resistance increase values.
    y_lim_top = 0
    y_lim_top_relaxation = 0
    
    # Go through the selected files.
    for kk in range(len( filepath_list )):
    
        # Set filepath.
        filepath = filepath_list[kk]
    
        # Dump data!
        dumped_resistances, dumped_resistance_errorbars, dump_timestamps, annotations = dump_csv(filepath)
        
        ## Plot 1 logic goes here.
        
        # Check whether to update the resistances.
        resistance_added_ohm = []
        time_taken_for_manipulation = []

        # Number of valid manipulation intervals is the minimum of the lengths.
        n_intervals = min(len(annotations['START_MANIPULATION']), len(annotations['STOP_MANIPULATION']))

        # Loop through each valid pair, that is, we are on the lookout for
        # erroneous entries that could happen if the device broke during
        # manipulation.
        for i in range(n_intervals):
            start_idx = annotations['START_MANIPULATION'][i]
            stop_idx  = annotations['STOP_MANIPULATION' ][i]
            
            # Get resistance at start and stop indices
            start_res = dumped_resistances[start_idx]
            stop_res  = dumped_resistances[stop_idx ]
            
            # Get timestamp at start and stop indices
            start_time = dump_timestamps[start_idx]
            stop_time  = dump_timestamps[stop_idx ]
            
            # Get differences.
            delta_res  = stop_res  - start_res
            delta_time = stop_time - start_time
            
            # Store.
            resistance_added_ohm.append(delta_res)
            time_taken_for_manipulation.append(delta_time)
        
        # Numpy conversion.
        resistance_added_ohm = np.array(resistance_added_ohm)
        time_taken_for_manipulation = np.array(time_taken_for_manipulation)
        
        # Calculate resistance as percentages of increase relative
        # to the original resistance value.
        resistance_added_percent = 100 * (resistance_added_ohm/dumped_resistances[0])
        ## Important! The math here has been checked.
        ## There is not supposed to be a -1 anywhere;
        ## the statement here is correct as-is.
        
        # Set new y axis limits?
        print("Weird Y limits found for MPW005A, see code.")
        if kk < 1:
            if np.max(resistance_added_percent) > y_lim_top:
                y_lim_top = np.max(resistance_added_percent)*1.1
        
        ## Plot 2 logic goes here.
        relaxation_trace_data_xy = []
        relaxation_stack_xy = []
        
        # Number of valid manipulation intervals is the minimum of the lengths.
        n_intervals = min(len(annotations['START_CREEP']), len(annotations['STOP_CREEP']))
        
        # Loop through each valid pair, that is, we are on the lookout for
        # erroneous entries that could happen if the device broke during
        # manipulation.
        for i in range(n_intervals):
            start_relaxation_idx = annotations['START_CREEP'][i]
            stop_relaxation_idx  = annotations['STOP_CREEP' ][i]
            
            # Get time snippet from these indices.
            ## Here, we want to normalise the time. So, subtract time[0]
            ## for all entries.
            relaxation_x = dump_timestamps[start_relaxation_idx:stop_relaxation_idx] - dump_timestamps[start_relaxation_idx]
            
            # Get resistance snippet from these indices.
            ## Here, we want to perform a percent conversion.
            ## Then, we want to move all entries down to y = 0.
            
            relaxation_y = 100*(dumped_resistances[start_relaxation_idx:stop_relaxation_idx]/dumped_resistances[0])
            relaxation_y -= relaxation_y[0] # Move down to y = 0
            
            # Append!
            relaxation_trace_data_xy.append( (relaxation_x,relaxation_y) )
            
            ## Plot 2b logic goes here!
            relaxation_stack_xy.append( (dump_timestamps[start_relaxation_idx +study_relaxation_here_idx] - dump_timestamps[0], relaxation_y[study_relaxation_here_idx]) )
            
            # Figure out y axis for the plot later.
            if np.max(relaxation_y) > y_lim_top_relaxation:
                y_lim_top_relaxation = np.max(relaxation_y)*1.1
        
        ## Plot 3 logic goes here.
        # Here, we want to find the deepest trench of the dip,
        # for each manipulation.
        resistance_trench_depth_percent = []
        resistance_increase_at_start_of_trench_percent = []
        
        # Number of valid manipulation intervals is the minimum of the lengths.
        n_intervals = min(len(annotations['START_MANIPULATION']), len(annotations['STOP_MANIPULATION']))
        
        # Loop through each valid pair, that is, we are on the lookout for
        # erroneous entries that could happen if the device broke during
        # manipulation.
        for i in range(n_intervals):
            start_idx = annotations['START_MANIPULATION'][i]
            stop_idx  = annotations['STOP_MANIPULATION' ][i]
            
            # For the dip, we are interested in how much "resistance percent"
            # had been added at the beginning of the dip.
            start_res = dumped_resistances[start_idx]
            
            # Grab the lowest point in the resistance interval, and compare
            # it to the starting value.
            dip_value = dumped_resistances[start_idx:stop_idx].min()
            dip_value_percent = ((start_res - dip_value) / dumped_resistances[0])*100
            
            # Store!
            resistance_increase_at_start_of_trench_percent.append( ((start_res / dumped_resistances[0])-1)*100 )
            resistance_trench_depth_percent.append(dip_value_percent)
        
        # Numpy conversion.
        resistance_increase_at_start_of_trench_percent = np.array(resistance_increase_at_start_of_trench_percent)
        resistance_trench_depth_percent = np.array(resistance_trench_depth_percent)
        
        
        ## Plot 4 logic goes here.
        # Compute time derivative of the resistances.
        # The gradient analysis will yield better results for evenly spaced data.
        # Thus, let's try to interpolate the data given the timestamps they were
        # collected on.
        
        # Array conversion.
        dump_timestamps    = np.array(dump_timestamps)
        dumped_resistances = np.array(dumped_resistances)
        
        # Calculate median time step.
        dt = np.median(np.diff(dump_timestamps))
        uniform_time = np.arange(dump_timestamps[0], dump_timestamps[-1], dt)
        
        # Interpolate resistance onto uniform time grid
        interp_func = interp1d(dump_timestamps, dumped_resistances, kind='linear', fill_value="extrapolate")
        uniform_resistance = interp_func(uniform_time)
        
        # Compute time derivatives
        dR_dt   = np.gradient(uniform_resistance, dt)
        d2R_dt2 = np.gradient(dR_dt, dt)
        d3R_dt3 = np.gradient(d2R_dt2, dt)
        
        ## Figure stuff!
        # Figure out colours and label tags.
        if ('mpw005a' in filepath.lower()):
            label_tag = '350 nm dense-oxide'
            colour_list = interpolate_hsv_colours("#C4EE1C", "#EE1C1C", len(relaxation_trace_data_xy))
        elif ('jjanneal01c' in filepath.lower()):
            label_tag = '200 nm sparse-oxide'
            colour_list = interpolate_hsv_colours("#1C70EE", "#1CEE70", len(relaxation_trace_data_xy))
        else:
            label_tag = None
            colour_list = interpolate_hsv_colours("#1C70EE", "#1CEE70", len(relaxation_trace_data_xy))
        
        ## Subplot 1: Time needed to add X resistance.
        
        axs1[0].scatter(time_taken_for_manipulation/60, resistance_added_percent, s=160, color=colour_list[-1], label=label_tag)
        axs1[0].plot(time_taken_for_manipulation/60, resistance_added_percent, '--', color="#000000")
        #axs1[0].set_title("Time derivatives of resistance")
        axs1[0].set_xlabel("Time needed for step [min]", fontsize=33)
        axs1[0].set_ylabel("Resistance increase during step [%]", fontsize=33)
        axs1[0].legend(fontsize=24)
        axs1[0].grid(True)
        
        # Set limits, scales, ticks.
        axs1[0].set_ylim(-0.5, y_lim_top)
        axs1[0].set_xscale('log')
        axs1[0].tick_params(axis='both', labelsize=23)
        
        ## Subplot 2:
        
        # Plot!
        for ii in range(len(relaxation_trace_data_xy)):
            axs2[0,kk].scatter(relaxation_trace_data_xy[ii][0]/60, relaxation_trace_data_xy[ii][1], color=colour_list[ii])
        
        # Grid.
        axs2[0,kk].grid(True)
        
        # Axis labels.
        axs2[0,kk].set_xlabel("Duration [min]", fontsize=33)
        axs2[0,kk].set_ylabel("Resistance increase [%]", fontsize=33)
        axs2[0,kk].tick_params(axis='both', labelsize=23)
        
        ## Subplot 3:
        
        # Plot!
        axs1[1].scatter(resistance_increase_at_start_of_trench_percent, resistance_trench_depth_percent, s=80, color=colour_list[-1], label=label_tag)
        #axs1[1].plot(   resistance_increase_at_start_of_trench_percent, resistance_trench_depth_percent,       color=colour_list[-1]                 )
        
        # Axis labels.
        axs1[1].set_xlabel("Resistance increase at manipulation start [%]", fontsize=33)
        axs1[1].set_ylabel("Largest resistance drop [%]", fontsize=33)
        
        # Grid and stuff!
        axs1[1].grid(True)
        axs1[1].tick_params(axis='both', labelsize=23)
        axs1[1].legend(fontsize=24)
        
        ## Subplot 4: Does the relaxation effect stack?
        
        # Figure out time axis.
        if normalise_time == 1:
            time_axis = uniform_time - uniform_time[0]
        else:
            time_axis = uniform_time
        
        # Plot!
        axs2[1,kk].plot(time_axis, dR_dt,   label=r"$\frac{dR}{dt}$",     color='tab:red')
        axs2[1,kk].plot(time_axis, d2R_dt2, label=r"$\frac{d^2R}{dt^2}$", color='tab:orange')
        axs2[1,kk].plot(time_axis, d3R_dt3, label=r"$\frac{d^3R}{dt^3}$", color='tab:purple')
        #axs2[1,kk].set_title("Time derivatives of resistance")
        axs2[1,kk].set_xlabel("Time [s]", fontsize=33)
        axs2[1,kk].set_ylabel("Rate of resistance change [$\Omega / s$, $\Omega^2 / s^2$, $\Omega^3 / s^3$]", fontsize=33)
        axs2[1,kk].legend(fontsize=24)
        axs2[1,kk].grid(True)
        axs2[1,kk].tick_params(axis='both', labelsize=23)
        
        # Set axis limits.
        axs2[1,kk].set_ylim(-15.0, +15.0)
        
        # Insert vertical lines for manipulation events
        plotted_labels = set()
        for idx in annotations.get("START_MANIPULATION", []):
            ts = dump_timestamps[idx]
            
            # Plot duration instead?
            if normalise_time == 1:
                ts -= dump_timestamps[0]
            
            label = "Start manipulation" if "start" not in plotted_labels else None
            axs2[1,kk].axvline(ts, color='cyan', linestyle='--', linewidth=1.5, label=label)
            plotted_labels.add("start")
        for idx in annotations.get("STOP_MANIPULATION", []):
            ts = dump_timestamps[idx]
            
            # Plot duration instead?
            if normalise_time == 1:
                ts -= dump_timestamps[0]
            
            label = "Stop manipulation" if "stop" not in plotted_labels else None
            axs2[1,kk].axvline(ts, color='magenta', linestyle='--', linewidth=1.5, label=label)
            plotted_labels.add("stop")
        
        ## Fig 3 i.e. subplot 5
        # Plot!
        for ii in range(len(relaxation_trace_data_xy)):
            if kk < 1:
                if ii == 0:
                    axs3[0].scatter(relaxation_trace_data_xy[ii][0]/60, relaxation_trace_data_xy[ii][1], color=colour_list[ii], label=label_tag)
                else:
                    axs3[0].scatter(relaxation_trace_data_xy[ii][0]/60, relaxation_trace_data_xy[ii][1], color=colour_list[ii], label="_something")
            
            # Add line to show where the relaxation is being studied.
            if ii == 0:
                axs3[0].axvline(relaxation_trace_data_xy[ii][0][study_relaxation_here_idx]/60, color="#EE1C1C", linestyle='--', linewidth=1.5, label="_something")
            
            # Right plot.
            axs3[1].plot(relaxation_stack_xy[ii][0]/3600, relaxation_stack_xy[ii][1], color=colour_list[ii])
            if ii == len(relaxation_stack_xy)-1:
                axs3[1].scatter(relaxation_stack_xy[ii][0]/3600, relaxation_stack_xy[ii][1], s=90, color=colour_list[ii], label=label_tag)
            else:
                axs3[1].scatter(relaxation_stack_xy[ii][0]/3600, relaxation_stack_xy[ii][1], s=90, color=colour_list[ii], label="_something")
            
            # Limits.
            ## Use same axes for the (a) subplot.
            axs3[0].set_ylim(-0.5, y_lim_top)
            axs3[1].set_ylim(-0.5, y_lim_top)
            ##axs3[0].set_ylim(-0.5,y_lim_top_relaxation)
            ##axs3[1].set_ylim(-0.5,y_lim_top_relaxation)
            
            # Grid.
            axs3[0].grid(True)
            axs3[1].grid(True)
            
            # Axis labels.
            axs3[0].set_xlabel("Duration [min]", fontsize=33)
            axs3[0].set_ylabel("Resistance increase [%]", fontsize=33)
            axs3[0].tick_params(axis='both', labelsize=23)
            axs3[1].set_xlabel("Time since beginning [h]", fontsize=33)
            axs3[1].set_ylabel("Resistance increase [%]", fontsize=33)
            axs3[1].tick_params(axis='both', labelsize=23)
            
            # Legends!
            axs3[0].legend(fontsize=24)
            axs3[1].legend(fontsize=24)
    
    # Tight!
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
        plt.figure(1)
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
        
        plt.figure(3)
        plt.savefig(savepath.replace(".png","_excerpt3.png"), dpi=164, bbox_inches='tight')
    
    # Plot shits!
    plt.show()
    
    
    # Return shits.
    return dumped_resistances, dumped_resistance_errorbars, dump_timestamps, annotations
    

def compare_junction_oxidation_dose_to_known_dataset(
    path_to_reference_data_file,
    list_of_normal_resistances_times_area_ohm_micrometer_squared = [],
    list_of_oxidation_times_in_minutes = [],
    list_of_oxidation_pressures_in_mbar = [],
    user_label_list = [],
    plot_reference_fit = False
    ):
    ''' Plots your values for some oxidation process that you have obtained,
        against the reference dataset.
        
        The reference fit comes from J. Phys. d: Appl. Phys. 48 (2015) 395308.
        
        The format of the dataset should be an .xlsx file.
        
        Format of file:
            A1: blank   B1: "R_sg"  C1: "R_n"   D1: blank   E1: "Area"  F1: "R_n * A"   G1: "G/A"       H1: "p*t"       I1: "t_ox"  J1: "p_ox"  K1: blank   L1: "t"
            A2: blank   B2: blank   C2: blank   D2: blank   E2: "µm^2"  F2: "µm^2"      G2: "mS/µm^2"   H2: "mbar s"    I2: "sec."  J2: "mbar"  K2: blank   L2: "min."
            C3:
                Insert cell data here! Remember that column K is blank.
    '''
    # User input sanitation.
    if not ((len(list_of_normal_resistances_times_area_ohm_micrometer_squared) == len(list_of_oxidation_times_in_minutes)) and (len(list_of_normal_resistances_times_area_ohm_micrometer_squared) == len(list_of_oxidation_pressures_in_mbar))):
        raise ValueError("Halted! The lengths of the input datasets do not match.\n"+\
        "len(list_of_normal_resistances_times_area_ohm_micrometer_squared): "+str(len(list_of_normal_resistances_times_area_ohm_micrometer_squared))+"\n"+\
        "len(list_of_oxidation_times_in_minutes): "+str(len(list_of_oxidation_times_in_minutes))+"\n"+\
        "len(list_of_oxidation_pressures_in_mbar): "+str(len(list_of_oxidation_pressures_in_mbar))
        )
    if not (len(user_label_list) == len(list_of_oxidation_pressures_in_mbar)):
        raise ValueError("Halted! The user-provided label list has an unexpected size. Did you forget something?")
    
    ## Calculate effective oxygen dose, using an empirical model:
    ## Fig. 4 in L. J. Zeng et al. 2015, J. Phys. D: Appl. Phys. 48 395308
    def effective_oxygen_dose(t_minutes, p_mbar):
        ''' t_minutes:  oxidation time in minutes
            p_mbar:     oxidation pressure in mbar
        '''
        return (t_minutes**0.65) * (p_mbar**0.43)
    
    # Open datafile, extract data.
    df = pd.read_excel(path_to_reference_data_file)

    # Extract data from columns I, J, and F.
    # Recall that Excel columns are 0-indexed in pandas as 8, 9, 5.
    # Extract data (skip first two rows: header and units)
    t_ox_seconds = df.iloc[2:, 8].dropna().to_numpy()       # Column I (t in seconds)
    p_ox_mbar = df.iloc[2:, 9].dropna().to_numpy()          # Column J (p in mbar)
    normal_state_resistance = df.iloc[2:, 5].dropna().to_numpy()  # Column F (R_N·A in Ω·µm^2)
    
    # Ensure all lists are the same length (safe fallback)
    min_len = min(len(t_ox_seconds), len(p_ox_mbar), len(normal_state_resistance))
    t_ox_seconds = t_ox_seconds[:min_len]
    p_ox_mbar = p_ox_mbar[:min_len]
    normal_state_resistance = normal_state_resistance[:min_len]
    
    # Convert time to minutes.
    t_ox_minutes = t_ox_seconds / 60
    
    # Calculate effective oxygen dose.
    dose = effective_oxygen_dose(t_ox_minutes, p_ox_mbar)
    
    # Plot setup.
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.loglog(dose, normal_state_resistance, 'o', color="#000000", markerfacecolor='none', label="J. Phys. D: Appl. Phys. 48 (2015) 395308")
    plt.xlabel("D = t₀^0.65 · p₀^0.43 [min^0.65 · mbar^0.43]", fontsize=33)
    plt.ylabel("Normal resistance [Ω · µm^2]", fontsize=33)
    plt.title("R_N vs. oxygen dose", fontsize=38)
    
    ## Now, plot the user-added data?
    if len(list_of_normal_resistances_times_area_ohm_micrometer_squared) != 0:
        
        for ii in range(len(list_of_normal_resistances_times_area_ohm_micrometer_squared)):
            normal_resistances_ohm_micrometer_squared = np.array(list_of_normal_resistances_times_area_ohm_micrometer_squared[ii])
            oxidation_times_in_minutes = np.array(list_of_oxidation_times_in_minutes[ii])
            oxidation_pressures_in_mbar = np.array(list_of_oxidation_pressures_in_mbar[ii])
            
            # Get user dose.
            user_dose = effective_oxygen_dose(oxidation_times_in_minutes, oxidation_pressures_in_mbar)
            
            # Plot user things.
            if ii == 0:
                marker_string = 'p'
            elif ii == 1:
                marker_string = '*'
            elif ii == 2:
                marker_string = 'H'
            elif ii == 3:
                marker_string = '^'
            elif ii == 4:
                marker_string = 'x'
            elif ii == 5:
                marker_string = 'v'
            else:
                marker_string = 'o'
            plt.loglog(user_dose, normal_resistances_ohm_micrometer_squared, marker_string, markersize = 8, label=user_label_list[ii])
    
    # Plot reference fit line?
    if plot_reference_fit:
        x_fit = np.linspace(1e-1, 1e2, 1000)
        def fit_func(x):
            return 57.25 * (x**(1.0139))
        plt.plot(x_fit, fit_func(x_fit), ':', color="red")
    
    # Axes adjustments.
    #plt.ylim(2e0, 3e10)#3e3)
    #plt.xlim(1e-1, 3e6)#1e2)
    ax.tick_params(axis='both', labelsize=23)
    
    # Grid and legend.
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=24)
    plt.tight_layout()
    
    # Show stuff!
    plt.show()

def compare_aging_vs_junction_sizes(
    savepath = ''
    ):
    ''' Reconstruct plots from Maurizio Toselli's thesis and his raw data,
        but adjust for his later reported inaccuracy, that is, the x-axis
        in the thick-oxide plot is incorrect due to him being given the
        wrong junction size numbers.
        
        Important: the Y data is extracted from the rastered plot,
        this Y data is thus accurate when viewed in print, but less accurate
        when worked with digitally.
    '''
    
    def transform(data):
        ''' Changes reference from 1.00 being the start, into [%] of increase.
            That is, 20 means that the sample has increased its resistance
            by 20 %'''
        return [(x, (y - 1) * 100) for x, y in data]

    def shift_data(data, reference_day_timestamp, shift_with_this_many_days):
        ''' Accounts for the thick-oxide being measured
            <shift_with_this_many_days> days after the wafer's manufacture.
            AND, also convert the time entry into decimal-days.
        '''
        def convert_string_into_timestamp(string_format):
            dt = datetime.strptime(string_format, "%Y_%m_%d__%H_%M_%S")
            return dt.timestamp()
        
        # Get reference.
        reference = convert_string_into_timestamp(reference_day_timestamp)
        
        # Fix the number! Remember that the X of the double (i.e., entry 0)
        # is supposed to be a number in days, not seconds.
        # Hence, divide by 86400
        return [((convert_string_into_timestamp(x) -reference)/86400 +shift_with_this_many_days, y) for x, y in data]
    
    def unpack(data):
        return zip(*data)
    
    ## Junction size data!
    
    thin_x  = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    thin_y  = np.array([33.247058823529414, 28.2, 22.41176470588235, 21.24705882352941, 18.03529411764706, 16.129411764705882, 15.494117647058824, 13.905882352941177, 12.882352941176471, 12.458823529411765])
    thick_x = np.array([150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    thick_y = np.array([3.2823529411764705, 3.0352941176470587, 2.788235294117647, 2.611764705882353, 2.2588235294117647, 1.9058823529411764, 1.9058823529411764, 1.7647058823529411, 1.2352941176470589, 1.2352941176470589])
    
    ## Time data!
    
    # Thin 1!
    '''aging_vs_time_100nm_daily_25nA = transform([
        (1, 1.0),
        (2, 1.0629310344827585),
        (3, 1.118103448275862),
        (4, 1.1534482758620688),
        (5, 1.171551724137931),
        (6, 1.2293103448275862),
        (7, 1.2551724137931033),
        (8, 1.2698275862068964),
        (9, 1.285344827586207),
        (10, 1.3060344827586206),
        (11, 1.314655172413793),
        (12, 1.3275862068965516),
        (13, 1.3370689655172412),
        (14, 1.3474137931034482),
        (15, 1.3568965517241378)
    ])
    aging_vs_time_100nm_weekly_25nA = transform([
        (1, 1.0),
        (9, 1.2672413793103448),
        (15, 1.3206896551724137)
    ])
    aging_vs_time_100nm_biweekly_25nA = transform([
        (1, 1.0),
        (4, 1.139655172413793),
        (9, 1.2784482758620688),
        (12, 1.311206896551724),
        (15, 1.3353448275862068)
    ])
    aging_vs_time_100nm_bimonthly_25nA = transform([
        (1, 1.0),
        (15, 1.3681034482758618)
    ])'''
    reference_day_thin1 = "2024_05_14__15_40_00" # Estimate from fabrication logs.
    aging_vs_time_100nm_daily_25nA = transform([ #C12R0
        ("2024_05_16__18_53_31", 1.0),
        ("2024_05_17__14_42_13", 1.0629310344827585),
        ("2024_05_18__10_57_36", 1.118103448275862),
        ("2024_05_19__12_44_11", 1.1534482758620688),
        ("2024_05_20__09_59_14", 1.171551724137931),
        ("2024_05_21__17_59_17", 1.2293103448275862),
        ("2024_05_22__14_15_12", 1.2551724137931033),
        ("2024_05_23__18_25_15", 1.2698275862068964),
        ("2024_05_24__18_00_44", 1.285344827586207),
        ("2024_05_25__15_39_32", 1.3060344827586206),
        ("2024_05_26__17_24_21", 1.314655172413793),
        ("2024_05_27__16_37_56", 1.3275862068965516),
        ("2024_05_28__13_35_14", 1.3370689655172412),
        ("2024_05_29__17_43_22", 1.3474137931034482),
        ("2024_05_30__09_50_49", 1.3568965517241378)
    ])
    aging_vs_time_100nm_weekly_25nA = transform([ #C20R0
        ("2024_05_16__19_02_33", 1.0),
        ("2024_05_24__18_44_14", 1.2672413793103448),
        ("2024_05_30__11_00_02", 1.3206896551724137)
    ])
    aging_vs_time_100nm_biweekly_25nA = transform([ #C16R0
        ("2024_05_16__18_59_23", 1.0),
        ("2024_05_19__12_46_39", 1.139655172413793),
        ("2024_05_24__18_05_19", 1.2784482758620688),
        ("2024_05_27__16_35_34", 1.311206896551724),
        ("2024_05_30__09_56_09", 1.3353448275862068)
    ])
    aging_vs_time_100nm_bimonthly_25nA = transform([ #C8R1
        ("2024_05_16__19_06_03", 1.0),
        ("2024_05_30__09_53_56", 1.3681034482758618)
    ])
    #C17R0:
    ## Day 09: 2024_05_24__18_41_00
    ## Day 12: 2024_05_27__17_08_39
    ## D1y 15: 2024_05_30__10_58_21
    
    
    # Thin 2!
    '''aging_vs_time_500nm_daily_1uA = transform([
        (1, 1.0),
        (2, 1.0192307692307692),
        (3, 1.0365384615384614),
        (4, 1.0466346153846153),
        (5, 1.05625),
        (6, 1.0860576923076923),
        (7, 1.0923076923076922),
        (8, 1.0985576923076923),
        (9, 1.103846153846154),
        (10, 1.1091346153846153),
        (11, 1.1139423076923076),
        (12, 1.1173076923076923),
        (13, 1.1216346153846153),
        (15, 1.1264423076923076)
    ])
    aging_vs_time_500nm_weekly_1uA = transform([
        (1, 1.0),
        (9, 1.1028846153846152),
        (15, 1.1245192307692307)
    ])
    aging_vs_time_500nm_biweekly_1uA = transform([
        (1, 1.0),
        (9, 1.1004807692307692),
        (12, 1.1115384615384616),
        (15, 1.1206730769230768)
    ])
    aging_vs_time_500nm_bimonthly_1uA = transform([
        (1, 1.0),
        (15, 1.1245192307692307)
    ])'''
    reference_day_thin2 = "2024_05_14__15_40_00" # Estimate from fabrication logs.
    aging_vs_time_500nm_daily_1uA = transform([ #C12R3
        ("2024_05_16__19_28_35", 1.0), # Suffix C18EZNR
        ("2024_05_17__14_48_41", 1.0192307692307692),
        ("2024_05_18__11_08_32", 1.0365384615384614),
        ("2024_05_19__13_22_13", 1.0466346153846153),
        ("2024_05_20__10_08_57", 1.05625),
        ("2024_05_21__18_14_36", 1.0860576923076923),
        ("2024_05_22__14_23_06", 1.0923076923076922),
        ("2024_05_23__18_53_42", 1.0985576923076923),
        ("2024_05_24__18_22_49", 1.103846153846154),
        ("2024_05_25__15_48_05", 1.1091346153846153),
        ("2024_05_26__17_29_55", 1.1139423076923076),
        ("2024_05_27__16_50_06", 1.1173076923076923),
        ("2024_05_28__13_46_16", 1.1216346153846153),
        ("2024_05_30__10_15_26", 1.1264423076923076)
    ])
    aging_vs_time_500nm_weekly_1uA = transform([ #C20R3
        ("2024_05_16__19_30_39", 1.0), ## C8R5?? ## Double-check this entry.
        ("2024_05_24__18_55_53", 1.1028846153846152),
        ("2024_05_30__10_46_12", 1.1245192307692307)
    ])
    aging_vs_time_500nm_biweekly_1uA = transform([ #C17R3
        ("2024_05_16__19_32_35", 1.0), ## C8R5?? ## Double-check this entry.
        ("2024_05_24__18_57_36", 1.1004807692307692),
        ("2024_05_27__17_17_59", 1.1115384615384616),
        ("2024_05_30__10_48_25", 1.1206730769230768)
    ])
    aging_vs_time_500nm_bimonthly_1uA = transform([ #C16R3
        ("2024_05_16__19_34_48", 1.0), ## C8R5?? ## Double-check this entry.
        ("2024_05_30__10_19_11", 1.1245192307692307)
    ])
    
    # Thick 1
    reference_day_thick1 = "2023_11_14__10_40_00" # Estimate from deposition start +2h 20' = (1h 10' per leg)
    aging_vs_time_200nm_daily_25nA = transform([ #C0R1
        ("2023_12_03__17_14_08", 1.0),
        ("2023_12_04__11_38_57", 1.003953934740883),
        ("2023_12_05__12_58_10", 1.0056429942418426),
        ("2023_12_06__10_34_38", 1.0016506717850289),
        ("2023_12_07__11_34_43", 1.0070249520153551),
        ("2023_12_08__14_51_25", 1.010556621880998),
        ("2023_12_09__12_11_53", 1.0222264875239924),
        ("2023_12_10__16_34_42", 1.0260652591170825),
        ("2023_12_11__18_15_21", 1.0252975047984645),
        ("2023_12_12__17_12_24", 1.02468330134357),
        ("2023_12_13__11_53_18", 1.022380038387716),
        ("2023_12_14__13_41_19", 1.0259117082533589),
        ("2023_12_15__13_37_40", 1.031439539347409),
        ("2023_12_16__12_26_33", 1.0325143953934741)
    ])
    aging_vs_time_200nm_weekly_25nA = transform([ # C4R1
        ("2023_12_03__17_19_20", 1.0),
        ("2023_12_10__17_01_03", 1.023147792706334),
        ("2023_12_16__12_28_58", 1.0252975047984645)
    ])
    aging_vs_time_200nm_biweekly_25nA = transform([ # C2R1
        ("2023_12_03__17_17_27", 1.0)
    ])
    aging_vs_time_200nm_bimonthly_25nA = transform([ #C6R1
        ("2023_12_03__17_21_27", 1.0),
        ("2023_12_16__12_30_08", 1.0268330134357007)
    ])
    
    # Thick 2
    reference_day_thick2 = "2023_11_14__10_40_00" # Estimate from deposition start +2h 20' = (1h 10' per leg)
    aging_vs_time_600nm_daily_1uA = transform([ # C0R4
        ("2023_12_03__17_43_31", 1.0),
        ("2023_12_04__11_47_28", 1.0018809980806143),
        ("2023_12_05__13_09_55", 1.0046449136276392),
        ("2023_12_06__11_16_50", 1.0009596928982725),
        ("2023_12_07__12_06_06", 1.003953934740883),
        ("2023_12_08__14_57_22", 1.0057965451055662),
        ("2023_12_09__12_23_49", 1.0113243761996162),
        ("2023_12_10__16_42_54", 1.0159309021113243),
        ("2023_12_11__18_22_06", 1.017658349328215),
        ("2023_12_12__17_18_23", 1.0163915547024953),
        ("2023_12_13__12_05_27", 1.0155854126679462),
        ("2023_12_14__13_48_46", 1.0154702495201535),
        ("2023_12_15__13_42_58", 1.0159309021113243),
        ("2023_12_16__12_42_15", 1.019616122840691)
    ])
    aging_vs_time_600nm_weekly_1uA = transform([ # C4R4
        ("2023_12_03__17_47_19", 1.0),
        ("2023_12_10__16_49_49", 1.0133973128598848),
        ("2023_12_16__12_44_32", 1.0140882917466412)
    ])
    aging_vs_time_600nm_biweekly_1uA = transform([ # C2R4
        ("2023_12_03__17_44_58", 1.0),
        ("2023_12_06__11_19_26", 1.0025719769673704),
        ("2023_12_10__16_45_43", 1.0163915547024953),
        ("2023_12_13__12_06_50", 1.0152399232245681),
        ("2023_12_16__12_43_23", 1.0190403071017275)
    ])
    aging_vs_time_600nm_bimonthly_1uA = transform([ # C6R4
        ("2023_12_03__17_49_24", 1.0),
        ("2023_12_16__12_45_43", 1.0120153550863724)
    ])
    
    ## Account for the old dense-oxide devices being much older
    ## than the old sparse-oxide devices.
    aging_vs_time_100nm_daily_25nA     = shift_data( aging_vs_time_100nm_daily_25nA,     reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_100nm_weekly_25nA    = shift_data( aging_vs_time_100nm_weekly_25nA,    reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_100nm_biweekly_25nA  = shift_data( aging_vs_time_100nm_biweekly_25nA,  reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_100nm_bimonthly_25nA = shift_data( aging_vs_time_100nm_bimonthly_25nA, reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_daily_1uA      = shift_data( aging_vs_time_500nm_daily_1uA,      reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_weekly_1uA     = shift_data( aging_vs_time_500nm_weekly_1uA,     reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_biweekly_1uA   = shift_data( aging_vs_time_500nm_biweekly_1uA,   reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_bimonthly_1uA  = shift_data( aging_vs_time_500nm_bimonthly_1uA,  reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_200nm_daily_25nA     = shift_data( aging_vs_time_200nm_daily_25nA,     reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_200nm_weekly_25nA    = shift_data( aging_vs_time_200nm_weekly_25nA,    reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_200nm_biweekly_25nA  = shift_data( aging_vs_time_200nm_biweekly_25nA,  reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_200nm_bimonthly_25nA = shift_data( aging_vs_time_200nm_bimonthly_25nA, reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_daily_1uA      = shift_data( aging_vs_time_600nm_daily_1uA,      reference_day_thick2, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_weekly_1uA     = shift_data( aging_vs_time_600nm_weekly_1uA,     reference_day_thick2, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_biweekly_1uA   = shift_data( aging_vs_time_600nm_biweekly_1uA,   reference_day_thick2, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_bimonthly_1uA  = shift_data( aging_vs_time_600nm_bimonthly_1uA,  reference_day_thick2, shift_with_this_many_days = 0)#20)
    
    # Create figure.
    fig = plt.figure(figsize=(31.09, 19.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.25)
    
    # Axes: left is subplot 1, right is merged 2+3
    ax1 = fig.add_subplot(gs[0])
    ax2_3 = fig.add_subplot(gs[1])
    
    # --- Subplot 1 (junction size vs resistance) ---
    ax1.scatter(thin_x, thin_y, s=80, color="#1C70EE", label="Aging, sparse-oxide")
    ax1.scatter(thick_x, thick_y, s=80, color="#EE1C1C", label="Aging, dense-oxide")
    ax1.set_xlim(0, 700)
    ax1.set_ylim(-1, 55)
    ax1.tick_params(axis='both', labelsize=23)
    ax1.grid(True, which="both", ls="--")
    ax1.set_xlabel("Junction width [nm]", fontsize=33)
    ax1.set_ylabel("Resistance increase [%]", fontsize=33)
    ax1.legend(fontsize=26)
    
    thin_colour_large_HSV  = ['#1cee70', '#1ceea4', '#1ceecb', '#1cddee', '#1cc0ee', '#1c9fee', '#1c84ee', '#1c70ee']
    thick_colour_large_HSV = ['#ee1c1c', '#ee5d1c', '#ee9d1c', '#eec01c', '#e6dd1c', '#d1ec1c', '#b9ee1c', '#c4ee1c']
    
    '''
    ax2.scatter(*unpack(aging_vs_time_100nm_daily_25nA),     marker='o', label="25 nA, soft 100 nm, daily",     color=thin_colour_large_HSV[0], s=80)
    ax2.scatter(*unpack(aging_vs_time_100nm_weekly_25nA),    marker='^', label="25 nA, soft 100 nm, weekly",    color=thin_colour_large_HSV[1], s=80)
    ax2.scatter(*unpack(aging_vs_time_100nm_biweekly_25nA),  marker='s', label="25 nA, soft 100 nm, biweekly",  color=thin_colour_large_HSV[2], s=80)
    ax2.scatter(*unpack(aging_vs_time_100nm_bimonthly_25nA), marker='*', label="25 nA, soft 100 nm, bimonthly", color=thin_colour_large_HSV[3], s=80)
    
    ax2.scatter(*unpack(aging_vs_time_500nm_daily_1uA),     marker='o', label="1 µA, soft 500 nm, daily",     color=thin_colour_large_HSV[4], s=80)
    ax2.scatter(*unpack(aging_vs_time_500nm_weekly_1uA),    marker='^', label="1 µA, soft 500 nm, weekly",    color=thin_colour_large_HSV[5], s=80)
    ax2.scatter(*unpack(aging_vs_time_500nm_biweekly_1uA),  marker='s', label="1 µA, soft 500 nm, biweekly",  color=thin_colour_large_HSV[6], s=80)
    ax2.scatter(*unpack(aging_vs_time_500nm_bimonthly_1uA), marker='*', label="1 µA, soft 500 nm, bimonthly", color=thin_colour_large_HSV[7], s=80)
    
    ax2.grid(True, which="both", ls="--")
    ax2.set_xlabel("Time [days]", fontsize=33)
    ax2.set_ylabel("Resistance increase [%]", fontsize=33)
    ax2.tick_params(axis='both', labelsize=23)
    ax2.legend(fontsize=24)
    ax2.set_xlim(0, 16)
    ax2.set_ylim(-1, 55)
    
    
    # ... Third subplot.
    ax3.scatter(*unpack(aging_vs_time_200nm_daily_25nA),     marker='o', label="25 nA, hard 200 nm, daily",     color=thick_colour_large_HSV[0], s=80)
    ax3.scatter(*unpack(aging_vs_time_200nm_weekly_25nA),    marker='^', label="25 nA, hard 200 nm, weekly",    color=thick_colour_large_HSV[1], s=80)
    ax3.scatter(*unpack(aging_vs_time_200nm_biweekly_25nA),  marker='s', label="25 nA, hard 200 nm, biweekly",  color=thick_colour_large_HSV[2], s=80)
    ax3.scatter(*unpack(aging_vs_time_200nm_bimonthly_25nA), marker='*', label="25 nA, hard 200 nm, bimonthly", color=thick_colour_large_HSV[3], s=80)
    
    ax3.scatter(*unpack(aging_vs_time_600nm_daily_1uA),     marker='o', label="1 µA, hard 600 nm, daily",     color=thick_colour_large_HSV[4], s=80)
    ax3.scatter(*unpack(aging_vs_time_600nm_weekly_1uA),    marker='^', label="1 µA, hard 600 nm, weekly",    color=thick_colour_large_HSV[5], s=80)
    ax3.scatter(*unpack(aging_vs_time_600nm_biweekly_1uA),  marker='s', label="1 µA, hard 600 nm, biweekly",  color=thick_colour_large_HSV[6], s=80)
    ax3.scatter(*unpack(aging_vs_time_600nm_bimonthly_1uA), marker='*', label="1 µA, hard 600 nm, bimonthly", color=thick_colour_large_HSV[7], s=80)
    
    ax3.grid(True, which="both", ls="--")
    ax3.set_xlabel("Time [days]", fontsize=33)
    ax3.set_ylabel("Resistance increase [%]", fontsize=33)
    ax3.tick_params(axis='both', labelsize=23)
    ax3.legend(fontsize=24)
    ax3.set_xlim(0, 16)
    ax3.set_ylim(-1, 55)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    if savepath != '':
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
        print("Figure saved to: " + str(savepath))'''
    
    '''
    YE OLDE
    
    ##plt.scatter(thin_x,  thin_y,  s=60, color="#0BB5F4", label="Aging, soft oxide")
    ##plt.scatter(thick_x, thick_y, s=80, color="#F44A0B", label="Aging, hard oxide")
    plt.scatter(thin_x,  thin_y,  s=80, color="#1C70EE", label="Aging, soft oxide")
    plt.scatter(thick_x, thick_y, s=80, color="#EE1C1C", label="Aging, hard oxide")
    
    # Axes adjustments.
    plt.ylim(0, 40)
    plt.xlim(0, 700)
    ax.tick_params(axis='both', labelsize=23)
    
    # Grid, legend, axis labels, title.
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=24)
    plt.xlabel("Junction width [nm]", fontsize=33)
    plt.ylabel("Resistance increase [%]", fontsize=33)
    
    ## As of writing, pope Leo XIV got announced. Random.
    plt.tight_layout()
    
    # Save?
    if savepath != '':
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
        print("Figure saved to: "+str(savepath))
    
    '''
    
    ## --- Subplot 2+3 combined (log-time aging) ---
    # --- Subplot 2+3 combined (lin-time aging) ---
    # Plot thin oxide datasets
    thin_sets = [
        aging_vs_time_100nm_daily_25nA, aging_vs_time_100nm_weekly_25nA,
        aging_vs_time_100nm_biweekly_25nA, aging_vs_time_100nm_bimonthly_25nA,
        aging_vs_time_500nm_daily_1uA, aging_vs_time_500nm_weekly_1uA,
        aging_vs_time_500nm_biweekly_1uA, aging_vs_time_500nm_bimonthly_1uA
    ]
    thick_sets = [
        aging_vs_time_200nm_daily_25nA, aging_vs_time_200nm_weekly_25nA,
        aging_vs_time_200nm_biweekly_25nA, aging_vs_time_200nm_bimonthly_25nA,
        aging_vs_time_600nm_daily_1uA, aging_vs_time_600nm_weekly_1uA,
        aging_vs_time_600nm_biweekly_1uA, aging_vs_time_600nm_bimonthly_1uA
    ]
    
    markers = ['o', '^', 's', '*']
    labels  = ['daily', 'weekly', 'biweekly', 'bimonthly']
    sizes_thin  = ['25 nA, 100 nm,','25 nA, 100 nm,','25 nA, 100 nm,','25 nA, 100 nm,','1 µA, 500 nm,','1 µA, 500 nm,','1 µA, 500 nm,','1 µA, 500 nm,']
    sizes_thick = ['25 nA, 200 nm,','25 nA, 200 nm,','25 nA, 200 nm,','25 nA, 200 nm,','1 µA, 600 nm,','1 µA, 600 nm,','1 µA, 600 nm,','1 µA, 600 nm,']
    thin_colors  = thin_colour_large_HSV
    thick_colors = thick_colour_large_HSV

    for i, (data, color) in enumerate(zip(thin_sets, thin_colors)):
        marker = markers[i % 4]
        ax2_3.scatter(*unpack(data), marker=marker, color=color, s=80, label=f"Sparse {sizes_thin[i]} {labels[i % 4]}")

    for i, (data, color) in enumerate(zip(thick_sets, thick_colors)):
        marker = markers[i % 4]
        ax2_3.scatter(*unpack(data), marker=marker, color=color, s=80, label=f"Dense {sizes_thick[i]} {labels[i % 4]}")

    ##ax2_3.set_xscale('log') # Log unnecessary
    ax2_3.set_ylim(-1, 55)
    ax2_3.set_xlim(-0.8, 32.8)
    ax2_3.grid(True, which="both", ls="--")
    ax2_3.tick_params(axis='both', labelsize=23)
    ax2_3.set_xlabel("Time since manufacture [days]", fontsize=33)
    ax2_3.set_ylabel("Resistance increase [%]", fontsize=33)
    ax2_3.legend(fontsize=26, ncol=2, loc='upper right')
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
        print("Saved to ", savepath)
    plt.show()
    
def plot_free_energy_vs_total_current_of_rf_squid():
    
    # Assume zero externally applied magnetic field.
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ##Phi_0 = 2.067833848e-15 [Wb]
    ##L = 100e-12 # [H]
    I_C = 100e-6  # [A]
    
    def beta_LRF(L, I_C):
        return 2*np.pi * (L*I_C)/(2.067833848e-15)
    
    def energy_in_rf_squid( i, beta_LRF):
        return (1/2) * beta_LRF * (i**2) - np.cos(beta_LRF * i)
    
    # Find energy, specifically E_tot / E_J
    currents = np.linspace(-1.5, 1.5, 400)
    free_energy_normalised_100pH = energy_in_rf_squid( currents, beta_LRF(100e-12, I_C) )
    free_energy_normalised_10pH = energy_in_rf_squid( currents, beta_LRF(50e-12, I_C) )
    free_energy_normalised_1pH = energy_in_rf_squid( currents, beta_LRF(10e-12, I_C) )
    free_energy_normalised_0p1pH = energy_in_rf_squid( currents, beta_LRF(0.1e-12, I_C) )
    
    
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.plot(currents, free_energy_normalised_100pH, label="L: 100 pH, I_C: 100 µA, β: "+f"{beta_LRF(100e-12, I_C):.2f}")
    plt.plot(currents, free_energy_normalised_10pH, label="L: 50 pH, I_C: 100 µA, β: "+f"{beta_LRF(50e-12, I_C):.2f}")
    plt.plot(currents, free_energy_normalised_1pH, label="L: 10 pH, I_C: 100 µA, β: "+f"{beta_LRF(10e-12, I_C):.2f}")
    plt.plot(currents, free_energy_normalised_0p1pH, label="L: 0.1 pH, I_C: 100 µA, β: "+f"{beta_LRF(0.1e-12, I_C):.2f}")
    
    plt.ylabel("E_tot / E_ J [-]", fontsize=33)
    plt.xlabel("I_s / I_C [-]", fontsize=33)
    ax.tick_params(axis='both', labelsize=23)
    
    plt.grid()
    plt.legend(fontsize=20)
    plt.show()
    
def plot_dip_removed(
    savepath = ''
    ):
    ''' Illustrate how much dip [s] was removed from the beginning of the active manipulation data traces.
    '''
    
    # Data: Dip removed vs. Voltage
    datasets = [
        {"voltages": [950, 900, 850, 800, 750], "dip_removed": [0, 0, 0, 0, 0]},
        {"voltages": [950, 900, 850, 800, 750], "dip_removed": [3.856, 0, 0, 0, 0]},
        {"voltages": [1000, 950, 925, 900, 850, 800], "dip_removed": [10.307, 7.097, 65.764, 30.316, 218.010, 146.505]},
        {"voltages": [1000, 950, 925, 900, 850, 800], "dip_removed": [7.063, 65.427, 26.532, 52.200, 133.327, 143.178]},
        {"voltages": [1050, 1000, 950, 900], "dip_removed": [0, 0, 0, 0]},
    ]
    colours = ['#C41CEE', '#1C70EE', '#1CEE70', '#C4EE1C', '#EE1C1C']
    #labels = ["Set (a): 200x200 nm, soft", "Set (b): 300x300 nm, soft", "Set (c): 318x318 nm, hard", "Set (d): 354x354 nm, hard", "Set (e): 350x350 nm, medium"]
    labels = ["Sparse 1", "Sparse 2", "Compact 1", "Compact 2", "Dense 1"]

    # Data: dip removed vs. max resistance change
    additional_dataset = [
        {"max_res_change": [96.22063964205007, 44.99158976805173, 25.279942849287096, 12.360495955953477, 5.968368957966308], "dip_removed": [0, 0, 0, 0, 0]},
        {"max_res_change": [76.9063451288274, 39.94898308378969, 16.76463788989173, 7.11048591268908, 3.934905378623821], "dip_removed": [3.856360673904419, 0, 0, 0, 0]},
        {"max_res_change": [7.577167605507995, 2.899798807851095, 2.333199200214886, 2.00698761012863, 0.5836589957764815, 0.4579783734475429], "dip_removed": [10.307, 7.097, 65.764, 30.316, 218.010, 146.505]},
        {"max_res_change": [7.604564690048288, 2.6164501359842163, 2.2059728006996515, 1.37511, 0.6087130768012594, 0.41831299658590115], "dip_removed": [7.063, 65.427, 26.532, 52.200, 133.327, 143.178]},
        {"max_res_change": [20.241740738032178, 14.231358388694026, 7.15015084526569, 3.849712062876298], "dip_removed": [0, 0, 0, 0]},
    ]
    
    # Create 2-row subplot
    ##fig1, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12.8, 9.81), sharey=False)
    fig1, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12.8, 8.70), sharey=False)

    # === First subplot: Dip removed vs. Voltage ===
    for i, data in enumerate(datasets):
        ax1.scatter(data["voltages"], data["dip_removed"], s=80, color=colours[i], label=labels[i])

    ax1.set_xlabel("Voltage [mV]", fontsize=33)
    ax1.set_ylabel("Dip removed [s]", fontsize=33, labelpad=36.3)
    ax1.set_xlim(-35, 1085)
    ax1.set_ylim(-10, 250)
    ax1.tick_params(axis='both', labelsize=30)
    ax1.legend(fontsize=26)
    ax1.grid()

    # === Second subplot: Dip removed vs. Max resistance change ===
    for i, data in enumerate(additional_dataset):
        ax2.scatter(data["max_res_change"], data["dip_removed"], s=80, color=colours[i], label=labels[i])

    ax2.set_xlabel("Max resistance change in plot [%]", fontsize=33, labelpad=18)
    ax2.set_ylabel("Dip removed [s]", fontsize=33, labelpad=36.3)
    ax2.set_xlim(-3.125, 103.125)
    ax2.set_ylim(-10, 250)
    ax2.tick_params(axis='both', labelsize=30)
    ax2.legend(fontsize=26)
    ax2.grid()

    # Final layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.32)  # Space between subplots

    # Save shits?
    if savepath != '':
        fig1.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show shits.
    plt.show()


def increase_in_percent_per_second_exponential(
    lower_voltage,
    higher_voltage,
    initial_resistance,
    alpha_0,
    gamma,
    ):
    ''' Calculate how many extra "percents per second" you get from
        cranking the manipulation voltage from lower_voltage
        to higher_voltage.
    '''
    
    def exponential_voltage_dependency( V, alpha_0, gamma ):
        return alpha_0 * ((np.e)**(gamma * V))
    
    # Get (linear) resistance manipulation rates.
    old_resistance_per_second = exponential_voltage_dependency( lower_voltage, alpha_0, gamma )
    new_resistance_per_second = exponential_voltage_dependency( higher_voltage, alpha_0, gamma )
    
    # In 1 second, I get 100 % more resistance, if old_resistance_per_second is initial_resistance.
    old_resistance_per_second_percent = (old_resistance_per_second / initial_resistance) * 100
    new_resistance_per_second_percent = (new_resistance_per_second / initial_resistance) * 100
    
    # Return the increase: "I get these many more percents per second"
    return new_resistance_per_second_percent / old_resistance_per_second_percent

def plot_dielectric_breakdown_data(
    folder_path,
    number_of_junction_sizes_probed,
    plot_broken_data = False,
    resistance_limit_for_shorted_junction_ohm = 260,
    area_list = [],
    max_voltage_for_defining_R_in_mV = 50,
    savepath = ''
    ):
    ''' Reads all .txt files in folder_path, extracts Voltage and Current data,
        and plots them on the same figure.
        
        Parameters
        ----------
        folder_path : str
            Path to folder containing .txt data files.
        plot_broken_data : boolean
            If true, plot the entire data line, including datapoints
            after which the junction broke.
        savepath : str, optional
            Path to save the plot. If just an empty string '',
            then the plot will not be saved.
    '''
    
    # Set up a colour table given how many junction sizes there were.
    # Define start and end hex colors
    start_hex   = "#EE1C1C"
    end_hex     = "#C4EE1C"
    
    # Function to convert hex to RGB (0-1 range)
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    
    # Function to convert RGB (0-1) to hex
    def rgb_to_hex(rgb):
        return "#{:02X}{:02X}{:02X}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    
    # Convert start and end to HLS
    start_rgb = hex_to_rgb(start_hex)
    end_rgb   = hex_to_rgb(end_hex)

    start_hls = colorsys.rgb_to_hls(*start_rgb)
    end_hls   = colorsys.rgb_to_hls(*end_rgb)
    
    # Generate 10 evenly spaced HSL-adjusted colors
    colours = []
    for i in range(number_of_junction_sizes_probed):
        t = i / (number_of_junction_sizes_probed-1)  # Interpolation factor
        h = start_hls[0] + (end_hls[0] - start_hls[0]) * t
        l = start_hls[1] + (end_hls[1] - start_hls[1]) * t
        s = start_hls[2] + (end_hls[2] - start_hls[2]) * t
        rgb = colorsys.hls_to_rgb(h, l, s)
        colours.append(rgb_to_hex(rgb))
    
    ## Colours is now a list where junction size colours can be chosen from,
    ## in the big plot.
    
    # Create data table, that will hold the averages of the breakdown data.
    ##list_of_violins = [[]] * number_of_junction_sizes_probed  ## Apparently this is classic Python bullshit :P
    list_of_violins = [[] for _ in range(number_of_junction_sizes_probed)]
    
    def trim_after_max_voltage(voltage_mV, current_uA):
        voltage_mV = np.array(voltage_mV)
        current_uA = np.array(current_uA)
        
        # Find index of the maximum voltage_mV
        max_index = np.argmax(voltage_mV)

        # Slice up to and including max_index
        return voltage_mV[:max_index+1], current_uA[:max_index+1]
    
    # Create figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.7 * 2, 11))
    
    # Go through all .txt files in the folder.
    colour_counter = 0
    number_of_traces_plotted = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Expecting exactly two rows: voltage_mV and current_uA.
            voltage_mV_line = lines[0].strip().split(",")[1:]  # Skip header text.
            current_uA_line = lines[1].strip().split(",")[1:]  # Skip header text.

            # Convert to floats.
            voltage_mV = np.array(list(map(float, voltage_mV_line)))
            current_uA = np.array(list(map(float, current_uA_line)))
            
            # Plot data after the junction is broken?
            if not plot_broken_data:
                ## No.
                voltage_mV, current_uA = trim_after_max_voltage(voltage_mV, current_uA)
            
            # Shorted junction?
            max_voltage_mV_idx = np.argmax(voltage_mV)
            end_resistance = (voltage_mV[max_voltage_mV_idx] / 1000) / (current_uA[max_voltage_mV_idx] / 1e6)
            if not(end_resistance < resistance_limit_for_shorted_junction_ohm):
            
                ## Insert stuffs into plot!
                
                # What label should be used?
                if (area_list == []):
                    ##label_tag = filename
                    raise ValueError("Halted! The user has to define the junction areas. A simple side-length will suffice, like \"600 nm\".")
                elif (number_of_traces_plotted >= len(area_list)):
                    # Then, stop plotting.
                    label_tag = None
                else:
                    # In this case, use the user-provided legend list.
                    label_tag = area_list[number_of_traces_plotted]
                
                # Plot shits.
                ax1.plot(current_uA, voltage_mV, label=label_tag, color = colours[colour_counter], alpha=0.6)
                
                ## At this point, we can take statistics.
                
                # 1. Find the dU/dI rate to get R.
                idx = np.argmax(voltage_mV > max_voltage_for_defining_R_in_mV)
                if voltage_mV[idx] > max_voltage_for_defining_R_in_mV:
                    linear_domain_mV = voltage_mV[:idx]
                    linear_domain_uA = current_uA[:idx]
                else:
                    linear_domain_mV = voltage_mV
                    linear_domain_uA = current_uA
                slope, intercept, r_value, p_value, std_err = linregress(linear_domain_uA/1e6, linear_domain_mV/1e3)
                # ... the slope is R.
                trace_resistance = slope # [Ω]
                
                # 2. (R·A) goes on the X axis.
                ## TODO: handle cases when the user has not provided areas.
                ## Convert sidelength to area.
                side_length_m = int(area_list[colour_counter].replace(' nm', '')) * 1e-9
                area_m2 = side_length_m ** 2
                resistivity_length = trace_resistance * area_m2
                
                # 3. Collect datapoint.
                ## Note here that the colour counter can be used to keep track of which set this datapoint belongs to.
                (list_of_violins[colour_counter]).append( (resistivity_length , (voltage_mV[max_voltage_mV_idx] / 1000)) )
                
                # Increase counters.
                number_of_traces_plotted += 1
                
            # Whether or not we plotted the curve, keep track of counters.
            colour_counter += 1
            colour_counter %= number_of_junction_sizes_probed
    
    ## We can now treat list_of_violins.
    
    # Prepare data for the violin plot.
    mean_res_lengths = []  # Mean resistivity-length for each group.
    x_stds = []            # Standard deviation of resistivity-length for each group.
    y_data = []            # Voltage data for violin.
    y_means = []           # Mean voltages.
    y_stds = []            # Standard deviation for voltages.
    
    for group in list_of_violins:
        if len(group) == 0:
            continue  # Skip empty sets!

        # Group is a list of tuples: (resistivity_length, voltage)
        resistivities = [item[0] for item in group]
        voltages      = [item[1] for item in group]
        
        mean_res = np.mean(resistivities)
        std_res  = np.std(resistivities)

        mean_res_lengths.append(mean_res)
        x_stds.append(std_res)

        y_data.append(voltages)
        y_means.append(np.mean(voltages))
        y_stds.append(np.std(voltages))
    
    # Plot violin plots at mean resistivity positions
    ##parts = ax2.violinplot(y_data, positions=mean_res_lengths, showmeans=False, showextrema=False) # This is very messy. Don't.
    ## # Colour the things to match ax1.
    ## for pc, col in zip(parts['bodies'], colours):
    ##     pc.set_facecolor(col)
    ##     pc.set_alpha(0.4)
    
    # Overlay mean and error bars.
    ## for item in range(len(mean_res_lengths)):
    ##     ## Note the 1e3, to divide the y-axis into millivolts.
    ##     ax2.errorbar(mean_res_lengths[item] * 1e9, y_means[item] * 1e3, yerr=y_stds[item] * 1e3, color=colours[item], fmt='o', capsize=5, markersize=8)
    for item in range(len(mean_res_lengths)):
        ax2.errorbar(
            mean_res_lengths[item] * 1e9, # X value
            y_means[item] * 1e3,          # Y value
            xerr  = x_stds[item] * 1e9,   # Horizontal error bar
            yerr  = y_stds[item] * 1e3,   # Vertical error bar
            color = colours[item],
            fmt='o',
            capsize=5,
            markersize=8,
            label=area_list[item]
        )
    
    ax2.grid()
    ax2.tick_params(axis='both', labelsize=23)
    ##ax2.set_xlabel("Mean resistivity-length [nΩ·m²]", fontsize=33)
    ax2.set_xlabel(r'$\overline{\mathrm{R} \!\cdot\! \mathrm{A}}$ [n$\Omega$·m$^2$]', fontsize=33)
    ax2.set_ylabel("Breakdown voltage [mV]", fontsize=33)
    ax2.set_ylim(-65, 1400) ## TODO fix proper limits.
    ax2.legend(fontsize=26, loc='lower right')
    ax2.set_xlim(-0.06, 1.26)
    
    # Formatting for the regular breakdown voltage plot.
    ax1.grid()
    ax1.tick_params(axis='both', labelsize=23)
    ax1.set_xlabel("Current [µA]", fontsize=33)
    ax1.set_ylabel("Voltage [mV]", fontsize=33)
    #ax1.set_title("Dielectric breakdown data")
    ax1.legend(fontsize=26)
    ax1.set_ylim(-65, 1400) ## TODO fix proper limits.
    
    plt.tight_layout()
    if not (savepath == ''):
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
        print("Figure saved to: " + str(savepath))
    plt.show()

def plot_ln2_data(
    file_path,
    experimentally_found_G0 = (0.8824 + 0.8799 + 0.8751)/3, ## This number was found through Maurizio Toselli's 2024 experiments.
    experimentally_found_T0 = (783.1 + 792.9 + 762.4)/3,    ## This number was found through Maurizio Toselli's 2024 experiments.
    normalise_resistances = 0,
    savepath = '',
    plot = True,
    plot_no_junction_resistance_under_ohm = 260,
    error_factor_threshold = 0.005
    ):
    ''' Plot the resistance manipulation and relaxation from electrical
        annealing performed onto a junction bathing in liquid nitrogen.
        Any datapoints where the error bar is absurd, that is, too large
        relative to the resistance, will be filtered out.
        
        plot_no_junction_resistance_under_ohm: threshold for where you
        know the junction to be shorted. Experimentally, I have found
        that 260 Ω is a good indicator.
    '''
    # Storage
    resistances = []   # in kΩ
    errors = []        # std_err of slope, in kΩ
    times_min = []     # time in minutes
    temperatures = []  # in K

    # Read and split into blocks by blank lines (works with CRLF too)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = [b.strip() for b in re.split(r'\r?\n\s*\r?\n', content) if b.strip()]

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        I_vals = None
        V_vals = None
        t_val = None
        T_val = None
        
        # Parse!
        for ln in lines:
            parts = ln.split(';')
            key = parts[0].strip()
            vals = parts[1:]

            # Robust identification by the header text
            if 'I [' in key and 'nA' in key:
                try:
                    I_vals = np.array([float(x) for x in vals], dtype=float)
                except ValueError:
                    I_vals = None
            elif 'V [' in key and 'uV' in key:
                try:
                    V_vals = np.array([float(x) for x in vals], dtype=float)
                except ValueError:
                    V_vals = None
            elif 'Time' in key:
                try:
                    t_val = float(vals[0])
                except (IndexError, ValueError):
                    t_val = None
            elif 'Temperature' in key:
                try:
                    T_val = float(vals[0])
                except (IndexError, ValueError):
                    T_val = None

        # Only attempt fit if we have both arrays and they match in length
        if I_vals is None or V_vals is None:
            # skip block if data missing
            continue
        if I_vals.size != V_vals.size:
            # skip if mismatched lengths
            continue
        
        # Linear regression V [µV] vs I [nA] → slope in [µV/nA] == kΩ
        try:
            slope, intercept, r_value, p_value, std_err = linregress(I_vals, V_vals)
        except Exception:
            slope = np.nan
            std_err = np.nan

        R_kOhm = slope         # Slope is already in kΩ
        err_kOhm = std_err     # std_err same units → kΩ
        
        # Filter: remove datapoints with absurd error values
        if np.isnan(R_kOhm) or np.isnan(err_kOhm):
            continue
        if (abs(err_kOhm) > abs(R_kOhm) * error_factor_threshold) or (abs(R_kOhm) < plot_no_junction_resistance_under_ohm/1000):
            # Skip this point
            continue
        
        # Append!
        resistances.append(R_kOhm)
        errors.append(err_kOhm)

        # Time in minutes (if missing, store NaN)
        times_min.append((t_val / 60.0) if (t_val is not None) else np.nan)
        temperatures.append(T_val if (T_val is not None) else np.nan)

    # Convert to numpy arrays
    resistances = np.array(resistances, dtype=float)
    errors = np.array(errors, dtype=float)
    times_min = np.array(times_min, dtype=float)
    temperatures = np.array(temperatures, dtype=float)

    # Store the initial resistance (before any normalization)
    res_initial = resistances[0] if resistances.size > 0 else np.nan
    
    # There will be another trace that is normalised
    # from cryogenic to room-temperature.
    ## The idea from this comes from Maurizio Toselli:
    ## The junction is a MIM diode whose resistance vs. temperature
    ## can be mapped as it cools. That way, it can be mapped to
    ## Simmon's model, and through the average of three fits (or more)
    ## one knows the temperature translation for that junction.
    ## Specifically, G_0 and T_0 for the equation.
    def simmons_model( T, T_0, G_0 ):
        ''' Convert the observed resistance for some temperature,
            into a room-temperature equivalent resistance.
            
            G(T) = G₀ · (1 + ( T / T₀ )^2)
        '''
        return G_0 * ( 1 + (T/T_0)**2 )
    
    
    
    

    # Normalise if requested (divide resistances and errors by res_initial)
    if normalise_resistances == 1 and not np.isnan(res_initial) and res_initial != 0:
        resistances = ((resistances / res_initial) - 1) * 100
        errors = (errors / res_initial) * 100
        ylabel_res = "Resistance increase [%]"
    elif normalise_resistances == 0:
        ylabel_res = "Resistance [kΩ]"
    else:
        raise ValueError("Unknown argument given for normalise_resistances: "+str(normalise_resistances))
    

    # Plotting
    if plot:
        fig, ax1 = plt.subplots(figsize=(12.5, 10))

        # Temperature on left y-axis (red)
        ax1.plot(times_min, temperatures, '-', linewidth=2, color="#1CEE70", label="Temperature [K]")
        ax1.set_xlabel("Time [min]", fontsize=33)
        ax1.set_ylabel("Temperature [K]", fontsize=33)##, color="#EE1C1C")
        ax1.set_ylim(bottom=-14.77625)
        ##ax1.tick_params(axis='y', labelcolor="#EE1C1C")

        # Resistance on right y-axis (blue)
        ax2 = ax1.twinx()
        ax2.errorbar(times_min, resistances, yerr=errors,
                     fmt='o-', markersize=6, ecolor="#EE1C1C", color="#EE1C1C",
                     capsize=3, label=ylabel_res)
        ax2.set_ylabel(ylabel_res, fontsize=33)##, color="#1C70EE")
        ##ax2.tick_params(axis='y', labelcolor="#1C70EE")
        
        # Bump up ticks. Add grid.
        ax1.tick_params(axis='both', labelsize=23)
        ax2.tick_params(axis='both', labelsize=23)
        
        # Combined legend.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Grid and legend.
        plt.grid()
        ## Important: setting ax1.legend here,
        ## will make the grid very visible through the legend box.
        ## But using ax2.legend solves this problem, for now.
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=26)##, frameon=True, facecolor='white', framealpha=0.8)
        
        plt.tight_layout()
        if not (savepath == ''):
            fig.savefig(savepath, dpi=164, bbox_inches='tight')
            print("Figure saved to: " + str(savepath))
        plt.show()

    # Return (time[min], resistance[kΩ or normalised], error[same units], temperature[K], res_initial[kΩ])
    return times_min, resistances, errors, temperatures, res_initial
    