#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

from random import randint
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys
import os
import re
from scipy.stats import moment

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
    ## Koch 2007 equation regarding transmon f_01, precision to second order.
    ## https://doi.org/10.1103/PhysRevB.77.180502
    second_order_correction_factor = -(E_C / 2) * (E_C / (8*E_J))
    f_01 = (np.sqrt(8 * E_J * E_C) -E_C + second_order_correction_factor)/h
    
    # Return value!
    return f_01

def calculate_RT_resistance_from_target_f01(
    target_f_01,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    R_N_initial_guess = 15000,
    acceptable_frequency_offset = 250,
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
    expected_resistance_creep = 0,
    verbose = True
    ):
    ''' Given a target frequency, calculate what resistance should be
        manipulated to, including resistance creep effects.
        
        The units of expected_aging and expected_resistance_creep are linear.
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
            assert increase >= 1.0, "Error! The resistance can only go up. Unable to tune junction to the target frequency. Needed tuning: "+str((increase-1)*100)+" %"
            print(f"The target room-temperature resistance is: {(room_temperature_resistance_to_hit):.3f} [Ω], which corresponds to {((increase-1)*100):.3f} %")
        else:
            print(f"The target room-temperature resistance is: {(room_temperature_resistance_to_hit):.3f} [Ω]")
    
    # Does the sample have some aging left to do?
    ## TODO
    if expected_aging > 0:
        raise NotImplementedError("Not finished.")
    
    # TODO the creep is not a fixed offset like this, it is time dependent.
    ## Well, to be picky, we know that the end number is pretty much
    ## a fixed resistance offset.
    if expected_resistance_creep > 0:
        print(f"Expecting {((expected_resistance_creep-1)*100):.3f} % worth of resistance creep.")
        room_temperature_resistance_to_hit = room_temperature_resistance_to_hit / expected_resistance_creep
        if original_resistance_of_junction > 0:
            increase = room_temperature_resistance_to_hit / original_resistance_of_junction
            print(f"Excluding resistance creep, expect to hit: {(room_temperature_resistance_to_hit):.3f} [Ω], which is {((increase-1)*100):.3f} %")
        else:
            print(f"Excluding resistance creep, expect to hit: {(room_temperature_resistance_to_hit):.3f} [Ω]")
    elif expected_resistance_creep < 0:
        raise ValueError("Error! The resistance creep is expected to be a positive number; the resistance is expected to increase post-manipulation.")
    
    return room_temperature_resistance_to_hit

def plot_josephson_junction_resistance_manipulation_and_creep(
    filepath,
    normalise_resistances = 2,
    normalise_time_to_creep_effect = True,
    attempt_to_color_plots_from_file_name = False,
    plot_no_junction_resistance_under_ohm = 0
    ):
    ''' Plot the data from the resistance manipulation and ensueing
        resistance creep.
        
        normalise_resistances will:
            0:  do not normalise resistances,
            1:  set the first measured resistance value as the initial
                resistance of the device; all subsequent values
                will be reported as a percentage of this initial value.
            2:  same as 1, but the resistance creep's datapoint_0
                will be the resistance that is normalised to.
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
        resistance_at_creep_start = 0
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
                                    # Set zero point, i.e., start of creep.
                                    zero_points[zz] = time_offset_due_to_appended_data
                                    zero_point_was_found = True
                                    
                                    # Log the resistance at the creep's start
                                    resistance_at_creep_start = current_resistance
                                    
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
            # Check whether the start of the creep was found.
            if resistance_at_creep_start > 0:
                resistances = (resistances / resistance_at_creep_start) - 1
                plt.ylabel("Resistance normalised to creep effect start [-]")
            else:
                # In that case, just take the last value and normalise to that.
                resistances = (resistances / resistances[-1]) - 1
        elif normalise_resistances == 1:
            resistances = (resistances / resistances[0]) - 1
            plt.ylabel("Resistance normalised to starting value [-]")
        else:
            plt.ylabel("Resistance [Ω]")
        
        # Normalise time axis?
        if normalise_time_to_creep_effect:
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
            plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colors(jj))
        
    plt.xlabel("Duration [s]")
    plt.title("Resistance vs. Time")
    plt.grid()
    plt.legend()
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
                Delta_mean_eV = list_of_doubles_of_Delta_eV_and_Delta_std_eV[ii][0],
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
    