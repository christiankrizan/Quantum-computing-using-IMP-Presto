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
from scipy.optimize import curve_fit
from scipy.stats import moment
from scipy.stats import linregress

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
            return "#F2F1Ed"
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



def plot_fourier_transform_of_resistance_creep(
    filepath,
    normalise_resistances = 0,
    normalise_time_to_creep_effect = False,
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
            
            ##plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colors(jj))
        
    ##plt.xlabel("Duration [s]")
    ##plt.title("Resistance vs. Time")
    ##plt.grid()
    ##plt.legend()
    ##plt.show()
    
def plot_josephson_junction_resistance_manipulation_and_creep(
    filepath,
    normalise_resistances = 2,
    normalise_time_to_creep_effect = True,
    attempt_to_color_plots_from_file_name = False,
    plot_no_junction_resistance_under_ohm = 0,
    colourise = False,
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
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=get_colourise(-2))
        #plt.figure(figsize=(10, 5), facecolor=get_colourise(-2))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        #plt.figure(figsize=(10, 5))
    
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
        '''if normalise_resistances == 3:
            # In that case, normalise to the lowest resistance found, that
            # is not an obvious short.
            resistances = (resistances / lowest_non_short_resistance_in_set) - 1
            y_label_text = "Resistance normalised to lowest resistance in the set [-]"
        el'''
        if normalise_resistances == 2:
            # Check whether the start of the creep was found.
            if resistance_at_creep_start > 0:
                resistances = (resistances / resistance_at_creep_start) - 1
                y_label_text = "Resistance normalised to creep effect start [-]"
            else:
                # In that case, just take the last value and normalise to that.
                resistances = (resistances / resistances[-1]) - 1
        elif normalise_resistances == 1:
            resistances = (resistances / resistances[0]) - 1
            y_label_text = "Resistance normalised to starting value [-]"
        else:
            y_label_text = "Resistance [Ω]"
        
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
                plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colors(jj))
        else:
            # Use patterned colour.
            plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=get_colourise((colourised_counter // 4) + ((colourised_counter % 4) + 1) / 10))
            colourised_counter += 1
        
    
    # Set axes' colour? Title colour? And so on.
    plt.grid()
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
    
    # Bump up the size of the ticks' numbers on the axes.
    ax.tick_params(axis='both', labelsize=13)
    
    assert colourise == True, "Halted! Code missing for setting colour to its default. Fix it." # TODO
    
    plt.xlabel("Duration [s]", color=get_colourise(-1), fontsize=16)
    plt.ylabel(y_label_text, color=get_colourise(-1), fontsize=16)
    plt.title("Resistance vs. Time", color=get_colourise(-1), fontsize=25)
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
    expected_resistance_creep = 1.0229,
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
                expected_resistance_creep = expected_resistance_creep,
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
    colourise = False,
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
    '''
    
    # Initially, let's define some functions for the fitting.
    def second_order_func(t, t_0, R_0, alpha, beta):
        return R_0 + (alpha * (t-t_0)) + (beta * (t-t_0)**2)
    
    def third_order_func(t, t_0, R_0, alpha, beta, delta):
        return R_0 + (alpha * (t-t_0)) + (beta * (t-t_0)**2) + (delta * (t-t_0)**3)
    
    def exponential_func(t, t_0, R_0, epsilon, gamma, tau):
        ##return R_0 + epsilon * (np.e)**((t-t_0) * gamma)
        return R_0 + epsilon * (1 - (np.e)**((t-t_0)/tau * -gamma))
    
    def power_func(t, t_0, R_0, A, B):
        return R_0 + A * ((t-t_0)**B)
    
    
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
                p0    = (t_0_guess, R_0_guess, alpha_guess, beta_guess)
            )
        elif fitter == 'third_order':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = third_order_func,
                xdata = time,
                ydata = resistances,
                p0    = (t_0_guess, R_0_guess, alpha_guess, beta_guess, delta_guess)
            )
        elif fitter == 'exponential':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = exponential_func,
                xdata = time,
                ydata = resistances,
                p0    = (t_0_guess, R_0_guess, epsilon_guess, gamma_guess, tau_guess)
            )
        elif fitter == 'power':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = power_func,
                xdata = time,
                ydata = resistances,
                p0    = (t_0_guess, R_0_guess, A_guess, B_guess)
            )
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Extract parameters.
        optimal_t_0    = optimal_vals[0]
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
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Get the fit errors.
        fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
        err_t_0 = fit_err[0]
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
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Get a fit curve!
        if fitter == 'second_order':
            fitted_curve = second_order_func(
                t     = times,
                t_0   = optimal_t_0,
                R_0   = optimal_R_0,
                alpha = optimal_alpha,
                beta  = optimal_beta
            )
        elif fitter == 'third_order':
            fitted_curve = third_order_func(
                t     = times,
                t_0   = optimal_t_0,
                R_0   = optimal_R_0,
                alpha = optimal_alpha,
                beta  = optimal_beta,
                delta = optimal_delta
            )
        elif fitter == 'exponential':
            fitted_curve = exponential_func(
                t       = times,
                t_0     = optimal_t_0,
                R_0     = optimal_R_0,
                epsilon = optimal_epsilon,
                gamma   = optimal_gamma,
                tau     = optimal_tau
            )
        elif fitter == 'power':
            fitted_curve = exponential_func(
                t   = times,
                t_0 = optimal_t_0,
                R_0 = optimal_R_0,
                A   = optimal_A,
                B   = optimal_B,
            )
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
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor=get_colourise(-2))
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=get_colourise(-2))
    else:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # Go through the files and add them to the plot.
    lowest_non_short_resistance_of_all = 1000000000
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
        
        # Set initial parameters.
        times = []
        resistances = []
        obvious_short = 100 # [Ω]  --  Define a resistance that defines "a short."
        lowest_non_short_resistance_in_set = 1000000000
        
        ## The new time format assumes that all data is set with reference
        ## to some initial time.
        time_0 = -1 # [s] -- Which would be 23:59:59 on December 31st 1969
        
        # Open file.
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
        
        # Normalise resistance axis?
        if normalise_resistances == 1:
            resistances = ((resistances / resistances[0]) - 1) * 100
            y_label_text = "Resistance increase [%]"
        else:
            y_label_text = "Resistance [Ω]"
        
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
        ## Select marker symbol.
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
            ## Legacy colouring?
            if   'thin'  in file_label.lower():
                colour_label = 'winter'
                
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
                
            elif 'thick' in file_label.lower():
                colour_label = 'autumn'
                
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
                
            else:
                colour_label = 'tab20'
            num_items_to_colour = len(filepath)
            colors = plt.cm.get_cmap(colour_label, num_items_to_colour)
            
            plt.figure(1) # Set figure 1 as active.
            plt.scatter(times, resistances, marker=marker_symbol, label=file_label, color=colors(jj))
        else:
            # Then follow the schema.
            plt.figure(1) # Set figure 1 as active.
            plt.plot(times, resistances, marker=marker_symbol, label=file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
        
        # Plot the fit curve.
        fit_label = ''
        for kk in range(len(fit_results[1])):
            fitted_values = (fit_results[1])[kk]
            fitted_errors = (fit_results[2])[kk]
            prefix = '?'
            if   kk == 0:
                prefix = 'R₀'
                fit_label += prefix+': '+(f"{fitted_values:.3f} ±{fitted_errors:.3f}")+'\n'
            elif kk == 1:
                prefix = 't₀'
                fit_label += prefix+': '+(f"{fitted_values:.3f} ±{fitted_errors:.3f}")+'\n'
            else:
                if fitter == 'second_order':
                    if   kk == 2:
                        prefix = 'α'
                    elif kk == 3:
                        prefix = 'β'
                elif fitter == 'third_order':
                    if   kk == 2:
                        prefix = 'α'
                    elif kk == 3:
                        prefix = 'β'
                    elif kk == 4:
                        prefix = 'δ'
                elif fitter == 'exponential':
                    if   kk == 2:
                        prefix = 'ε'
                    elif kk == 3:
                        prefix = 'γ'
                    elif kk == 4:
                        prefix = 'τ'
                else:
                    raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
                
                # Find a proper exponent of the number.
                exponent       = np.floor(np.log10(np.abs( fitted_values )))
                error_exponent = np.floor(np.log10(np.abs( fitted_errors )))
                fit_label += prefix+': '+(f"{(fitted_values * (10**(-exponent))):.3f}·10^{exponent} ±{(fitted_errors * (10**(-error_exponent))):.3f}·10^{error_exponent}")+'\n'
        if (not colourise):
            plt.figure(1) # Set figure 1 as active.
            plt.plot(times, fit_results[0], linestyle='--', label='Fit '+str(jj)+': '+fit_label, color=colors(jj))
        else:
            plt.figure(1) # Set figure 1 as active.
            plt.plot(times, fit_results[0], linestyle='--', label='Fit '+str(jj)+': '+fit_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
        
        # Get residuals plot?
        if fitter != 'none':
            ## fit_results[0]: the fit curve.
            ##    resistances: the actual numbers measured.
            ##       residual: actual_y - predicted_y
            residuals = resistances - fit_results[0]
            
            plt.figure(2) # Set figure 2 as active.
            plt.scatter(times, residuals, label='Residuals of fit '+str(jj+1))
    
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
    ax1.tick_params(axis='both', labelsize=13)
    ax2.tick_params(axis='both', labelsize=13)
    
    # Extend axes to include the origin?
    ## Do not extend the x-axis if trying to plot the UNIX time.
    if (np.all(times > 0)) and (normalise_time):
        ax1.set_xlim(xmin=0)
    #if np.all(lowest_non_short_resistance_of_all >= -5):
    #    ax1.set_ylim(ymin=-5)
    
    # Other figure formatting.
    if (not colourise):
        plt.figure(1) # Set figure 1 as active.
        plt.xlabel("Duration [s]", fontsize=16)
        plt.ylabel(y_label_text, fontsize=16)
        plt.title("Resistance vs. Time", fontsize=25)
        
        plt.figure(2) # Set figure 2 as active.
        plt.xlabel("Duration [s]", fontsize=16)
        plt.ylabel(y_label_text, fontsize=16)
        if fitter != 'none':
            if fitter == 'second_order':
                plt.title("Residuals, 2nd-ord-polyn.", fontsize=25)
            elif fitter == 'third_order':
                plt.title("Residuals, 3rd-ord-polyn.", fontsize=25)
            elif fitter == 'exponential':
                plt.title("Residuals, exponential func.", fontsize=25)
            elif fitter == 'power':
                plt.title("Residuals, power-law", fontsize=25)
    else:
        plt.figure(1) # Set figure 1 as active.
        plt.xlabel("Duration [s]", color=get_colourise(-1), fontsize=16)
        plt.ylabel(y_label_text, color=get_colourise(-1), fontsize=16)
        plt.title("Resistance vs. Time", color=get_colourise(-1), fontsize=25)
    
    # Show shits.
    for ll in range(2):
        plt.figure(ll+1)
        if ll == 1:
            plt.legend(fontsize=16)
        else:
            plt.legend()
    
    plt.show()
    
    

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

def plot_active_vs_total_resistance_gain(
    title_voltage_V,
    title_junction_size_nm,
    outlier_threshold_in_std_devs = 2.0,
    consider_outliers = True,
    plot_ideal_curve = False,
    colourise = False
    ):
    ''' Given some set of data, plot the active gain on the X axis in percent,
        and the total gain on the Y axis in percent relative to the
        initial resistance point. Also, make a fit.
        
        Datapoints that fall outside of outlier_threshold_in_std_devs are
        considered outliers, this limit was set to 2 standard deviations
        from the mean.
    '''
    
    # Counter to keep track of colourised patterns.
    colourise_counter = 0
    
    # TODO: acquire data.
    active_gain_percent = [2.45,  0.71,  10.05,  1.79,  5.02, 0.61, 3.52, 0.77, 1.01,  8.26, 2.39, 4.22, 4.15,  9.02,  7.71, 2.58, 1.77, 16.51, 5.13, 6.07, 11.03, 7.01, 0.28]
    total_gain_percent  = [3.587, 2.180, 11.969, 2.999, 7.30, 2.76, 6.76, 2.17, 3.06, 10.55, 4.61, 5.70, 6.10, 10.85, 11.78, 6.45, 5.91, 19.07, 6.61, 8.23, 13.13, 8.93, 1.77]
    
    # Sort lists together based on the active gain list.
    sorted_active, sorted_total = zip(*sorted(zip(active_gain_percent, total_gain_percent)))
    
    # Convert to numpy arrays!
    sorted_active = np.array(sorted_active, dtype=np.float64)
    sorted_total  = np.array(sorted_total, dtype=np.float64)
    
    # Report!
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
    if colourise:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=get_colourise(-2))
        #plt.figure(figsize=(10, 5), facecolor=get_colourise(-2))
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
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
    if colourise:
        plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color=get_colourise(colourise_counter), label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
        colourise_counter += 1
    else:
        plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color="#34D2D6", label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
    
    if consider_outliers:
        # Do the same with the filtered data.
        ## Remember to extend to 0.
        filtered_fitted_curve, filtered_k, filtered_m, filtered_err_k, filtered_err_m = linear_fitter(active_filtered, total_filtered)
        if colourise:
            plt.plot(np.insert(active_filtered, 0, 0), np.insert(filtered_fitted_curve, 0, filtered_m), color=get_colourise(colourise_counter), label=f"Trimmed fit: {filtered_k:.2f} · x + {filtered_m:.2f} [%]\nSlope error: ±{filtered_err_k:.2f}\nOffset error: ±{filtered_err_m:.2f}")
            colourise_counter += 1
        else:
            plt.plot(active_filtered, filtered_fitted_curve, color="#81D634", label=f"Trimmed fit: {filtered_k:.2f} · x + {filtered_m:.2f} [%]\nSlope error: ±{filtered_err_k:.2f}\nOffset error: ±{filtered_err_m:.2f}")
    
    # Plot an ideal curve?
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
    if (not colourise):
        if consider_outliers:
            plt.scatter(sorted_active, sorted_total, color="#8934D6", label=str(outlier_threshold_in_std_devs)+"σ outliers")
            ## This is a weird way to do this plotting, but it ensures that
            ## only the outliers are highlighted in the plot legend.
            ## Even though this dataset is the original! → Result = Good legend!
            plt.scatter(active_filtered, total_filtered, color="#D63834")
        else:
            plt.scatter(sorted_active, sorted_total, color="#D63834")
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
    ax.tick_params(axis='both', labelsize=13)
    
    # Extend axes to include the origin?
    if np.all(sorted_active >= 0):
        ax.set_xlim(xmin=0)
    if np.all(sorted_total >= 0):
        ax.set_ylim(ymin=0)
    
    # Fancy colours?
    if (not colourise):
        plt.xlabel("Active manipulation [%]", fontsize=16)
        plt.ylabel("Total manipulation [%]", fontsize=16)
        plt.title(f"Active vs. total manipulation\n30 minutes after stopping\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", fontsize=25)
    else:
        plt.xlabel("Active manipulation [%]", color=get_colourise(-1), fontsize=16)
        plt.ylabel("Total manipulation [%]",  color=get_colourise(-1), fontsize=16)
        plt.title(f"Active vs. total manipulation\n30 minutes after stopping\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", color=get_colourise(-1), fontsize=25)
    
    # Show shits.
    plt.legend()
    plt.show()