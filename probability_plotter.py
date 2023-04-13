#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import json
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
 
def barplot(
    filepaths_to_plot,
    title = '',
    halt_on_distribution_data_missing = False,
    attempt_to_fix_string_input_argument = True
    ):
    ''' Plot some state distribution set(s) in a bar graph.
    '''
    
    # User syntax repair
    if attempt_to_fix_string_input_argument:
        if isinstance(filepaths_to_plot, str):
            filepaths_to_plot = os.path.abspath(filepaths_to_plot)
            filepaths_to_plot = [filepaths_to_plot]
    
    # For all entries!
    for item in filepaths_to_plot:
        
        # Skip this file? (Reset flag from below)
        skip = False
        
        # Get current path
        filepath_to_file_with_probability_data_to_plot = item
        
        # Ensure that the filepath received can be processed.
        if type(filepath_to_file_with_probability_data_to_plot) == list:
            filepath_to_file_with_probability_data_to_plot = "".join(filepath_to_file_with_probability_data_to_plot)
        
        # Establish how many states are covered, and what the probabilities are.
        with h5py.File(filepath_to_file_with_probability_data_to_plot, 'r') as h5f:
            log_dict_list = json.loads( h5f.attrs["log_dict_list"] )
            processed_data = h5f["processed_data"][()]
        
        # If there are too many arrays within arrays within arrays,
        # then unpack the values until satisfactory
        printable = processed_data
        new_arr = []
        done = False
        strike = False
        while (not done) and (not skip):
            
            # Reset the strike value. Strike will go high whenever there is any
            # value in the distribution data that is unnecesarily packed in a list
            # or numpy array.
            strike = False
            for jj in range(len(processed_data)):
                
                # Is the current value packed in some list?
                if (type(processed_data[jj]) == list) or (type(processed_data[jj]) == np.ndarray):
                    if len(processed_data[jj]) <= 1:
                        # Fix the issue if any.
                        new_arr.append( (processed_data[jj])[0] )
                        try:
                            # Check if we need another round. Strike if yes.
                            if len((processed_data[jj])[0]) > 0:
                                strike = True
                        except TypeError:
                            # Great, not a list!
                            pass
                    else:
                        error_msg = "Error! The file that was provided " +\
                            "to the distribution bar plotter had an "    +\
                            "unexpected shape on the distribution data." +\
                            " The distribution data was: "               +\
                            "\n"+str(printable)
                        if halt_on_distribution_data_missing:
                            raise ValueError( error_msg )
                        else:
                            print(error_msg)
                            print("\nNo barplot was made for this data, skipping file.")
                            skip = True
            
            if not skip:
                # Update the array and blank new_arr.
                processed_data = np.array(new_arr)
                new_arr = []
                # Did we loop through the entire list without any packed values?
                if (not strike):
                    done = True  # Then, we may stop here.
        del printable
        del new_arr
        del strike
        del done
        
        # Actual file for plotting or not?
        if not skip:
        
            # Ensure that the total probability exactly equals 100%
            runsum = 0.0
            for item in processed_data:
                runsum += item
            assert runsum == 1.0, "Halted! The total probability of the loaded file does not equal 100% - but "+str(runsum)
            
            # Get involved distributions
            data = {}
            for ii in range(len(log_dict_list)):
                data.update( {  (log_dict_list[ii]["name"]).replace('Probability for state ','')   : processed_data[ii] } )
            
            # Create and plot the dataset!
            states = list(data.keys())
            probabilities = list(data.values())
            fig = plt.figure(figsize = (5, 5))
            
            # Drawing lines.
            # TODO: Find highest 20%-chunk above the highest percentage. (Like, 80% if 68% is the highest percent in the data series)
            # TODO: Divide this 20% chunk value (in this csae, 80%) into 4 equal pieces.
            # TODO: Draw dashed lines at every interval (so, one line at 20%, one at 40%, 60%, 80%)
            plt.axhline(y=0.2, color="#bfbfbf", linestyle=":")
            plt.axhline(y=0.4, color="#bfbfbf", linestyle=":")
            plt.axhline(y=0.6, color="#bfbfbf", linestyle=":")
            plt.axhline(y=0.8, color="#bfbfbf", linestyle=":")
            plt.axhline(y=1.0, color="#bfbfbf", linestyle=":")
            
            plt.bar(states, probabilities, color ='#648fff', width = 0.2)
            plt.xticks(rotation = 45)
            
            # Labels and title
            ##plt.xlabel("States")
            plt.ylabel("Probabilities")
            if title != '':
                plt.title( title )
            else:
                plt.title( filepath_to_file_with_probability_data_to_plot )
            
            # Plot!
            plt.show()

def plot_confusion_matrix(
    filepaths_to_plot,
    title = '',
    attempt_to_fix_string_input_argument = True,
    figure_size_tuple = (15,12),
    maximum_state_to_attempt_to_plot = 10,
    ):
    ''' For some readout space datafile, extract the probabilities for reading
        out some state |N⟩ given that state |M⟩ was prepared.
        
        Put the confusion matrix data in a pretty plot.
        
        figure_size_tuple is a tuple, somehow defining the dimensions of the
        output plot in yankeedoodleland units.
        
        maximum_state_to_attempt_to_plot = 10 denotes that the function
        will attempt to dig out states |0⟩ → |10⟩ from the data.
    '''
    
    # User syntax repair
    if attempt_to_fix_string_input_argument:
        if isinstance(filepaths_to_plot, str):
            filepaths_to_plot = os.path.abspath(filepaths_to_plot)
            filepaths_to_plot = [filepaths_to_plot]
    
    # For all entries!
    for item in filepaths_to_plot:
        
        # Get current path
        filepath_to_file_with_probability_data_to_plot = item
        
        # Ensure that the filepath received can be processed.
        if type(filepath_to_file_with_probability_data_to_plot) == list:
            filepath_to_file_with_probability_data_to_plot = "".join(filepath_to_file_with_probability_data_to_plot)
        
        # As we find more and more states, we'll zero-pad the final canvas.
        # The padder however assumes a two-dimensional object.
        canvas = np.array([np.array([0.0, 0.0]), np.array([0.0, 0.0])])
        
        # Get confusion matrix data.
        with h5py.File(filepath_to_file_with_probability_data_to_plot, 'r') as h5f:
            
            # There may or may not be probability data for the current file,
            # for the sought-for state.
            for prepared_state in range(maximum_state_to_attempt_to_plot+1):
                for measured_state in range(maximum_state_to_attempt_to_plot+1):
                    
                    # Attempt to get data.
                    entry_to_look_up = "prob" + \
                        "_meas"+str(measured_state)+\
                        "_prep"+str(prepared_state)
                    try:
                        # Get data if possible.
                        value_to_add = h5f.attrs[entry_to_look_up]
                        
                        # Are we attempting to fill out data in the canvas
                        # outside of its current matrix edges?
                        canvas_shape = canvas.shape
                        
                        # Note: account for state |0>, subtract with 1.
                        if (prepared_state > canvas_shape[0] -1):
                            # We are. We'll pad the canvas if there is
                            # any information to gather here.
                            canvas = np.pad( \
                                canvas, \
                                (0, (prepared_state - len(canvas[0]))+1 ) \
                            )
                        elif (measured_state > canvas_shape[1] -1):
                            # We are. We'll pad the canvas if there is
                            # any information to gather here.
                            canvas = np.pad( \
                                canvas, \
                                (0, (measured_state - len(canvas))+1 ) \
                            )
                        
                        # Insert data into canvas.
                        canvas[prepared_state][measured_state] = value_to_add
                        
                        # Clean up
                        del value_to_add
                    
                    except KeyError:
                        # The sought-for confusion matrix entry does not exist.
                        pass
            
        # At this point, we've acquired the confusion matrix.
        # Time to plot.
        
        # Make X and Y axes.
        x_axis = np.arange(0.0, canvas.shape[1], 1)
        y_axis = np.arange(0.0, canvas.shape[0], 1)
        x_axis_str = []
        y_axis_str = []
        for item in x_axis:
            x_axis_str.append('Measured |'+str(int(item))+'⟩')
        for item in y_axis:
            y_axis_str.append('Prepared |'+str(int(item))+'⟩')
        
        # Set figure size
        plt.figure(figsize = figure_size_tuple)
        
        # Better to plot the x axis along the top.
        plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        
        # Add numeric value to each box.
        for (j,i),label in np.ndenumerate(canvas):
            if float(label) <= 0.45:
                plt.text( i, j, label, ha='center', va='center', weight='bold', color = 'white', fontsize=35)
            else:
                plt.text( i, j, label, ha='center', va='center', weight='bold', fontsize=35)
        
        # Make pixel palette.
        plt.imshow(canvas, interpolation='nearest')
        plt.xticks(x_axis, x_axis_str, fontsize = 21)
        plt.yticks(y_axis, y_axis_str, fontsize = 21)
        if title == '':
            plt.title("Confusion matrix", fontsize = 35, pad = 20)
        else:
            plt.title(title, fontsize = 35, pad = 20)
        
        # Show plot!
        plt.show()