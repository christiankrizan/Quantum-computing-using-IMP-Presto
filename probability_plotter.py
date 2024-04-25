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
            assert ((runsum >= 0.9999999999) and (runsum <= 1.0000000001)), "Halted! The total probability of the loaded file does not equal 100% - but "+str(runsum)
            
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

def plot_diff_1q_2q_confusion_matrix(
    qb_A_confusion_matrix,
    qb_B_confusion_matrix,
    qb_AB_confusion_matrix,
    title = '',
    prepared_states = ['00','01','02','10','11','12','20','21','22'],
    figure_size_tuple = (15,14),
    round_data_to_decimal_places = 0,
    export_filepath = '',
    plot_output = False,
    ):
    ''' Take two one-qubit confusion matrices.
        From this data, construct an expected two-qubit confusion matrix.
        Then, given a two-qubit confusion matrix, compare the difference.
        Finally, plot.
        
        It is assumed that:
        - Confusion matrix ROWS    signify the prepared state.
        - Confusion matrix COLUMNS signify the measured state.
        
        In the returned difference matrix, a POSITIVE difference means that
        the real-life measured value, was bigger than the calculated value.
        Whereas a NEGATIVE difference, means that the real-life measured value,
        was smaller than the calculated value.
    '''
    
    # Type casting.
    if type(qb_A_confusion_matrix) != np.ndarray:
        qb_A_confusion_matrix = np.array(qb_A_confusion_matrix)
    if type(qb_B_confusion_matrix) != np.ndarray:
        qb_B_confusion_matrix = np.array(qb_B_confusion_matrix)
    if type(qb_AB_confusion_matrix) != np.ndarray:
        qb_AB_confusion_matrix = np.array(qb_AB_confusion_matrix)
    
    # Assert square dimensions.
    for mat in [qb_A_confusion_matrix, qb_B_confusion_matrix, qb_AB_confusion_matrix]:
        if mat.shape[0] != mat.shape[1]:
            raise TypeError("The provided matrices are not square, and are thus not valid confusion matrices.")
    
    # Assert that the single-qubit confusion matrices
    # feature identical dimensions.
    if (qb_A_confusion_matrix.shape[0] != qb_B_confusion_matrix.shape[0]) or (qb_A_confusion_matrix.shape[1] != qb_B_confusion_matrix.shape[1]):
        raise TypeError( \
            "The provided single-qubit matrices do not have the same"+\
            "dimensions ("+\
            str(qb_A_confusion_matrix.shape[0])+"x"+\
            str(qb_A_confusion_matrix.shape[1])+\
            " vs. "+\
            str(qb_B_confusion_matrix.shape[0])+"x"+\
            str(qb_B_confusion_matrix.shape[0])+")")
    if qb_B_confusion_matrix.shape[0] != qb_B_confusion_matrix.shape[1]:
        raise TypeError("The provided matrices are not square, and are thus not valid confusion matrices.")
    
    # Assert assert that the AB confusion matrix can be constructed from
    # the single-qubit confusion matrix, dimension-wise.
    if (qb_A_confusion_matrix.shape[0])**2 != qb_AB_confusion_matrix.shape[0]:
        raise TypeError("The provided single-qubit matrices would not "+\
        "combine into a two-qubit matrix with the same dimensions as the "+\
        "two-qubit matrix.")
    
    # Construct two-qubit confusion matrix.
    ## ROWS: Prepared state.
    ## COLS: Measured state.
    from time import sleep as snooz # TODO
    constructed_matrix = np.zeros_like(qb_AB_confusion_matrix)
    for row_A in range(qb_A_confusion_matrix.shape[0]):
        for row_B in range(qb_B_confusion_matrix.shape[0]):
            for col_A in range(qb_A_confusion_matrix.shape[1]):
                for col_B in range(qb_B_confusion_matrix.shape[1]):        
                    index_row = (qb_A_confusion_matrix.shape[0])*row_A + row_B
                    index_col = (qb_A_confusion_matrix.shape[1])*col_A + col_B
                    val = qb_A_confusion_matrix[row_A,col_A] * qb_B_confusion_matrix[row_B,col_B]
                    constructed_matrix[index_row,index_col] = val
    
    # Get difference between matrices.
    diff_matrix = qb_AB_confusion_matrix - constructed_matrix
    
    # Round the data to some number of decimal places?
    if round_data_to_decimal_places != 0:
        
        # Send warning that the data is being truncated / rounded?
        print("WARNING: the argument round_data_to_decimal_places was set to "+str(round_data_to_decimal_places)+", your confusion matrix is thus rounded to some decimal position accuracy. Typically, you should set round_data_to_decimal_places to 0 unless the plot itself becomes too messy to read.")
        
        # Round.
        diff_matrix = np.round(diff_matrix, round_data_to_decimal_places)
    
    # Plot?
    if plot_output or (export_filepath != ''):
        
        # Prepare canvas.
        canvas = diff_matrix
        
        # Make X and Y axes.
        x_axis = np.arange(0.0, canvas.shape[1], 1)
        y_axis = np.arange(0.0, canvas.shape[0], 1)
        x_axis_str = []
        y_axis_str = []
        for item in range(len(x_axis)):
            x_axis_str.append('|'+prepared_states[item]+'⟩')
            ## It's fine to use 'prepared_states' here
            ## since the matrices were asserted to be square.
        for item in range(len(y_axis)):
            y_axis_str.append('|'+prepared_states[item]+'⟩')
        
        # Set figure size
        plt.figure(figsize = figure_size_tuple)
        ##plt.figure(figsize = figure_size_tuple, dpi=600.0)
        
        # Better to plot the x axis along the top.
        plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        
        # Add numeric value to each box.
        for (j,i),label in np.ndenumerate(canvas):
            # if (float(label) < 0.15) or (float(label) >= 0.60):
            if (float(label) <= -0.2):
                plt.text( i, j, label, ha='center', va='center', weight='bold', color = 'white', fontsize=18)
            else:
                plt.text( i, j, label, ha='center', va='center', weight='bold', fontsize=18)
        
        # Make pixel palette.
        ##plt.imshow(canvas, interpolation='nearest')
        plt.imshow(canvas, interpolation='nearest', cmap = 'magma')
        plt.xticks(x_axis, x_axis_str, fontsize = 20)
        plt.yticks(y_axis, y_axis_str, fontsize = 20)
        if title == '':
            plt.title("Measured 2-qb confusion matrix\nminus calculated 2-qb confusion matrix", fontsize = 35, pad = 20)
        else:
            plt.title(title, fontsize = 35, pad = 20)
        
        # Show plot?
        if plot_output:
            plt.show()
        
        # Save plot?
        if export_filepath != '':
            if title == '':
                title = 'Confusion matrix difference'
            print("Saving plot "+title+".png")
            if export_filepath.endswith("Desktop"):
                print("You tried to name the export plot \"Desktop\", this is probably an error. Attempting to correct.")
                export_filepath = export_filepath + "\\"
            plt.savefig(export_filepath+title+".png")
        
    # Return result.
    return diff_matrix
    
    ## TODO:
    
    ## PLOT.
    ## RETURN RESULTING MATRIX.

def extract_confusion_matrix_data_from_larger_file(
    filepaths_to_plot,
    name_of_key_that_described_q1,
    name_of_key_that_described_q2,
    title = '',
    attempt_to_fix_string_input_argument = True,
    prepared_states = ['00','01','02','10','11','12','20','21','22'],
    figure_size_tuple = (15,14),
    round_data_to_decimal_places = 0,
    export_filepath = '',
    plot_output = False,
    ):
    ''' Given the filepath provided, extract the probability data.
        
        The format will be a matrix, x-axis is Measured state |0⟩, |1⟩, |2⟩,
        and the y-axis is Prepared state |00⟩, |01⟩, |02⟩ ... |21⟩, |22⟩.
    '''
    
    # We require some API tools from the Labber API.
    import Labber
    
    def get_stepchannelvalue(StepChannels, name):
        for i, item in enumerate(StepChannels):
            if item['name'] == name:
                return item['values']
        return np.nan
    
    # Before doing anything too advanced, figure out what the
    # interval is for the possible prepared states.
    minimum_possible_measured_state = 1048576
    maximum_possible_measured_state = 0
    number_of_qubits_detected = 0
    for current_item in prepared_states:
        counter = 0
        for i in current_item:
            # Is this state a lower state than the current
            # minimum_possible_state?
            if int(i) < minimum_possible_measured_state:
                print("Found new lowest possible state: |"+i+"⟩")
                minimum_possible_measured_state = int(i)
            # Is this state a higher state than the current
            # maximum_possible_state?
            if int(i) > maximum_possible_measured_state:
                print("Found new highest possible measured state: |"+i+"⟩")
                maximum_possible_measured_state = int(i)
            counter += 1
            if counter > number_of_qubits_detected:
                number_of_qubits_detected = counter
                print("Found one more qubit to analyse! The current number of qubits is "+str(number_of_qubits_detected))
    
    # Verify that all to-be-analysed measured states, contain the
    # correct number of qubits.
    for all_items in prepared_states:
        assert len(all_items) == number_of_qubits_detected, "Error! Detected that the requested measured state "+str(all_items)+" does not contain the expected number of qubits, which was detected to be "+str(number_of_qubits_detected)+"."
    
    ## The variables minimum_possible_measured_state, and
    ## maximum_possible_measured_state, contain the min and max values
    ## for the X axis, in the final confusion matrix plot.
    
    # User syntax repair
    if attempt_to_fix_string_input_argument:
        if isinstance(filepaths_to_plot, str):
            filepaths_to_plot = os.path.abspath(filepaths_to_plot)
            filepaths_to_plot = [filepaths_to_plot]
    
    # For all entries!
    return_vector = []
    for item in filepaths_to_plot:
        
        # Get current path
        filepath_to_file_with_probability_data_to_plot = item
        
        # Ensure that the filepath received can be processed.
        if type(filepath_to_file_with_probability_data_to_plot) == list:
            filepath_to_file_with_probability_data_to_plot = "".join(filepath_to_file_with_probability_data_to_plot)
        
        # Send warning that the data is being truncated / rounded?
        if round_data_to_decimal_places != 0:
            print("WARNING: the argument round_data_to_decimal_places was set to "+str(round_data_to_decimal_places)+", your confusion matrix is thus rounded to some decimal position accuracy. Typically, you should set round_data_to_decimal_places to 0 unless the plot itself becomes too messy to read.")
        
        # Get confusion matrix data.
        with h5py.File(filepath_to_file_with_probability_data_to_plot, 'r') as h5f:
            
            # Gather data to plot.
            hfile = Labber.LogFile(filepath_to_file_with_probability_data_to_plot)
            
            # Build data file axes.
            stepchannels = hfile.getStepChannels()
            q1_axis = get_stepchannelvalue(
                stepchannels,
                name_of_key_that_described_q1
            )
            q2_axis = get_stepchannelvalue(
                stepchannels,
                name_of_key_that_described_q2
            )
            
            # Assert that the axes are legal.
            for axes_to_check in [q1_axis, q2_axis]:
                # Check whether a bad argument was passed; this would likely
                # result in a NaN argument.
                try:
                    if np.isnan(axes_to_check):
                        raise TypeError("Error! One value axis passed a NaN value" + \
                            " from the Labber API. Check your input arguments!" + \
                            " Perhaps you wrote a \'\' or similar?")
                except ValueError:
                    # We expect that matrices fetched from Labber
                    # are normally multi-entried. "This is fine."
                    # However, we'll hate Labber's data structures for
                    # this odd bs. of Pythonic try-except tango.
                    pass
                
                # Check length.
                if len(axes_to_check) <= 0:
                    raise TypeError("Error! The axis "+str(axes_to_check) + \
                        " has zero length. Check your input arguments.")
            
            # Construct output matrix.
            '''aranged_measured_axis = \
                np.arange(minimum_possible_measured_state, \
                maximum_possible_measured_state+1)
            measured_axis = []
            for measured_state_yo in aranged_measured_axis:
                measured_axis += ['Measured |'+str(measured_state_yo)+'⟩']'''
            
            # For constructing the Measured axis, we are only
            # interested in the prepared values.
            measured_axis = []
            for measured_state_yo in prepared_states:
                #measured_axis += ['Measured |'+str(measured_state_yo)+'⟩']
                measured_axis += ['|'+str(measured_state_yo)+'⟩']
            
            # Prepare canvas of numbers.
            canvas = np.zeros([len(prepared_states),len(measured_axis)])
            
            ## Fetch data! ##
            ## At this point, the data structure will be weird from Labber's side.
            ## We need to slice it up, according to the length of our axes.
            ## We will progress through the data column-wise.
            counter_for_columns_in_output_canvas = 0
            for measured_state_yo in prepared_states:
                # Get the next column of the output canvas.
                print('Analysing data from measured state |'+str(measured_state_yo)+'⟩')
                data = hfile.getData(name = 'State Discriminator - Average 2Qstate P'+str(measured_state_yo))
                column = data.flatten()
                
                # Round the data to some number of decimal places?
                if round_data_to_decimal_places != 0:
                    for ll in range(len(column)):
                        column[ll] = round(column[ll], round_data_to_decimal_places)
                
                # Put the column in the canvas. Increment counter.
                canvas[:, counter_for_columns_in_output_canvas] = column
                counter_for_columns_in_output_canvas += 1
            
            # At this point, we've acquired the confusion matrix.
            # Time to plot?
            if plot_output or (export_filepath != ''):
            
                # Make X and Y axes.
                x_axis = np.arange(0.0, canvas.shape[1], 1)
                y_axis = np.arange(0.0, canvas.shape[0], 1)
                x_axis_str = measured_axis
                y_axis_str = []
                for item in range(len(y_axis)):
                    y_axis_str.append('|'+prepared_states[item]+'⟩')
                    #y_axis_str.append('Prepared |'+prepared_states[item]+'⟩')
                
                # Set figure size
                ##plt.figure(figsize = figure_size_tuple)
                plt.figure(figsize = figure_size_tuple, dpi=600.0)
                
                # Better to plot the x axis along the top.
                plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                
                # Add numeric value to each box.
                for (j,i),label in np.ndenumerate(canvas):
                    # if (float(label) < 0.15) or (float(label) >= 0.60):
                    if (float(label) <= 0.65):
                        plt.text( i, j, label, ha='center', va='center', weight='bold', color = 'white', fontsize=18)
                    else:
                        plt.text( i, j, label, ha='center', va='center', weight='bold', fontsize=18)
                
                # Make pixel palette.
                ##plt.imshow(canvas, interpolation='nearest')
                plt.imshow(canvas, interpolation='nearest', cmap = 'magma')
                plt.xticks(x_axis, x_axis_str, fontsize = 20)
                plt.yticks(y_axis, y_axis_str, fontsize = 20)
                if title == '':
                    plt.title("Confusion matrix, CZ & iSWAP", fontsize = 35, pad = 20)
                else:
                    plt.title(title, fontsize = 35, pad = 20)
                
                # Show plot?
                if plot_output:
                    plt.show()
                
                # Save plot?
                if export_filepath != '':
                    if title == '':
                        title = 'Confusion matrix'
                    print("Saving plot "+title+".png")
                    if export_filepath.endswith("Desktop"):
                        print("You tried to name the export plot \"Desktop\", this is probably an error. Attempting to correct.")
                        export_filepath = export_filepath + "\\"
                    plt.savefig(export_filepath+title+".png")
            
            # Append to return vector.
            if len(filepaths_to_plot) > 1:
                return_vector += [canvas]
            else:
                return_vector = canvas
    
    # Return result.
    return return_vector

def plot_logic_table(
    filepaths_to_plot,
    name_of_key_that_described_q1,
    name_of_key_that_described_q2,
    name_of_key_that_described_the_gates_attempted,
    names_of_gates_in_key_for_gates_attempted = ['iSWAP','CZ','Decomposed SWAP'],
    names_of_input_states_for_the_single_qubits = ['0','1','+','-','i','-i'],
    states_to_analyse = ['00','01','02','10','11','12','20','21','22'],
    attempt_to_fix_string_input_argument = True,
    figure_size_tuple = (12,40),
    export_filepath = '',
    plot_output = False,
    ):
    ''' For the input .hdf5 file, plot matrices showing a quantum
        logic table of sorts.
        
        figure_size_tuple is a tuple, somehow defining the dimensions of the
        output plot in yankeedoodleland units.
    '''
    
    # We require some API tools from the Labber API.
    import Labber
    
    def get_stepchannelvalue(StepChannels, name):
        for i, item in enumerate(StepChannels):
            if item['name'] == name:
                return item['values']
        return np.nan
    
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
        
        ## TODO: We may change our stance on how many states
        ##       that have to be plotted. The best way would be
        ##       to open the file, and see how many P( |00> ), P( |01> )...
        ##       there are in the file.
        
        # Gather data to plot.
        hfile = Labber.LogFile(filepath_to_file_with_probability_data_to_plot)
        
        # Build data file axes.
        stepchannels = hfile.getStepChannels()
        q1_axis = get_stepchannelvalue(
            stepchannels,
            name_of_key_that_described_q1
        )
        q2_axis = get_stepchannelvalue(
            stepchannels,
            name_of_key_that_described_q2
        )
        gate_sweep_axis = get_stepchannelvalue(
            stepchannels,
            name_of_key_that_described_the_gates_attempted
        )
        
        # Assert that the axes are legal.
        for axes_to_check in [q1_axis, q2_axis, gate_sweep_axis]:
            # Check whether a bad argument was passed; this would likely
            # result in a NaN argument.
            try:
                if np.isnan(axes_to_check):
                    raise TypeError("Error! One value axis passed a NaN value" + \
                        " from the Labber API. Check your input arguments!" + \
                        " Perhaps you wrote a \'\' or similar?")
            except ValueError:
                # We expect that matrices fetched from Labber
                # are normally multi-entried. "This is fine."
                # However, we'll hate Labber's data structures for
                # this odd bs. of Pythonic try-except tango.
                pass
            
            # Check length.
            if len(axes_to_check) <= 0:
                raise TypeError("Error! The axis "+str(axes_to_check) + \
                    " has zero length. Check your input arguments.")
            
            # Again.
            if len(names_of_gates_in_key_for_gates_attempted) == 0:
                raise TypeError("Error! No names provided for the "+\
                    "two-qubit gates that you swept.")
            elif len(gate_sweep_axis) != len(names_of_gates_in_key_for_gates_attempted):
                raise TypeError("Error! The argument bearing the names of" + \
                    " the two-qubit gates being analysed, has a length "+\
                    "that does not match the length of the swept key in "+\
                    "the Labber log file. Perhaps you erroneously used "+\
                    "the default names in this function, but in reality "+\
                    "made a smaller gate sweep in Labber? Or, you have "+\
                    "perhaps provided the wrong key name for the swept "+\
                    "parameter in your measurement that describes which "+\
                    "two-qubit gate you are currently analysing?")
            
            # And again.
            if len(names_of_input_states_for_the_single_qubits) == 0:
                raise TypeError("Error! No names provided for the "+\
                    "single-qubit input states that you swept.")
            elif len(q1_axis) != len(names_of_input_states_for_the_single_qubits):
                raise TypeError("Error! The argument bearing the names of" + \
                    " the single-qubit input states being analysed, has a"+\
                    " length that does not match the length of the axis in "+\
                    "the Labber log file that holds the input states for "+\
                    "qubit 1. Did you remember to provide the names for "+\
                    "the single-qubit input states as an argument?")
            elif len(q2_axis) != len(names_of_input_states_for_the_single_qubits):
                raise TypeError("Error! The argument bearing the names of" + \
                    " the single-qubit input states being analysed, has a"+\
                    " length that does not match the length of the axis in "+\
                    "the Labber log file that holds the input states for "+\
                    "qubit 2. Did you remember to provide the names for "+\
                    "the single-qubit input states as an argument?")
        
        # Make the matrices that will be output in the end.
        # The matrix matrices_to_output has the format [[], [], [] ... []]
        # ... where the number of empty matrices correspond to the number
        # of gate types that you are trying to analyse.
        matrices_to_output = [[]] * len(gate_sweep_axis)
        
        # We will set a flag here to keep track of whether we have a
        # big_fat_matrix that holds the data for the entire measurement output.
        big_fat_matrix_formed = False
        
        ## Fetch data! ##
        ## At this point, the data structure will be weird from Labber's side.
        ## We need to slice it up, according to the length of our axes.
        for current_output_state in range(len(states_to_analyse)):
            print("Analysing results for output state |"+\
                str(states_to_analyse[current_output_state])+"⟩") # <-- Careful! Unseen character!
            
            # Fetch the current column of all of the matrices' output.
            data = hfile.getData(name = 'State Discriminator - Average 2Qstate P'+str(states_to_analyse[current_output_state]))
            
            ## NOTE! Here, we may actually flatten the file.
            ##       The data held in data, will then corresond to the
            ##       entire P_xy column, for every matrix in the .hdf5 file.
            flattened_data = data.flatten()
            
            # We may now form the big_fat_matrix,
            # if it doesn't have the correct dimensions yet.
            if not big_fat_matrix_formed:
                
                # big_fat_matrix will be
                # flattened_data * "number of discretised states" in size.
                total_number_of_matrices = int(len(flattened_data) / (len(q1_axis) * len(q2_axis)))
                
                # Find out how many different configurations of matrices
                # that the user swept. I.e. parameters not related to
                # the actual matrix forming, in a way.
                number_of_configurations = int(total_number_of_matrices/len(gate_sweep_axis))
                
                # Print to the user.
                print("Detected that the entire file contains "+\
                    str(total_number_of_matrices)+" matrices, which " + \
                    "would be "+str(len(gate_sweep_axis))+" gates analysed, "+\
                    "for a total of "+str(number_of_configurations)+\
                    " different configurations.")
                
                # Let's form big_fat_matrix.
                big_fat_matrix = np.zeros((len(flattened_data),len(states_to_analyse)))
                
                # Set flag that the matrix has been formed.
                big_fat_matrix_formed = True
            
            # We now have the big_fat_matrix, and may start filling up
            # its columns. Which, would be flattened_data.
            big_fat_matrix[:, current_output_state] = flattened_data
        
        # At this point, we have formed the entire big_fat_matrix that contains
        # all of the population data for all matrices in the data file.
        ## Remember: columns in big_fat_matrix correspond to P00, P01, P02...
        print("Finished assembling matrices for every input- and output state.")
        
        ## For the different gate types, we assemble the output matrices
        ## one by one.
        matrix_of_matrices = [[]] * total_number_of_matrices
        names_of_the_finished_matrices = []
        iteration_counter = 0
        for iteration in range(number_of_configurations):
            for current_gate_worked_on in range(len(names_of_gates_in_key_for_gates_attempted)):
                
                # Get name of table.
                name_of_current_matrix = \
                    names_of_gates_in_key_for_gates_attempted[current_gate_worked_on] +" "+ str(iteration+1)
                names_of_the_finished_matrices.append( name_of_current_matrix )
                
                # Insert data into this table. Select the
                # (len(q1_axis) * len(q2_axis))_th number of rows from
                # the big_fat_matrix.
                number_of_input_states = int(len(q1_axis) * len(q2_axis))
                row_index_start = 0 + (iteration_counter * number_of_input_states)
                row_index_stop  = (number_of_input_states - 1) + (iteration_counter * number_of_input_states)
                
                # Cut out a portion of the big_fat_matrix and use it as the
                # matrix you're looking for.
                matrix_of_matrices[ iteration_counter ] = \
                    big_fat_matrix[ row_index_start:(row_index_stop+1), :] # NOTE!! +1 here is crucial! Try it yourself in numpy to see why :)
                
                # Keep track of the number of iterations.
                iteration_counter += 1
        
        # At this point, matrix of matrices contains all of the
        # output matrices, one by one. And, names_of_the_finished_matrices
        # contains their names.
        
        # Now, we plot them!
        iteration_counter = 0
        for matrix_to_plot in matrix_of_matrices:
            
            # Create the axes.
            # These numbers here serve as indices when plotting below.
            x_axis = np.arange(0.0, matrix_to_plot.shape[1], 1)
            y_axis = np.arange(0.0, matrix_to_plot.shape[0], 1)
            
            # For the Y-axis, we need to create all combiations of
            # input states from names_of_input_states_for_the_single_qubits.
            x_axis_str = []
            for iteration in states_to_analyse:
                i = iteration[0]
                j = iteration[1]
                if len(iteration) != 2:
                    raise ValueError("Halted! The plot function of this "+\
                        "file cannot plot other output states than 2-qubit"+\
                        " states. This fact should be fairly straight-"+\
                        "forward to patch.")
                if ((i == '-i') or (j == '-i')) or ((i == '-') and (j == 'i')) or ((i == 'i') and (j == '-')):
                    x_axis_str.append( '|'+i+','+j+'⟩' )
                else:
                    x_axis_str.append( '|'+i+j+'⟩' )
            # Remember to clean up!
            del i
            del j
            
            y_axis_str = []
            for i in names_of_input_states_for_the_single_qubits:
                for j in names_of_input_states_for_the_single_qubits:
                    if ((i == '-i') or (j == '-i')) or ((i == '-') and (j == 'i')) or ((i == 'i') and (j == '-')):
                        y_axis_str.append( '|'+i+','+j+'⟩' )
                    else:
                        y_axis_str.append( '|'+i+j+'⟩' )
            
            ## At this point, we have created the x- and y axes.
            
            # Set figure size
            plt.figure(figsize = figure_size_tuple)
            
            # Better to plot the x axis along the top.
            plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            
            # Make rounded edition to plot more nicely.
            rounded_matrix_to_plot = matrix_to_plot.copy()
            for rows in range(len(matrix_to_plot[:])):
                for cols in range(len(matrix_to_plot[0,:])):
                    rounded_matrix_to_plot[rows,cols] = np.round(matrix_to_plot[rows,cols],3)
            
            # Add numeric value to each box.
            for (j,i),label in np.ndenumerate(rounded_matrix_to_plot):
                if float(label) <= 0.45:
                    plt.text( i, j, label, ha='center', va='center', weight='bold', color = 'white', fontsize=35-28)
                else:
                    plt.text( i, j, label, ha='center', va='center', weight='bold', fontsize = 35-26)
            
            # Pad x- and y-axes.
            plt.tick_params(axis='x', which='major', pad=20)
            plt.tick_params(axis='y', which='major', pad=20)
            
            # Make pixel palette.
            ##plt.imshow(matrix_to_plot, interpolation='nearest')
            plt.imshow(rounded_matrix_to_plot, interpolation='nearest')
            plt.xticks(x_axis, x_axis_str, fontsize = 21-5)
            plt.yticks(y_axis, y_axis_str, fontsize = 21-5)
            
            # Get title.
            title = names_of_the_finished_matrices[ iteration_counter ]
            plt.title( title, fontsize = 35-8, pad = 27)
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot!
            if export_filepath != '':
                print("Saving plot "+title+".png")
                plt.savefig(export_filepath+title+".png")
            
            # Show plot?
            if plot_output:
                plt.show()
            
            # Increament iteration counter.
            iteration_counter += 1
    
    # Return numbers.
    return matrix_of_matrices

def plot_conditional_and_cross_Ramsey_expected(
    t1_of_qubit,
    duration_of_gates,
    figure_size_tuple = (2.654, 1.887),
    title = '',
    export_filepath = '',
    plot_output = True,
    ):
    ''' Plot the expected oscillations for conditional- and cross-Ramsey
        measurements.
        
        The figure size tuple is specified in eagles per hamburger,
        where 1.0 eagle per hamburger = 25.4 mm
        
        t1_of_qubit is provided in seconds. The exported filename
        will still attempt to list this number in µs.
        
        duration_of_gates is provided in seconds.
    '''
    
    # There will be three gates plotted.
    for gate_to_plot in ['CZ', 'iSWAP', 'SWAP']:
        
        # Set figure size
        plt.figure(figsize = figure_size_tuple, dpi=600.0)
        
        # Define x-axis.
        if gate_to_plot == 'CZ':
            pi_or_2pi = 1 * np.pi
        elif gate_to_plot == 'iSWAP':
            pi_or_2pi = 1 * np.pi # Override to ±1π radians for now.
            ##pi_or_2pi = 2 * np.pi
        elif gate_to_plot == 'SWAP':
            pi_or_2pi = 1 * np.pi # Override to ±1π radians for now.
            ##pi_or_2pi = 2 * np.pi
        else:
            raise AttributeError("Error! Attempting to plot an unknown gate's ideal x-axis values.")
        
        phase_values_half = np.arange(-pi_or_2pi, 0, 0.05);
        phase_values = phase_values_half
        phase_values = np.append(phase_values, np.array([0.0]))
        phase_values = np.append(phase_values, -1.0 * np.flip(phase_values_half))
        
        # Define axis limits.
        ax = plt.gca()
        ax.set_xlim([-pi_or_2pi, pi_or_2pi])
        ax.set_ylim([0.0, 1.0])
        
        # Define amplitude values.
        if gate_to_plot == 'CZ':
            ideal_frequency = 1/(2*pi_or_2pi)
        elif gate_to_plot == 'iSWAP':
            ideal_frequency = 1/(2*pi_or_2pi) # Override to ±1π radians for now.
            #ideal_frequency = 1/(2*pi_or_2pi) * 2 # Because the plot runs from -2π to 2 π.
        elif gate_to_plot == 'SWAP':
            ideal_frequency = 1/(2*pi_or_2pi) # Override to ±1π radians for now.
            #ideal_frequency = 1/(2*pi_or_2pi) * 2 # Because the plot runs from -2π to 2 π.
        
        ideal_offset = +0.50
        
        ## As for the /2 in the ideal_amplitude_points: remember that cos sways
        ## from -1 to +1. The ideal plot sways from 0 to +1, hence the /2 and
        ## +0.5 ideal offset.
        
        if gate_to_plot == 'CZ':
            ideal_amplitude_points = (np.cos(2 * np.pi * ideal_frequency * phase_values)) / 2 + ideal_offset
        elif gate_to_plot == 'iSWAP':
            ideal_amplitude_points = (-np.sin(2 * np.pi * ideal_frequency * phase_values)) / 2 + ideal_offset
        elif gate_to_plot == 'SWAP':
            ideal_amplitude_points = (np.cos(2 * np.pi * ideal_frequency * phase_values)) / 2 + ideal_offset
        else:
            raise AttributeError("Error! Attempting to plot an unknown gate's ideal amplitude values.")
        
        # Plot gate data.
        if gate_to_plot == 'CZ':
            dur = duration_of_gates[0]
            if t1_of_qubit > 0.0:
                plt.plot(phase_values, (ideal_amplitude_points) * np.e**(-dur/t1_of_qubit), ':', color="#34d2d6")
                plt.plot(phase_values, (-1.0 * ideal_amplitude_points + 1.0) * np.e**(-dur/t1_of_qubit), ':', color="#8934d6")
            else:
                plt.plot(phase_values, (ideal_amplitude_points), color="#34d2d6")
                plt.plot(phase_values, (-1.0 * ideal_amplitude_points + 1.0), color="#8934d6")
        elif gate_to_plot == 'iSWAP':
            dur = duration_of_gates[1]
            if t1_of_qubit > 0.0:
                plt.plot(phase_values, (ideal_amplitude_points) * np.e**(-dur/t1_of_qubit), ':', color="#d63834")
                plt.plot(phase_values, (-1.0 * ideal_amplitude_points + 1.0) * np.e**(-dur/t1_of_qubit), ':', color="#81d634")
            else:
                plt.plot(phase_values, (ideal_amplitude_points), color="#d63834")
                plt.plot(phase_values, (-1.0 * ideal_amplitude_points + 1.0), color="#81d634")
        elif gate_to_plot == 'SWAP':
            dur = duration_of_gates[2]
            if t1_of_qubit > 0.0:
                plt.plot(phase_values, (ideal_amplitude_points) * np.e**(-dur/t1_of_qubit), ':', color="#d63834")
                plt.plot(phase_values, (+1.0 * ideal_amplitude_points + 0.0) * np.e**(-dur/t1_of_qubit), ':', color="#81d634")
            else:
                plt.plot(phase_values, (ideal_amplitude_points), color="#d63834")
                plt.plot(phase_values, (+1.0 * ideal_amplitude_points + 0.0), ':', color="#81d634")
        
        # Disable axis ticks.
        plt.axis('off')
        
        # Save plot!
        if export_filepath != '':
            # Default the title to the gate that the user is trying to plot?
            if title == '':
                use_this_title = gate_to_plot
            else:
                use_this_title = title
            print("Saving plot "+use_this_title+".png")
            if export_filepath.endswith("Desktop"):
                print("You tried to name the export plot \"Desktop\", this is probably an error. Attempting to correct.")
                export_filepath = export_filepath + "\\"
            t1_of_qubit_formatted = str(int(t1_of_qubit*1e6))
            plt.savefig(export_filepath+use_this_title+"_"+str(t1_of_qubit_formatted)+"µs"+".png", bbox_inches="tight", pad_inches = 0, transparent=True)
        
        # Show plot?
        if plot_output:
            plt.show()

def plot_assignment_fidelity_vs_amplitude(
    amplitude_sweep_matrix,
    matrix_of_assignment_fidelities_vs_amplitude_sweep,
    export_filepath = '',
    figure_size_tuple = (23.5,10),
    set_transparent = False,
    ):
    ''' Given a matrix (that can be one row only) containing assignment
        fidelities of some dataset, and given an matrix of amplitudes swept,
        plot!
        
        TODO: in the future, this function should instead read input files.
    '''
    
    """ Here is a set of example data. """
    
    amplitude_sweep_matrix = [
        np.append(np.linspace(37e-3, 85e-3, 17),np.array([87e-3])),
        np.append(np.linspace(12e-3, 60e-3, 17),np.array([62e-3]))
    ]
    
    matrix_of_assignment_fidelities_vs_amplitude_sweep = [np.array([
        0.7926666666666666,
        0.7893333333333333,
        0.8162083333333333,
        0.8120833333333334,
        0.8145416666666667,
        0.8395833333333333,
        0.8505833333333334,
        0.8445833333333334,
        0.822,
        0.820125,
        0.8040416666666667,
        0.8537916666666666,
        0.7787083333333333,
        0.8392083333333333,
        0.8668333333333333,
        0.8792916666666667,
        0.8186666666666667,
        0.8117916666666667]),np.array([
        0.712625,
        0.758625,
        0.798625,
        0.8185833333333333,
        0.8387083333333333,
        0.8587916666666666,
        0.8694583333333333,
        0.8142916666666666,
        0.885125,
        0.8603333333333333,
        0.8654583333333333,
        0.8873333333333333,
        0.874,
        0.8837916666666666,
        0.8764583333333333,
        0.8667916666666666,
        0.8749583333333333,
        0.8680416666666667])
    ]
    
    # Create figure.
    plt.figure(figsize = figure_size_tuple, dpi = 600.0)
    plt.grid(linestyle='--', linewidth = 0.5)
    
    # Axis and figure size dimension management.
    plt.ylabel('Assignment fidelity [-]', fontsize = 37)
    plt.yticks(fontsize = 31)
    
    plt.xlabel('Readout IF amplitude [mV]', fontsize = 37)
    plt.xticks(fontsize = 31)
    
    # For every row,
    colour_table = ["#d63834","#81d634","#34d2d6","#8934d6"]
    
    ## TODO fix dynamic legend.
    
    
    for idx in range(len(matrix_of_assignment_fidelities_vs_amplitude_sweep)):
        
        # Append to plot:
        if idx < 4:
            plt.plot(amplitude_sweep_matrix[idx] * 1000, matrix_of_assignment_fidelities_vs_amplitude_sweep[idx], 'o', markersize = 20, color = colour_table[idx], label=f'Resonator {idx+1}')
        else:
            plt.plot(amplitude_sweep_matrix[idx] * 1000, matrix_of_assignment_fidelities_vs_amplitude_sweep[idx], 'o')
    
    # Legend and title.
    plt.legend(fontsize = 24)
    plt.title('RO amplitude optimisation', fontsize = 36)
    
    # Set axes.
    ##plt.xlim() No need.
    plt.ylim(0.65, 1.0)
    plt.xticks(fontsize = 31)
    plt.yticks(fontsize = 31)
    
    # Export?
    if export_filepath != '':
        if not (export_filepath[-1] == '\\'):
            export_filepath += '\\'
        print("Exporting to: "+str(export_filepath))
        plt.savefig(export_filepath + 'Readout amplitude optimisation' + ".png", bbox_inches = 'tight', transparent = set_transparent)
    
    # Show plot.
    plt.show()
    
    # No return.
    ##return
    
    