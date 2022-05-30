#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
 
def barplot( filepath_to_file_with_probability_data_to_plot ):
    ''' Plot some state distribution set in a bar graph.
    '''
    
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
    while (not done):
        
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
                    raise ValueError( \
                        "Error! The file that was provided to the "         +\
                        "distribution bar plotter had an unexpected "       +\
                        "shape on the distribution data. The distribution " +\
                        "data was: \n"+str(printable))
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
    
    # Ensure that the total probability exactly equals 100%
    runsum = 0.0
    for item in processed_data:
        runsum += item
    assert runsum == 1.0, "Halted! The total probability of the loaded file does not equal 100%"
    
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
    
    plt.bar(states, probabilities, color ='#648fff',width = 0.2)
    plt.xticks(rotation=45)
    
    # Labels and title
    ##plt.xlabel("States")
    plt.ylabel("Probabilities")
    plt.title("Synthetic SWAP gate")
    
    # Plot!
    plt.show()