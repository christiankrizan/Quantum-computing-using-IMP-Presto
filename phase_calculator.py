#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np

def get_legal_phase( value, available_phases_arr ):
    ''' Takes some array of phases deemed legal,
        and returns a value that is present in this array.
        
        All phases are done in radians.
    '''
    # Trim off 2-pi period from the value, and find where the nearest value is.
    diff_arr = np.absolute( available_phases_arr - (value % (2*np.pi)) )
    index = diff_arr.argmin()
    print(available_phases_arr[index])
    
    # Return result.
    return available_phases_arr[index]
    
def bandsign( if_value, default_to_lsb = False ):
    ''' Return +pi/2 or -pi/2 depending on the sign of the input value.
        But, never return 0.
    '''
    if (not default_to_lsb):
        return (np.sign(if_value)*np.pi/2) if (if_value != 0.0) else (-np.pi/2)
    else:
        return (np.sign(if_value)*np.pi/2) if (if_value != 0.0) else (+np.pi/2)