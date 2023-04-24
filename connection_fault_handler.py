#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

from time import sleep
from time_calculator import get_timestamp_string
from presto.utils import ssh_reboot

def force_system_restart_over_ssh(
    ip_address
    ):
    ''' Restart the Presto lock-in AWG instrument remotely
        on Windows 10061 errors.
        
        Warning: if somebody else is running a measurement
        using the instrument, running this script will
        stop their measurement entirely.
    '''
    
    # Ensure legal IP address
    assert isinstance(ip_address, str), "Error! Could not parse provided IP address string."
    
    # Initiate remote reboot.
    print(  "The Presto died at "+str(get_timestamp_string(pretty = True)) + \
            ". Requiescat in frusta.\n\nRebooting...\n\n")
    
    # Check that the user has the required fabric package
    fabric_import_successful = False
    try:
        import fabric
        fabric_import_successful = True
    except ModuleNotFoundError:
        pass
    if not fabric_import_successful:
        raise ModuleNotFoundError("Error! Could not find the required package \"fabric\" - ensure that you have it installed.")
    del fabric_import_successful
    
    try:
        ssh_reboot(ip_address)
    except UnexpectedExit:
        # I have been asked to simply ignore this error.
        pass
    
    # Allow time for the instrument to reboot.
    sleep(10)
    print("Reboot attempt completed.")
    