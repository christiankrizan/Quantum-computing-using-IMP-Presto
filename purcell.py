#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
import matplotlib.pyplot as plt

def plot_qubit_quality_vs_t1_tp(
    list_of_lists_of_resonator_frequencies_Hz,
    list_of_lists_of_qubit_frequencies_Hz,
    list_of_lists_of_dispersive_shift_chi_Hz,
    list_of_lists_of_anharmonicities_Hz,
    list_of_lists_of_resonator_linewidth_kappa_Hz,
    list_of_lists_of_T1_s,
    transpose_axes = False,
    savepath = ''
    ):
    ''' Plot qubit quality factor versus T₁/T_{purcell}.
        
        Using N.T. Bronn 2015, https://ieeexplore.ieee.org/document/7156088,
        professor Per Delsing found how to calculate the Purcell decay
        time. And, the qubit-resonator coupling factor.
        
        list_of_lists_of_{rest of variable name}:
        assumed to be in the format of list( list( ) ) -- example, let's say
        you had four qubits studied in the first round, and five qubits
        studied in the second round. The correct argument would then be:
            list_of_lists_of_resonator_frequencies_Hz = [ [1,2,3,4] , [1,2,3,4,5] ]
            list_of_lists_of__qubit_frequencies_Hz    = [ [1,2,3,4] , [1,2,3,4,5] ]
        
        Note here that each sub-list must have a length correspondence.
            assert len( list_of_lists_of_resonator_frequencies_Hz[0] ) == len( list_of_lists_of__qubit_frequencies_Hz[o] )
        
        And, this assertion must be valid for all list_of_lists_of_{parameter}!
    '''
    
    # Define formulas.
    def calculate_g(
        omega_r,
        omega_q,
        chi,
        eta
        ):
        ''' omega_r: Resonator angular frequency.
            omega_q: Qubit angular frequency.
            chi:     Dispersive shift.
            eta:     Transmon anharmonicity.
        '''
        Delta = omega_q - omega_r
        return np.sqrt((Delta * chi) * (eta + Delta)/eta)
    
    def calculate_Tp(
        omega_r,
        omega_q,
        g,
        kappa
        ):
        ''' omega_r: Resonator angular frequency.
            omega_q: Qubit angular frequency.
            g:       Qubit-resontor coupling strength, see "calculate_g".
            kappa:   Resonator linewidth.
        '''
        Delta = omega_q - omega_r
        simplified_term = (Delta**2) / ((g**2)*kappa)
        term1 = 1
        term2 = -2*( Delta/omega_q )
        term3 = (5/4) * ( (Delta**2)/(omega_q**2) )
        term4 = -(1/4) * ( (Delta**3)/(omega_q**3) )
        corr_factor = term1 + term2 + term3 + term4
        return simplified_term * corr_factor, (Delta/(2*np.pi))
    
    def calculate_qubit_quality_factor( omega_q, T1 ):
        return omega_q * T1
    
    # Create figure for plotting.
    fig, ax1 = plt.subplots(1, figsize=(12.6, 11.083), sharey=False)
    
    # For plotting purposes, keep track of the highest Y value in the plot.
    highest_ylim = 0.0
    
    # Loop through all the list_of_lists_of objects.
    # Each entry corresponds to one qubit.
    list_of_set_colours = ["#EE1C1C", "#1CEE70", "#1C70EE", "#C41CEE", "#C4EE1C"]
    list_of_qubit_scatter_symbols = ['s', '^', 'o', 'v', 'd', '*', 'x', 'p']
    for ii in range(len(list_of_lists_of_resonator_frequencies_Hz)):
        
        # Prepare axes.
        t1_tp_axis = []
        qubit_quality_axis = []
        
        # Get data.
        curr_omega_r_set = []
        curr_omega_q_set = []
        for kk in range(len(list_of_lists_of_resonator_frequencies_Hz[ii])):
            try:
                curr_omega_r_set.append( 2*np.pi * list_of_lists_of_resonator_frequencies_Hz[ii][kk] )
            except TypeError:
                curr_omega_r_set.append( None )
            try:
                curr_omega_q_set.append( 2*np.pi * list_of_lists_of_qubit_frequencies_Hz[ii][kk]     )
            except:
                curr_omega_q_set.append( None )
        curr_chi_set   = list_of_lists_of_dispersive_shift_chi_Hz[ii]
        curr_eta_set   = list_of_lists_of_anharmonicities_Hz[ii]
        curr_kappa_set = list_of_lists_of_resonator_linewidth_kappa_Hz[ii]
        curr_T1_set    = list_of_lists_of_T1_s[ii]
        
        # Statistics is nice. Let's prepare a T_p list.
        curr_Tp_list = []
        curr_Delta_Hz_list = []
        
        ## Let's make one scatter mass in the plot from index ii.
        
        # Calculate the X-axis T₁/T_p, and the Y-axis Q_qb.
        for jj in range(len(curr_omega_r_set)):
            
            # Are all parameters for this entry not None?
            if (curr_omega_r_set[jj] is not None) and (curr_omega_q_set[jj] is not None) and (curr_chi_set[jj] is not None) and (curr_eta_set[jj] is not None) and (curr_kappa_set[jj] is not None):
            
                # Get the current coupling strength.
                curr_g = calculate_g(
                    omega_r = curr_omega_r_set[jj],
                    omega_q = curr_omega_q_set[jj],
                    chi = curr_chi_set[jj],
                    eta = curr_eta_set[jj]
                )
                
                # Calculate the Purcell decay time for this entry.
                curr_Tp, curr_Delta_Hz = calculate_Tp(
                    omega_r = curr_omega_r_set[jj],
                    omega_q = curr_omega_q_set[jj],
                    g = curr_g,
                    kappa = curr_kappa_set[jj]
                )
                
                # Calculate T₁/T_p for current entry.
                t1_tp_axis.append( curr_T1_set[jj] / curr_Tp )
                
                print("Set "+str(ii+1)+", Qubit "+str(jj+1)+", calculated Tp: "+str(curr_Tp)+" [s], calculated Delta: "+str(curr_Delta_Hz)+" [Hz]")
                curr_Tp_list.append(curr_Tp)
                curr_Delta_Hz_list.append(curr_Delta_Hz)
            
                ## Now, get qubit quality factor for the Y-axis.
                curr_quality_factor = calculate_qubit_quality_factor( 
                    omega_q = curr_omega_q_set[jj],
                    T1 = curr_T1_set[jj],
                )
                if curr_quality_factor > highest_ylim:
                    highest_ylim = curr_quality_factor
                qubit_quality_axis.append( curr_quality_factor )
                
                # Detect whether we are plotting a cross;
                # if we are, then increase its size.
                if (jj == 6):
                    bump_up_size = 140
                else:
                    bump_up_size = 0
                
                # At this point, we have one scatter dot to plot.
                if not transpose_axes:
                    try:
                        ax1.scatter(t1_tp_axis[-1], qubit_quality_axis[-1]/1e6, s=130+bump_up_size, label=None, marker=list_of_qubit_scatter_symbols[jj], color=list_of_set_colours[ii])
                    except ValueError:
                        print(list_of_set_colours[ii])
                else:
                    ax1.scatter(qubit_quality_axis[-1]/1e6, t1_tp_axis[-1], s=130+bump_up_size, label=None, marker=list_of_qubit_scatter_symbols[jj], color=list_of_set_colours[ii])
            
            else:
                # Then this entry is a None. Something is missing.
                t1_tp_axis.append( None )
                qubit_quality_axis.append( None )
        
        # At this point, we can print some stats about curr_Tp_list
        curr_Tp_array = np.array(curr_Tp_list)
        curr_Delta_Hz_array = np.array(curr_Delta_Hz_list)
        print("Set "+str(ii+1)+", Tp mean: "+str(np.mean(curr_Tp_array))+" [s], Tp std. deviation: "+str(np.std(curr_Tp_array, ddof=1))+" [s]")
        print("Set "+str(ii+1)+", Delta mean: "+str(np.mean(curr_Delta_Hz_array/1e9))+" [GHz], Delta std. deviation: "+str(np.std(curr_Delta_Hz_array/1e9, ddof=1))+" [GHz]")
    
    # Labels and formatting stuff.
    ax1.grid()
    ## Adjust to [10^6] in quality factors, to plot correctly.
    highest_ylim /= 1e6
    if not transpose_axes:
        ax1.set_xlabel(r"$T_1 / T_p$ [-]", fontsize=33)
        ax1.set_ylabel("Qubit quality factor [$10^6$]", fontsize=33)
        ax1.set_xlim(-0.05, 1.05)
        ## ax1.set_ylim(-highest_ylim * 0.05, highest_ylim * 1.05)
        ##ax1.set_ylim(0.0, highest_ylim * 1.05)
        ax1.set_ylim(0.0, 2.603896)
        
    else:
        # Note that the metric keeping track of the highest T₁/T_p
        # in the plot, is still labeled "highest_ylim", even though it's
        # now an x_lim...
        ax1.set_ylabel(r"$T_1 / T_p$ [-]", fontsize=33)
        ax1.set_xlabel("Qubit quality factor [$10^6$]", fontsize=33)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(-highest_ylim * 0.05, highest_ylim * 1.05)
    ax1.tick_params(axis='both', labelsize=26)
    
    # Tight layout.    
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show stuff!
    plt.show()
    