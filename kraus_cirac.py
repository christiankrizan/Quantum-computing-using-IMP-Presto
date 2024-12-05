#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
from scipy.linalg import fractional_matrix_power
from datetime import datetime

def check_equivalency_class( t_x, t_y, t_z ):
    ''' Checks the user-provided exponents against a list of known
        two-qubit gate equivalency classes.
    '''
    # Alert user as to what (in)equivalency class is being investigated.
    if ((t_x == 0) and (t_y == 0) and (t_z == 0)):
        print("Equivalency class: Identity")
    elif ((t_x == 1/2) and (t_y == 0) and (t_z == 0)):
        print("Equivalency class: CNOT (CZ)")
    elif ((t_x == 1/2) and (t_y == 1/2) and (t_z == 0)):
        print("Equivalency class: iSWAP (DCNOT)")
    elif ((t_x == 1/2) and (t_y == 1/2) and (t_z == 1/2)):
        print("Equivalency class: SWAP")
    elif ((t_x == 1/4) and (t_y == 0) and (t_z == 0)):
        print("Equivalency class: CV (controlled-√X)")
    elif ((t_x == 1/4) and (t_y == 1/4) and (t_z == 0)):
        print("Equivalency class: √iSWAP")
    elif ((t_x == 3/8) and (t_y == 3/8) and (t_z == 0)):
        print("Equivalency class: DB (Dagwood-Bumstead)")
    elif ((t_x == 1/4) and (t_y == 1/4) and (t_z == 1/4)):
        print("Equivalency class: √SWAP")
    #elif ((t_x' == 3/4) and (t_y' == 1/4) and (t_z' == 1/4)):
    #    print("Equivalency class: √SWAP†")  ## TODO: Invertibility needed.
    elif ((t_x == 1/2) and (t_y == 1/4) and (t_z == 0)):
        print("Equivalency class: B (Berkeley)")
    elif ((t_x == 1/2) and (t_y == 1/4) and (t_z == 1/4)):
        print("Equivalency class: ECP (Entanglement concentration protocol)")
    elif ((t_x == 1/2) and (t_y == 1/2) and (t_z == 1/4)):
        print("Equivalency class: QFT (Quantum Fourier transform)")
    elif ((t_x == 1/2) and (t_y == 1/2) and (t_z == 1/12)):
        print("Equivalency class: Sycamore")
    elif ((t_x != 0) and (t_y == 0) and (t_z == 0)):
        print("Equivalency: Ising class gate (a controlled-something)")
    elif ((t_x != 0) and (t_y != 0) and (t_z == 0)):
        print("Equivalency: XY class gate")
    elif ((t_x == 1/2) and (t_y == 1/2) and (t_z != 0)):
        print("Equivalency: PSWAP class gate")
    elif ((t_x != 0) and (t_y != 0) and (t_z != 0)):
        print("Equivalency: Exchange-type gate")
    else:
        print("Unknown equivalency class: t_x = "+str(t_x)+", t_y = "+str(t_y)+", t_z = "+str(t_z)+"")

def sanitise(
    matrix,
    threshold = 1e-10
    ):
    ''' Take some matrix, and set obviously very small numbers to 0.
    '''
    return (np.where(np.abs(matrix) < threshold, 0, matrix))

def check_if_local_gates_work(
    equivalency_class_canonical_gate,
    K_1,
    K_2,
    K_3,
    K_4,
    target_2q_gate
    ):
    ''' Performs check whether a canonical gate, along with selected
        local operations, is equivalent to a target gate.
        K_1 ... K_4 is the single-qubit gates that wrap around the
        Canonical gate. K_1 ⊗ K_2 are the two single-qubit gates that run
        before the Canonical gate, and K_3 ⊗ K_4 are the single-qubit
        gates that run after the Canonical gate.
        Hence, the equation to check is:
        IF:
            Target = (K_3 ⊗ K_4) Can (K_1 ⊗ K_2)
        THEN:
            success
        ELSE:
            failure
    '''
    
    # Calculate the Kronecker products K_1 ⊗ K_2 and K_3 ⊗ K_4.
    K_12 = np.kron(K_1, K_2)
    K_34 = np.kron(K_3, K_4)
    
    # Calculate (K_34 Can K_12)
    gate = np.dot(K_34, np.dot(equivalency_class_canonical_gate, K_12))
    
    # Check if this gate is the target gate!
    ## Here, we have tolerancy errors due to computational stuff.
    ## Hence, we use np.allclose() instead.
    return np.allclose(target_2q_gate, gate)

def check_if_gate_is_SU2(
    single_qubit_gate,
    tolerance = 1e-10
    ):
    ''' Verifies that a 1-qubit gate is in SU(2).
        For this to be true, it must be unitary, and its determinant = 1.
        Checking the unitary condition is done through M * M^† = I.
    '''
    # Check if the single-qubit gate is unitary: M * M^† = I
    identity = np.eye(2, dtype=np.complex128)
    is_unitary = np.allclose(single_qubit_gate @ single_qubit_gate.conj().T, identity, atol=tolerance)
    
    # Check if the determinant is 1
    determinant = np.linalg.det(single_qubit_gate)
    is_determinant_one = np.isclose(determinant, 1, atol=tolerance)
    
    return (is_unitary and is_determinant_one)

def check_if_common(
    single_qubit_gate
    ):
    ''' Check if the reported gate is in the "known" single-qubit gate set.
    '''
    
    # Gates!
    identity = np.array([[1+0j,0+0j],[0+0j,1+0j]])
    pauli_x = np.array([[0+0j,1+0j],[1+0j,0+0j]])
    pauli_y = np.array([[0+0j,0-1j],[0+1j,0+0j]])
    pauli_z = np.array([[1+0j,0+0j],[0+0j,-1+0j]])
    v_gate = np.array([[(1+1j)/2, (1-1j)/2],[(1-1j)/2, (1+1j)/2]]) # Also known as sqrt(X)
    inverse_v_gate = np.array([[(1-1j)/2, (1+1j)/2],[(1+1j)/2, (1-1j)/2]]) # Also known as inverse of sqrt(X)
    pseudo_hadamard_gate = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],[-1/np.sqrt(2), 1/np.sqrt(2)]]) # Also known as sqrt(Y)†, or h
    inverse_pseudo_hadamard_gate = np.array([[1/np.sqrt(2), -1/np.sqrt(2)],[1/np.sqrt(2), 1/np.sqrt(2)]]) # Also known as sqrt(Y), or h†
    s_gate = np.array([[1+0j, 0+0j],[0+0j, 0+1j]])
    s_dag_gate = np.array([[1+0j, 0+0j],[0+0j, 0-1j]])
    hadamard = np.array([[(1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2)],[(1+0j)/np.sqrt(2), (-1+0j)/np.sqrt(2)]])
    t_gate = np.array([[1+0j, 0+0j],[0+0j, 0.70710678118+0.70710678118j]])
    t_dag_gate = np.array([[1+0j, 0+0j],[0+0j, 0.70710678118-0.70710678118j]])
    
    # Single-qubit gate set:
    gate_set = [identity, pauli_x, pauli_y, pauli_z, v_gate, inverse_v_gate, pseudo_hadamard_gate, inverse_pseudo_hadamard_gate, s_gate, s_dag_gate, hadamard, t_gate, t_dag_gate]
    names = ["Identity", "Pauli X", "Pauli Y", "Pauli Z", "V gate", "Inverse V gate", "Pseudo-Hadamard", "Inverse pseudo-Hadamard", "S gate", "S† gate", "Hadamard", "T gate", "T† gate"]
    
    # Is the gate common?
    for suspect in range(len(gate_set)):
        if np.allclose(single_qubit_gate, gate_set[suspect]):
            # The gate is common!
            print(names[suspect])
            return True
    return False

def check_if_K_needs_sqrt(
    single_qubit_gate
    ):
    ''' If the local operation needs some square-root magic, verify that
        it is some known thing at least.
    '''
    
    # Prepare suspicious gate flag.
    suspicious = False
    
    # Define legal gates.
    gate_v = np.array([[(1+1j)/2, (1-1j)/2],[(1-1j)/2, (1+1j)/2]])
    gate_inv_v = np.array([[(1-1j)/2, (1+1j)/2],[(1+1j)/2, (1-1j)/2]])
    gate_pseudo_h = np.array([[(1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2)],[(-1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2)]])
    gate_inv_pseudo_h = np.array([[(1+0j)/np.sqrt(2), (-1+0j)/np.sqrt(2)],[(1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2)]])
    gate_hadamard = np.array([[(1+0j)/np.sqrt(2), (1+0j)/np.sqrt(2)],[(1+0j)/np.sqrt(2), (-1+0j)/np.sqrt(2)]])
    gate_cycling = np.array([[(1-1j)/2, (-1-1j)/2],[(1-1j)/2, (+1+1j)/2]])
    
    ##gate_rot_x_0 = np.array([[np.cos(0/2), (0-1j)*np.sin(0/2)],[(0-1j)*np.sin(0/2), np.cos(0/2)]])
    gate_rot_x_pi_div2 = np.array([[np.cos(np.pi/4), (0-1j)*np.sin(np.pi/4)],[(0-1j)*np.sin(np.pi/4), np.cos(np.pi/4)]])
    gate_rot_x_pi = np.array([[np.cos(np.pi/2), (0-1j)*np.sin(np.pi/2)],[(0-1j)*np.sin(np.pi/2), np.cos(np.pi/2)]])
    gate_rot_x_3pi_div2 = np.array([[np.cos(3*np.pi/4), (0-1j)*np.sin(3*np.pi/4)],[(0-1j)*np.sin(3*np.pi/4), np.cos(3*np.pi/4)]])
    
    ##gate_rot_y_0 = np.array([[np.cos(0/2),-np.sin(0/2)],[np.sin(0/2),np.cos(0/2)]])
    gate_rot_y_pi_div2 = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])
    gate_rot_y_pi = np.array([[np.cos(np.pi/2),-np.sin(np.pi/2)],[np.sin(np.pi/2),np.cos(np.pi/2)]])
    gate_rot_y_3pi_div2 = np.array([[np.cos(3*np.pi/4),-np.sin(3*np.pi/4)],[np.sin(3*np.pi/4),np.cos(3*np.pi/4)]])
    
    ##gate_rot_z_0 = np.array([[1, 0],[0, 1]])
    gate_rot_z_pi_div2 = np.array([[0.707106781-0.707106781j, 0+0j],[0+0j, 0.707106781+0.707106781j]])
    gate_rot_z_pi = np.array([[0-1j, 0+0j],[0+0j, 0+1j]])
    gate_rot_z_3pi_div2 = np.array([[-0.707106781-0.707106781j, 0+0j],[0+0j, -0.707106781+0.707106781j]])
    
    # Define illegal gates for checking later.
    matrix_numbers_that_can_be_illegal = {
        (1 + 1j) / 2,
        (1 - 1j) / 2,
        (1 + 0j) / np.sqrt(2),
        (-1 + 0j) / np.sqrt(2),
        (+1 + 1j) / 2,
        (-1 - 1j) / 2,
    }
    
    # Attempted sqrt(X)?
    if np.any(np.isclose(single_qubit_gate, gate_v)):
        # Some value reminds us of a V gate.
        if (np.all(np.isclose(single_qubit_gate, gate_v))):
            # The gate was identified as a V gate.
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Attempted inverse sqrt(X)?
    if np.any(np.isclose(single_qubit_gate, gate_inv_v)):
        # Some value reminds us of an inverse V gate.
        if (np.all(np.isclose(single_qubit_gate, gate_inv_v))):
            # The gate was identified as an inverse V gate.
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Attempted sqrt(Y)?
    if np.any(np.isclose(single_qubit_gate, gate_pseudo_h)):
        # Some value reminds us of an h gate.
        if (np.all(np.isclose(single_qubit_gate, gate_pseudo_h))):
            # The gate was identified as an h gate.
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Attempted inverse sqrt(Y)?
    if np.any(np.isclose(single_qubit_gate, gate_inv_pseudo_h)):
        # Some value reminds us of an inverse h gate.
        if (np.all(np.isclose(single_qubit_gate, gate_inv_pseudo_h))):
            # The gate was identified as an inverse h gate.
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Attempted Hadamard?
    if np.any(np.isclose(single_qubit_gate, gate_hadamard)):
        # Some value reminds us of a hadamard gate.
        if (np.all(np.isclose(single_qubit_gate, gate_hadamard))):
            # The gate was identified as a hadamard gate.
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Attempted cycling?
    if np.any(np.isclose(single_qubit_gate, gate_cycling)):
        # Some value reminds us of a cycling gate.
        if (np.all(np.isclose(single_qubit_gate, gate_cycling))):
            # The gate was identified as a cycling gate.
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Rotation around X?
    if np.any(np.isclose(single_qubit_gate, gate_rot_x_pi_div2)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_x_pi_div2))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    if np.any(np.isclose(single_qubit_gate, gate_rot_x_pi)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_x_pi))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    if np.any(np.isclose(single_qubit_gate, gate_rot_x_3pi_div2)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_x_3pi_div2))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Rotation around Y?
    if np.any(np.isclose(single_qubit_gate, gate_rot_y_pi_div2)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_y_pi_div2))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    if np.any(np.isclose(single_qubit_gate, gate_rot_y_pi)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_y_pi))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    if np.any(np.isclose(single_qubit_gate, gate_rot_y_3pi_div2)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_y_3pi_div2))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # Rotation around Z?
    if np.any(np.isclose(single_qubit_gate, gate_rot_z_pi_div2)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_z_pi_div2))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    if np.any(np.isclose(single_qubit_gate, gate_rot_z_pi)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_z_pi))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    if np.any(np.isclose(single_qubit_gate, gate_rot_z_3pi_div2)):
        if (np.all(np.isclose(single_qubit_gate, gate_rot_z_3pi_div2))):
            return False
        else:
            # The gate is suspicious.
            suspicious = True
    
    # In case nobody reported suspicion, check whether the gate is
    # some combination of square root magic.
    if (not suspicious):
        
        # Flatten the matrix and check if any element is in the
        # set of illegal numbers. If so, return True, it is suspicious.
        if (any(element in matrix_numbers_that_can_be_illegal for element in single_qubit_gate.flatten())):
            return True
    
    # Finally, report suspicious gate?
    return suspicious
    

def brute_force_local_gates_from_known_2q_equivalency_class(
    target_2q_gate,
    suspect_tx_exponent,
    suspect_ty_exponent,
    suspect_tz_exponent,
    disregard_su2_requirement = False
    ):
    ''' Given a known 2-qubit (in)equivalency class, and given some known
        2-qubit gate that the user suspects is part of that (in)equivalency
        class, where the Pauli-XX, Pauli-YY, Pauli-ZZ exponent coordinates in
        the Weyl chamber are given as t_x, t_y, t_z
        (that is, Can( t_x, t_y, t_z ) = XX^t_x · YY^t_y · ZZ^t_z),
        then BRUTE FORCE from a set of possible complex numbers,
        the four different single qubit gates K_1 .... K_4,
        that would assert that the suspected canonical gate Can(t_x, t_y, t_z),
        can be transformed into the gate that the user suspects is part of
        that (in)equivalency class that is represented by t_x, t_y, t_z).
        While, also asserting that the single qubit gates K_1 ... K_4 belong
        to SU(2). Unless the flag disregard_su2_requirement is set, that is.
    '''
    
    # Create shorthand.
    t_x = suspect_tx_exponent
    t_y = suspect_ty_exponent
    t_z = suspect_tz_exponent
    
    # Type conversion.
    target_2q_gate = np.array(target_2q_gate)
    
    # Alert user regarding the checked equivalency class.
    check_equivalency_class( t_x, t_y, t_z )
    
    # Set initial flag.
    success = False
    
    ## Algorithm:
    ## Create the Canonical gate based on the provided exponents.
    
    ## Generate K_i, check legality of K_i, proeed with K_{i+1} until K_4 = OK.
    ## Create the Kronecker products K_1 ⊗ K_2 and K_3 ⊗ K_4.
    ## Check whether TARGET = K_12 Can K_34. If yes, update success flag.
    
    ## If success, break!
    ## Else: if the last possible combination was tried, report failure.
    
    # Define initial local operations.
    K_1 = np.array([[1+0j,0+0j],[0+0j,1+0j]])
    K_2 = np.array([[1+0j,0+0j],[0+0j,1+0j]])
    K_3 = np.array([[1+0j,0+0j],[0+0j,1+0j]])
    K_4 = np.array([[1+0j,0+0j],[0+0j,1+0j]])
    
    # Define Pauli_XX, Pauli_YY, Pauli_ZZ matrices.
    pauli_XX = np.array([[0, 0, 0,  1], [0,  0, 1, 0], [0, 1,  0, 0],  [1, 0, 0, 0]])
    pauli_YY = np.array([[0, 0, 0, -1], [0,  0, 1, 0], [0, 1,  0, 0], [-1, 0, 0, 0]])
    pauli_ZZ = np.array([[1, 0, 0,  0], [0, -1, 0, 0], [0, 0, -1, 0],  [0, 0, 0, 1]])
    
    # Create the target Can( t_x, t_y, t_z ) gate in the Weyl chamber based on
    # the user-provided coordinates.
    ## Note here the usage of the sanitise function, defined above.
    XX = fractional_matrix_power(pauli_XX, t_x)
    YY = fractional_matrix_power(pauli_YY, t_y)
    ZZ = fractional_matrix_power(pauli_ZZ, t_z)
    equivalency_class_canonical_gate = sanitise(np.dot(XX, np.dot(YY, ZZ)))
    
    # Define lists of parameters that will be used when trying
    # to make the local operations.
    ## Note that modifying this list, even though it's probably a good thing,
    ## will add a metric ass-tonne of computational complexity => time goes up!
    '''legal_values = [0+0j, 1+0j, -1+0j, 0-1j, 0+1j]'''
    legal_values = [0+0j, 1+0j, -1+0j, 0-1j, 0+1j, (1+1j)/2, (1-1j)/2, 0.70710678118+0j, -0.70710678118+0j, 0+0.70710678118j, 0-0.70710678118j, 0.70710678118+0.70710678118j, 0.70710678118-0.70710678118j, -0.70710678118+0.70710678118j, -0.70710678118-0.70710678118j]
    #legal_values = [0+0j, 1+0j, -1+0j, 0-1j, 0+1j, (1+1j)/2, (1-1j)/2, 0.70710678118+0j, -0.70710678118+0j]
    
    # Define indices.
    K_1_00 = K_1_01 = K_1_10 = K_1_11 = 0+0j
    K_2_00 = K_2_01 = K_2_10 = K_2_11 = 0+0j
    K_3_00 = K_3_01 = K_3_10 = K_3_11 = 0+0j
    K_4_00 = K_4_01 = K_4_10 = K_4_11 = 0+0j
    
    # Let's count how many attempts were needed.
    # Below, we will discard all legal solutions that are not in SU(2).
    ## Unless explicitly requested by the user.
    attempts = 0
    
    # Print timestamp.
    print("Began at: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # May God forgive me. If he still cares.
    ## Single-qubit gate K_1
    for K_1_00 in legal_values:
        ## Not super-pretty way of a multi-nestled break-continue clause.
        if not success:
            for K_1_11 in legal_values:
                for K_1_01 in legal_values:
                    for K_1_10 in legal_values:
                    
                        # Construct local matrix K_1.
                        K_1 = np.array([[K_1_00, K_1_01], [K_1_10, K_1_11]])
                        
                        # SU(2) fulfilled?
                        K1_is_su2 = ((check_if_gate_is_SU2(K_1) or disregard_su2_requirement))
                        
                        # If it was, check its reasonability.
                        if K1_is_su2:
                            # Barrier for banning-off unknown combinations
                            # of sqrt(X) and sqrt(Y) gates.
                            K1_has_sqrt_issues = check_if_K_needs_sqrt(K_1)
                        
                        # Continue?
                        if K1_is_su2 and (not K1_has_sqrt_issues):
                            
                            ## Single-qubit gate K_2
                            for K_2_00 in legal_values:
                                for K_2_11 in legal_values:
                                    for K_2_01 in legal_values:
                                        for K_2_10 in legal_values:
                                            
                                            # Construct local matrix K_2.
                                            K_2 = np.array([[K_2_00, K_2_01], [K_2_10, K_2_11]])
                                            
                                            # SU(2) fulfilled?
                                            K2_is_su2 = ((check_if_gate_is_SU2(K_2) or disregard_su2_requirement))
                                            
                                            # If it was, check its reasonability.
                                            if K2_is_su2:
                                                # Barrier for banning-off unknown combinations
                                                # of sqrt(X) and sqrt(Y) gates.
                                                K2_has_sqrt_issues = check_if_K_needs_sqrt(K_2)
                                            
                                            # Continue?
                                            if K2_is_su2 and (not K2_has_sqrt_issues):
                                                
                                                ## Single-qubit gate K_3
                                                for K_3_00 in legal_values:
                                                    for K_3_11 in legal_values:
                                                        for K_3_01 in legal_values:
                                                            for K_3_10 in legal_values:
                                                                
                                                                # Construct local matrix K_3.
                                                                K_3 = np.array([[K_3_00, K_3_01], [K_3_10, K_3_11]])
                                                                
                                                                # SU(2) fulfilled?
                                                                K3_is_su2 = ((check_if_gate_is_SU2(K_3) or disregard_su2_requirement))
                                                                
                                                                # If it was, check its reasonability.
                                                                if K3_is_su2:
                                                                    # Barrier for banning-off unknown combinations
                                                                    # of sqrt(X) and sqrt(Y) gates.
                                                                    K3_has_sqrt_issues = check_if_K_needs_sqrt(K_3)
                                                                
                                                                # Continue?
                                                                if K3_is_su2 and (not K3_has_sqrt_issues):
                                                                    
                                                                    ## Single-qubit gate K_4
                                                                    for K_4_00 in legal_values:
                                                                        for K_4_11 in legal_values:
                                                                            for K_4_01 in legal_values:
                                                                                for K_4_10 in legal_values:
                                                                                    
                                                                                    # Construct local matrix K_4.
                                                                                    K_4 = np.array([[K_4_00, K_4_01], [K_4_10, K_4_11]])
                                                                                    
                                                                                    # SU(2) fulfilled?
                                                                                    K4_is_su2 = ((check_if_gate_is_SU2(K_4) or disregard_su2_requirement))
                                                                                    
                                                                                    # If it was, check its reasonability.
                                                                                    if K4_is_su2:
                                                                                        # Barrier for banning-off unknown combinations
                                                                                        # of sqrt(X) and sqrt(Y) gates.
                                                                                        K4_has_sqrt_issues = check_if_K_needs_sqrt(K_4)
                                                                                    
                                                                                    # Continue?
                                                                                    if K4_is_su2 and (not K4_has_sqrt_issues):
                                                                                    
                                                                                        # Verify!
                                                                                        test_happens_here = check_if_local_gates_work(
                                                                                            equivalency_class_canonical_gate,
                                                                                            K_1,
                                                                                            K_2,
                                                                                            K_3,
                                                                                            K_4,
                                                                                            target_2q_gate
                                                                                            )
                                                                                        
                                                                                        # Increment counter
                                                                                        attempts += 1
                                                                                        
                                                                                        # I'm unsure why I have to write things like this, tbh.
                                                                                        if test_happens_here:
                                                                                            success = True
                                                                                            
                                                                                            # Check if the solution uses common gates.
                                                                                            common_K1 = check_if_common(K_1)
                                                                                            common_K2 = check_if_common(K_2)
                                                                                            common_K3 = check_if_common(K_3)
                                                                                            common_K4 = check_if_common(K_4)
                                                                                            if (common_K1 and common_K2 and common_K3 and common_K4):
                                                                                                print("------------------------------------------------")
                                                                                                print("COMMON SOLUTION FOUND at attempt "+str(attempts)+":")
                                                                                                print("K_1 is:\n"+str(K_1))
                                                                                                print("\nK_2 is:\n"+str(K_2))
                                                                                                print("\nK_3 is:\n"+str(K_3))
                                                                                                print("\nK_4 is:\n"+str(K_4))
                                                                                                print("------------------------------------------------")
                                                                                            else:
                                                                                                # We accept all SU(2) gates.
                                                                                                print("\nSuccessful find at attempt "+str(attempts)+":")
                                                                                                print("K_1 is:\n"+str(K_1))
                                                                                                print("\nK_2 is:\n"+str(K_2))
                                                                                                print("\nK_3 is:\n"+str(K_3))
                                                                                                print("\nK_4 is:\n"+str(K_4))
        else:
            # Quicker way to break the nestled-break hack.
            break
    
    # Report.
    if not success:
        print("Failure! Could not find local operations K_1 ... K_4, that transforms Can( t_x = "+str(t_x)+", t_y = "+str(t_y)+", t_z = "+str(t_z)+" ) into the requested gate.")
    
    # Print timestamp.
    print("Tested "+str(attempts)+" legal combinations. Finished at: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Return.
    return success
    