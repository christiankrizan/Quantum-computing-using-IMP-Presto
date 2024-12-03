#############################################################################################
#   Creative Commons Attribution Non Commercial Share Alike 4.0 International
#   CC-BY-NC-SA-4.0
#   https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#   https://github.com/christiankrizan/Quantum-computing-using-IMP-Presto/blob/master/LICENSE
#############################################################################################

import numpy as np
from scipy.linalg import fractional_matrix_power

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

def brute_force_local_gates_from_known_2q_equivalency_class(
    target_2q_gate,
    suspect_tx_exponent,
    suspect_ty_exponent,
    suspect_tz_exponent
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
        to SU(2).
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
    legal_values = [0+0j, 1+0j, -1+0j, 0-1j, 0+1j]
    
    # Define indices.
    K_1_00 = K_1_01 = K_1_10 = K_1_11 = 0+0j
    K_2_00 = K_2_01 = K_2_10 = K_2_11 = 0+0j
    K_3_00 = K_3_01 = K_3_10 = K_3_11 = 0+0j
    K_4_00 = K_4_01 = K_4_10 = K_4_11 = 0+0j
    
    # Let's count how many attempts were needed.
    # Below, we will discard all legal solutions that are not in SU(2).
    attempts = 0
    
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
                        
                        # Is K_1 in SU(2)? Legality check!
                        if check_if_gate_is_SU2(K_1):
                            
                            ## Single-qubit gate K_2
                            for K_2_00 in legal_values:
                                for K_2_11 in legal_values:
                                    for K_2_01 in legal_values:
                                        for K_2_10 in legal_values:
                                            
                                            # Construct local matrix K_2.
                                            K_2 = np.array([[K_2_00, K_2_01], [K_2_10, K_2_11]])
                                            
                                            # Is K_2 in SU(2)? Legality check!
                                            if check_if_gate_is_SU2(K_2):
                                            
                                                ## Single-qubit gate K_3
                                                for K_3_00 in legal_values:
                                                    for K_3_11 in legal_values:
                                                        for K_3_01 in legal_values:
                                                            for K_3_10 in legal_values:
                                                                
                                                                # Construct local matrix K_3.
                                                                K_3 = np.array([[K_3_00, K_3_01], [K_3_10, K_3_11]])
                                                                
                                                                # Is K_3 in SU(2)? Legality check!
                                                                if check_if_gate_is_SU2(K_3):
                                                                
                                                                    ## Single-qubit gate K_4
                                                                    for K_4_00 in legal_values:
                                                                        for K_4_11 in legal_values:
                                                                            for K_4_01 in legal_values:
                                                                                for K_4_10 in legal_values:
                                                                                    
                                                                                    # Construct local matrix K_4.
                                                                                    K_4 = np.array([[K_4_00, K_4_01], [K_4_10, K_4_11]])
                                                                                    
                                                                                    # Is K_4 in SU(2)? Legality check!
                                                                                    if check_if_gate_is_SU2(K_4):
                                                                                    
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
        print("Failure! Could not find local operations K_1 ... K_4, that transforms Can( t_x, t_y, t_z ) into the requested gate.")
    