#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:50:28 2024

@author: yz23558
"""

import numpy as np
import math
#from scipy import linalg as la
from numpy import linalg as la
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import time
#import dmrg_tenpy 
import random
import itertools

import qiskit
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit_aer import Aer
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from scipy.stats import unitary_group
from qiskit.extensions import UnitaryGate

pauli_error
def get_noise(p_device, p_channel,n_qubit= 2,channel_basis = 'X'):

    error_device = depolarizing_error(p_device,n_qubit ) #isotropic error model
    error_channel = pauli_error([(channel_basis, p_channel), ('I', 1 - p_channel)]) # this is the quantum channeling process
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_device,'U')
    noise_model.add_all_qubit_quantum_error(error_channel,'id')

    return noise_model

def ising_gs(num_qubits):
    H = build_TFIZ(num_qubits,-1,-1,0,pbc=True).todense()
    evl, evcs = la.eigh(H)
    psi_gs = np.array(evcs[:,0])
    U = evcs
    return psi_gs

def reduced_dm(num_qubits,sub_size,basis = 'x',p = 0):
    psi_gs = ising_gs(num_qubits)
    inds = (f'{i}' for i in range(num_qubits))
    psi = qtn.Tensor(psi_gs.reshape([2]*num_qubits), inds)
    inds_reduced = tuple([str(i) for i in range(num_qubits-sub_size,num_qubits)])
    return partial_dm_dephasing(psi, inds_reduced,p = p, basis = basis).data

def pt_moment(rho,sub_size,tp_size = 0,order = 2, ob = np.eye(2)):
    ob_total = np.kron(ob,np.identity(2**(sub_size-1)))
    rho_cp = rho.copy()
    for i in range (tp_size):
        rho_cp = rho_cp.swapaxes(i,i+sub_size)
    rho_reshaped = rho_cp.reshape(2**(sub_size), 2**(sub_size))
    prod = rho_reshaped.copy()@ob_total
    for i in range (order-1):
        prod = prod@rho_reshaped@ob_total
    return prod
    
def pt_moment_normalized(num_qubits,sub_size,tp_size = 0,order = 2,basis = 'x',p = 0, ob= np.eye(2)):
    '''
    ob: whether adding a observable
    basis: direction of channeling
    sub_size: size of the subsystem
    tp_size: transpose the qubits within this size
    '''
    rho = reduced_dm(num_qubits,sub_size, basis = basis,p = p)

    pt_mo = pt_moment(rho,sub_size,tp_size = tp_size,order = order,ob = ob)
    norm = pt_moment(rho,sub_size,tp_size = 0,order = order)

    return np.trace(pt_mo)/np.trace(norm)

def gs(X):
    Q, R = np.linalg.qr(X)
    return Q

def psi_to_Uni(psi):
    # reshape the MPS tensor into matrix (p*chi, chi):
    psi_uni = np.identity(len(psi))+0j
    for i in range (len(psi)):
        psi_uni[i][0] = psi[i]
    psi_Uni = gs(psi_uni)
    psi_Uni[:,0] = psi
    return psi_Uni
# here is the noise circuit

def bitphaseflip_circuit(p):
    '''
    defines the noise channel
    '''
    K0 = np.sqrt(1-p)*np.eye(2)
    K1 = np.sqrt(p)*np.array([[0,1j],[1j,0]])
    qml.QubitChannel([K0, K1], wires=0)
    return qml.expval(qml.PauliZ(0))



def shadow_state_reconstruction(shadow, qubit_list, transposed = [],obs = None):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.

    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.

    Returns:
        reduced density matrix of the reconstructed quantum state.
    """
    num_snapshots = shadow[0].shape[0]

    # classical values
    b_lists, obs_lists = shadow

    # Averaging over snapshot states.
    shadow_rho1 = np.zeros((2 ** len(qubit_list), 2 ** len(qubit_list)), dtype=complex)
    shadow_rho2 = np.zeros((2 ** len(qubit_list), 2 ** len(qubit_list)), dtype=complex)
    shadow_rho3 = np.zeros((2 ** len(qubit_list), 2 ** len(qubit_list)), dtype=complex)

    for i in range(num_snapshots):
        rho = snapshot_state(b_lists[i], obs_lists[i], qubit_list, transposed)
        if obs != None:
            for ob in obs:
                shadow_rho1 += rho@ob
                shadow_rho2 += rho@ob@rho@ob
                shadow_rho3 += rho@ob@rho@ob@rho@ob
        else: 
            shadow_rho1 += rho
            shadow_rho2 += rho@rho
            shadow_rho3 += rho@rho@rho
            
    return shadow_rho1, shadow_rho2, shadow_rho3

def snapshot_state(b_list, obs_list, qubit_list, transposed = []):
    """
    Helper function for `shadow_state_reconstruction` that reconstructs
     a state from a single snapshot in a shadow.

    Implements Eq. (S44) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        b_list (array): The list of classical outcomes for the snapshot.
        obs_list (array): Indices for the applied Pauli measurement.

    Returns:
        Numpy array with the reconstructed snapshot.
    """
    num_qubits = len(b_list)

    # computational basis states
    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    # local qubit unitaries
    phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    # undo the rotations that were added implicitly to the circuit for the Pauli measurements
    unitaries = [hadamard, hadamard @ phase_z, identity]

    # reconstructing the snapshot state from local Pauli measurements
    rho_snapshot = [1]
    for i in qubit_list:
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        # applying Eq. (S44)
        local_rho = 3 * (U.conj().T @ state @ U) - identity
        if i in transposed:
            rho_snapshot = np.kron(rho_snapshot, local_rho.T)
        else:
            rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot

import quimb as qu
import quimb.tensor as qtn
def partial_dm(psi, inds):
    '''
    input: 
        psi :quimb wavefunction, 
        inds: indices that one wants to save
    returns: the quimb representation of the partial trace
    '''
    psi_h = psi.H
    for ind in psi_h.inds:
        
        if ind in inds:
            psi_h = psi_h.reindex({ind:ind+'c'})
    return (psi_h&psi)^all  

def partial_dm_dephasing(psi, inds, p, basis = 'z'):
    '''
    save as above, with the dephasing channels in the end
    '''
    if basis == 'x':
        pauli = np.array([[ 0.,  1.,],[ 1,  0]])
    if basis == 'y':
        pauli = np.array([[ 0 ,  -1j],[ 1j, 0]])
    if basis == 'z':    
        pauli = np.array([[ 1.,  0.,],[ 0, -1]])
    p_dep_tensor = np.zeros([2,2,2])+0j
    p_dep_tensor[0] = (1-p)*np.eye(2)
    p_dep_tensor[1] = p *pauli
    
    pc_dep_tensor = np.zeros([2,2,2])+0j
    pc_dep_tensor[0] = np.eye(2)
    pc_dep_tensor[1] = np.conj(pauli)

    psi_h = psi.H
    for ind in psi_h.inds:
        
        if ind in inds:
            psi_h = psi_h.reindex({ind:ind+'c'})
            
            inds_d ='a'+ind,ind,ind+'n'
            t = qtn.Tensor(p_dep_tensor,inds_d)
            inds_d ='a'+ind,ind+'c',ind+'cn'
            tc = qtn.Tensor(pc_dep_tensor,inds_d)
            
            psi_h = psi_h&tc
            psi = psi&t
    return (psi_h&psi)^all  

def second_renyi_entropy(rho1, rho2):
    return np.trace(rho1@rho2)

def critical_state_circuit(params,**kwargs):
    observables = kwargs.pop("observable")
    qml.QubitUnitary(U, wires=list(np.arange(0,num_qubits)))
    return [qml.expval(o) for o in observables]
#dev = qml.device("default.mixed", wires=num_qubits,shots=1)
#@qml.qnode(dev)
def critical_state_circuit_noise(params,**kwargs):
    p, basis = params

    observables = kwargs.pop("observable")
    qml.QubitDensityMatrix(rho, wires=range(num_qubits))
    if basis == 'x':
        pauli = np.array([[ 0.,  1.,],[ 1,  0]])
    if basis == 'y':
        pauli = np.array([[ 0 ,  -1j],[ 1j, 0]])
    if basis == 'z':    
        pauli = np.array([[ 1.,  0.,],[ 0, -1]])

    K0 = np.sqrt(1-p)*np.eye(2)
    K1 = np.sqrt(p)*pauli
    for i in range (l):
        qml.QubitChannel([K0, K1], wires=i)
    return [qml.expval(o) for o in observables]

def local_shadow_calculation(shadow_input, batch_size,):
    '''
    given a set of random measurement result
    returns the list of single qubit classical shadows
    output shape is (n_shadow, qubit)
    '''
    shadow_list = []
    for i in range(batch_size):
        rho_list = []
        for j in range(num_qubits):
            rho = shadow_state_reconstruction([np.array([shadow_input[0][i]]), np.array([shadow_input[1][i]])], qubit_list=[j])
            rho_list.append(rho)
        shadow_list.append(rho_list)
    return shadow_list

#shadow_list = local_shadow_calculation(shadow, batch_size,)

def PT_moment(shadow_list, Renyi_index, A_qubits, B_qubits):
    '''
    given:
    shadow_list: a list of single qubit classical shadows
    Renyi_index: an integer
    A_qubits: a list of qubits that region A is consist of 
    B_qubits: a list of qubits that region B is consist of 
    calculate the PT moment
    '''
    permutations = list(itertools.permutations(np.arange(len(shadow_list)), Renyi_index))
    estimate = 0
    for combo in permutations:
        trace = 1
        
        for qubit in A_qubits: #which qubit
            pdm = np.eye(2)
            for c in combo: #which shadow 
                pdm = pdm@shadow_list[c][qubit].T
            trace = trace*np.trace(pdm)
            
        for qubit in B_qubits: #which qubit
            pdm = np.eye(2)
            for c in combo: #which shadow 
                pdm = pdm@shadow_list[c][qubit]
            trace = trace*np.trace(pdm)
        estimate += trace
    PT = estimate /len(permutations)
    return PT

def PT_moment_third(shadow, A_qubits, B_qubits):
    '''
    given:
    shadow_list: a list of single qubit classical shadows
    Renyi_index: an integer
    A_qubits: a list of qubits that region A is consist of 
    B_qubits: a list of qubits that region B is consist of 
    calculate the PT moment
    '''
    estimate = 0
    qubit_list = A_qubits + B_qubits
    Nu = len(shadow[0])
    rho_1, rho_2, rho_3 = shadow_state_reconstruction(shadow, qubit_list, transposed = A_qubits)
    PT = (np.einsum('ab,bc,ca', rho_1, rho_1,rho_1) - 3*np.einsum('ab,ba', rho_1,rho_2, ) + 2*np.einsum('aa', rho_3))/(Nu*(Nu-1)*(Nu-2))

    return PT
def PT_moment(shadow, A_qubits, B_qubits, order = 2):
    '''
    given:
    shadow_list: a list of single qubit classical shadows
    Renyi_index: an integer
    A_qubits: a list of qubits that region A is consist of 
    B_qubits: a list of qubits that region B is consist of 
    calculate the PT moment
    '''
    estimate = 0
    qubit_list = A_qubits + B_qubits
    Nu = len(shadow[0])
    rho_1, rho_2, rho_3 = shadow_state_reconstruction(shadow, qubit_list, transposed = A_qubits)
    if order ==3:
        PT = (np.einsum('ab,bc,ca', rho_1, rho_1,rho_1) - 3*np.einsum('ab,ba', rho_1,rho_2, ) + 2*np.einsum('aa', rho_3))/(Nu*(Nu-1)*(Nu-2))
    else: 
        PT = (np.einsum('ab,ba', rho_1, rho_1) - np.einsum('aa', rho_2, ))/(Nu*(Nu-1))
    return PT


def PT_moment_second(shadow, A_qubits,B_qubits,obs = None):
    '''
    given:
    shadow_list: a list of single qubit classical shadows
    Renyi_index: an integer
    A_qubits: a list of qubits that region A is consist of 
    B_qubits: a list of qubits that region B is consist of 
    calculate the PT moment
    '''
    estimate = 0
    qubit_list = A_qubits + B_qubits
    if obs!= None:
        Nu = len(shadow[0])* len(obs)
    else: 
        Nu = len(shadow[0])
    rho_1, rho_2, rho_3 = shadow_state_reconstruction(shadow, qubit_list, transposed = A_qubits,obs = obs)
    PT = (np.einsum('ab,ba', rho_1, rho_1) - np.einsum('aa', rho_2, ))/(Nu*(Nu-1))

    return PT