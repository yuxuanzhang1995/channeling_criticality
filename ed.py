import scipy.sparse
import numpy as np

paulis = {'I': scipy.sparse.csr_matrix(np.eye(2).astype(np.complex64)),
          'x': scipy.sparse.csr_matrix(np.array([[0, 1], [1, 0]]).astype(np.complex64)),
          'y': scipy.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]).astype(np.complex64)),
          'z': scipy.sparse.csr_matrix(np.array([[1, 0], [0, -1]]).astype(np.complex64))}

def build_two_body(interactions, L: int):
    ham = scipy.sparse.csr_matrix((int(2 ** L), (int(2 ** L))), dtype=complex)
    for (scalar, term_1, term_2) in interactions:
        # XX term
        tprod = ["I" for _ in range(L)]
        loc_i = term_1[1]
        loc_j = term_2[1]
        tprod[loc_i] = term_1[0]
        tprod[loc_j] = term_2[0]
        p = paulis[tprod[0]]
        for op in range(1, L):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        ham += scalar * p
    return ham


def build_one_body(interactions, L: int):
    ham = scipy.sparse.csr_matrix((int(2 ** L), (int(2 ** L))), dtype=complex)
    for (scalar, term_1) in interactions:
        # XX term
        tprod = ["I" for _ in range(L)]
        loc_i = term_1[1]
        tprod[loc_i] = term_1[0]
        p = paulis[tprod[0]]
        for op in range(1, L):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        ham += scalar * p
    return ham


def build_XXZ(L, J: float, Delta: float, pbc: bool = False):
    """
    builds tfim Hamiltonian
    """
    ## Setup basis

    ## Operator lists
    if pbc:
        J_xx = [[J, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_yy = [[J, ('y', i), ('y', (i + 1) % L)] for i in range(L)]  # PBC
        J_zz = [[Delta * J, ('z', i), ('z', (i + 1) % L)] for i in range(L)]  # PBC
    else:
        J_xx = [[J, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_yy = [[J, ('y', i), ('y', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_zz = [[Delta * J, ('z', i), ('z', (i + 1) % L)] for i in range(L - 1)]  # OBC
    interactions = J_xx + J_yy + J_zz
    H = build_two_body(interactions, L)
    return H


def build_SDIsing(L, Jxx: float, Jz: float, V: float, pbc=False):
    """
    builds tfim Hamiltonian
    """
    ## Operator lists
    if pbc:
        J_xx = [[Jxx, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_z = [[Jz, ('z', i)] for i in range(L)]  # OBC
        V_xx = [[V, ('x', i), ('x', (i + 2) % L)] for i in range(L)]  # PBC
        V_zz = [[V, ('z', i), ('z', (i + 1) % L)] for i in range(L)]  # PBC
    else:
        J_xx = [[Jxx, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_z = [[Jz, ('z', i)] for i in range(L)]  # OBC
        V_xx = [[V, ('x', i), ('x', (i + 2) % L)] for i in range(L - 2)]  # OBC
        V_zz = [[V, ('z', i), ('z', (i + 1) % L)] for i in range(L - 1)]  # OBC

    interactions_two_body = J_xx + V_xx + V_zz
    ham_two = build_two_body(interactions_two_body, L)
    ham_one = build_one_body(J_z, L)
    ham = ham_two + ham_one
    return ham


# here are the ED results

def build_TFIZ(L, Jxx: float, Jz: float, Jx: float, pbc=False):
    """
    builds tfimz Hamiltonian
    """

    ## Operator lists
    if pbc:
        J_xx = [[Jxx, ('x', i), ('x', (i + 1) % L)] for i in range(L)]  # PBC
        J_x = [[Jx, ('x', i)] for i in range(L)]  # PBC
        J_z = [[Jz, ('z', i)] for i in range(L)]  # PBC
    else:
        J_xx = [[Jxx, ('x', i), ('x', (i + 1) % L)] for i in range(L - 1)]  # OBC
        J_x = [[Jx, ('x', i)] for i in range(L)]  # PBC
        J_z = [[Jz, ('z', i)] for i in range(L)]  # PBC

    ham_two = build_two_body(J_xx, L)
    interactions_one_body = J_x + J_z
    ham_one = build_one_body(interactions_one_body, L)
    ham = ham_two + ham_one

    return ham


# Hatano- Nelson
def build_HN(L, J, r, pbc=False):
    """
    builds tfim Hamiltonian
    """
    ## Operator lists
    raise NotImplementedError
    if pbc == True:
        J_pm = [[J + r, i, (i + 1) % L] for i in range(L)]  # PBC
        J_mp = [[J - r, i, (i + 1) % L] for i in range(L)]  # PBC
    else:
        J_pm = [[J + r, i, (i + 1) % L] for i in range(L - 1)]  # OBC
        J_mp = [[J - r, i, (i + 1) % L] for i in range(L - 1)]  # OBC

    interactions_two_body = J_xx + V_xx + V_zz
    ham = build_two_body(interactions_two_body, L)

    return ham


if __name__ == '__main__':
    ham = build_XXZ_scipy(4, 1, 1, pbc=False)
    print(ham.todense())
    print(np.linalg.eigvalsh(ham.todense()))
    ham = build_SDIsing(4, 1, 1, 1, pbc=False)
    print(ham.todense())
    print(np.linalg.eigvalsh(ham.todense()))
    ham = build_TFIZ(4, 1, 1, 1, pbc=False)
    print(ham.todense())
    print(np.linalg.eigvalsh(ham.todense()))
