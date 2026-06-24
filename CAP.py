import numpy as np

def CAP(Cin, Px, Mf, mode="auto"):
    n = np.log2(Mf.shape[0])
    m = np.log2(Mf.shape[1])
    copula_func = get_copula_func(Cin, Px, mode=mode)
    pin = get_MV_from_copula(copula_func, Px)
    Qn = get_Q(n)
    Qm = get_Q(m)
    pout = Qm @ Mf @ np.linalg.inv(Qn) @ pin
    return get_C_and_Px_from_MV(pout, m)

def get_copula_func(Cin, Px, mode="auto"):
    pass

def get_MV_from_copula(copula_func, Px):
    n = len(Px)
    pin = np.empty((2 ** n, ))
    for i in range(2 ** n):
        Px_star = np.empty((n, ))
        b = binary_vector(i, n)
        for j in range(n):
            Px_star[j] = Px[n-1-j] if b[j] else 1
        pin[i] = copula_func(*Px_star)
    return pin

def get_Q(n):
    Q0 = np.array([[1, 1], [0, 1]])
    Qn = Q0
    for _ in range(1, n):
        Qn = np.kron(Qn, Q0)
    return Qn

def get_C_and_Px_from_MV(p, n):
    C = np.zeros((n, n))
    Px = np.zeros((n,))
    for i in range(n):
        p_idx_i = 2 ** (n - 1 - i) #switch to 2 ** n for P(X_2, X_1, X_0 = 0,0,1) indexing
        pi = p[p_idx_i]
        Px[i] = pi
        for j in range(n):
            p_idx_j = 2 ** (n - 1 - j)
            p_idx_ij = p_idx_i + p_idx_j
            pj = p[p_idx_j]
            pij = p[p_idx_ij]
            C[i, j] = scc(pi, pj, pij)
    return C, Px

def scc(px, py, pxy):
    cov = pxy - px * py
    if cov > 0:
        return cov / (min(px, py) - px * py)
    else:
        return cov / (px * py - max(px + py - 1, 0))

def binary_vector(value, m):
    """Convert a positive integer value into an m-bit bit vector"""
    arr = np.array(
        list(np.binary_repr(value).zfill(m))
    ).astype(bool)

    return arr