import numpy as np


def Ham(t, m, U, L):
    Hamiltonian = np.zeros((2 ** L, 2 ** L), complex)
    for i in range(2 ** L):
        fermion_string = construct_fermion_string(i, L)
        for x in range(L):
            Hamiltonian[i, i] += -m * fermion_string[x] * (-1) ** x + U * (fermion_string[x] - .5) * (
                        fermion_string[(x + 1) % L] - .5)
            if fermion_string[x]:
                if 1 - fermion_string[(x - 1) % L]:
                    j = binary_string_to_int(hop(x, (x - 1) % L, fermion_string, L))
                    Hamiltonian[j, i] += -t * (1 - 2 * (x == 0))
                if 1 - fermion_string[(x + 1) % L]:
                    j = binary_string_to_int(hop(x, (x + 1) % L, fermion_string, L))
                    Hamiltonian[j, i] += -t * (1 - 2 * (x == L - 1))
    return Hamiltonian


def hop(i_particle, i_hole, st, L):
    i_0 = min([i_particle, i_hole])
    i_f = max([i_particle, i_hole])
    new = list(st[:i_0]) + [1 - st[i_0]] + list(st[i_0 + 1:i_f]) + [1 - st[i_f]] + list(st[i_f + 1:])
    return np.array(new)


def binary_string_to_int(fst):
    s = 0
    for i, x in enumerate(fst[::-1]):
        s += x * (2 ** i)
    return s


def construct_fermion_string(n, L):
    string_st = bin(n)[2:]
    return np.array([int(x) for x in '0' * (L - len(string_st)) + string_st])
