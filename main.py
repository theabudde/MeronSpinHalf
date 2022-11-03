import numpy as np
from MeronAlgorithmSpinHalfMassless import MeronAlgorithmSpinHalfMassless
from MeronAlgorithm import MeronAlgorithm
import time


def main():
    n = 10  # number of lattice points
    N = 10  # number of half time steps (#even + #odd)
    beta = 0.1  # beta
    mc_steps = 1000  # number of mc steps

    epsilon = beta / N
    t = 70
    m = 1
    u = - 2 / epsilon * np.log(np.cosh(epsilon * np.sqrt(m ** 2 + t ** 2)) - t / np.sqrt(m ** 2 + t ** 2) * np.sinh(
        epsilon * np.sqrt(m ** 2 + t ** 2)))

    w_a = np.exp(epsilon * u / 4)
    w_b = t / np.sqrt(m ** 2 + t ** 2) * np.sinh(epsilon * np.sqrt(m ** 2 + t ** 2)) * np.exp(-epsilon * u / 4)
    w_c = 0.5 * m / np.sqrt(m ** 2 + t ** 2) * np.sinh(epsilon * np.sqrt(m ** 2 + t ** 2)) * np.exp(-epsilon * u / 4)

    w_a = 0.5
    w_b = 0.5

    algorithm = MeronAlgorithmSpinHalfMassless(n, N, w_a, w_b, mc_steps)
    t0 = time.time()
    # print_nr = 100
    for mc in range(mc_steps):
        algorithm.mc_step()
    t1 = time.time()

    print('Code took:', t1 - t0)


if __name__ == "__main__":
    main()
