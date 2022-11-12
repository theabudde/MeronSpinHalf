import numpy as np
from MeronAlgorithmImprovedEstimators import MeronAlgorithmImprovedEstimators
import time
import scipy
import pandas as pd
import sys


def main(argv):
    # argv parameters: U, t, beta, lattice_width, time_steps, mc_steps, result_path
    U = float(argv[1])
    t = float(argv[2])
    beta = float(argv[3])
    lattice_width = int(argv[4])
    if lattice_width % 2 != 0:
        raise ValueError('lattice can not have an odd number of sites')
    if lattice_width < 4:
        raise ValueError('Algorithm only works for grid sizes 4 and higher')
    time_steps = int(argv[5])
    mc_steps = int(argv[6])
    if mc_steps % 100 != 0:
        raise ValueError('error can only be calculated if mc steps is a multiple of 100')
    result_path = argv[7]

    eps = beta / time_steps
    if eps > 0.15:
        return

    w_a = np.exp(- eps * U / 4)
    w_b = np.exp(eps * U / 4) * np.sinh(eps * t)

    algorithm = MeronAlgorithmImprovedEstimators(lattice_width, 2 * time_steps, w_a, w_b, mc_steps)
    for i in range(1000):
        algorithm.mc_step()
    algorithm.produce_data(U, t, beta, lattice_width, time_steps, mc_steps, result_path)


if __name__ == "__main__":
    main(sys.argv)
