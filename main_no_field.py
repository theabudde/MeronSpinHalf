import numpy as np
from MeronAlgorithmSpinHalfMassless import MeronAlgorithmSpinHalfMassless
from MeronAlgorithm import MeronAlgorithm
from MeronAlgorithmImprovedEstimators import MeronAlgorithmImprovedEstimators
import time
import scipy
import pandas as pd


def main():
    mc_steps = 100000  # number of mc steps
    n = 8  # number of lattice points
    U = 2

    t = U / 2

    for N in [20, 200]:
        for beta in [0.1, 1, 10, 100]:
            eps = 2 * beta / N
            if eps > 0.15:
                continue
            print('starting run N =', N, ' beta =', beta)

            w_a = np.exp(- eps * U / 4)
            w_b = np.exp(eps * U / 4) * np.sinh(eps * t)

            algorithm = MeronAlgorithmImprovedEstimators(n, N, w_a, w_b, mc_steps)
            # algorithm.produce_data(U, t, beta, n, N, mc_steps)
            algorithm.plot_data(U, t, beta, n, N, mc_steps)


if __name__ == "__main__":
    main()
