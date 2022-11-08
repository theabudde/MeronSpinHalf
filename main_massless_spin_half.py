import numpy as np
from MeronAlgorithmSpinHalfMassless import MeronAlgorithmSpinHalfMassless
from MeronAlgorithm import MeronAlgorithm
from MeronAlgorithmImprovedEstimators import MeronAlgorithmImprovedEstimators
import time
import sys


def main():
    mc_steps = 100000  # number of mc steps
    n = 8  # number of lattice points
    U = 1

    original_stdout = sys.stdout
    with open('correlation_function.txt', 'w') as f:
        f.write(f'mc_steps = {mc_steps}\n')
        f.write(f'n = {n}\n')
        f.write(f'U = {U}\n')
        f.write('N, beta, site, w_a, w_b, time, result\n')

    for N in [10, 100]:
        for beta in [0.1, 1, 10]:
            for site in range(1, 4):
                if beta / N > 0.15:
                    continue
                print('starting run N =', N, ' beta =', beta, ' site =', site)

                w_a = np.cosh(beta / N * U / 2)
                w_b = np.sinh(beta / N * U / 2)

                algorithm = MeronAlgorithmSpinHalfMassless(n, N, w_a, w_b, mc_steps)
                t0 = time.time()
                result = algorithm.improved_two_point_function(0, site)
                t1 = time.time()

                with open('correlation_function.txt', 'a') as f:
                    f.write(f'{N}, {beta}, {site}, {w_a}, {w_b}, {t1 - t0}, {result} \n')


if __name__ == "__main__":
    main()
