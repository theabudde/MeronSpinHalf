import os

import numpy as np
import pandas as pd

from MeronAlgorithmSpinHalfBruteForce import MeronAlgorithmSpinHalfBruteForce
import sys


def main(argv):
    # argv parameters: U, t, beta, lattice_width, time_steps, mc_steps, result_path, job_array_id
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
    if mc_steps % 10 != 0:
        raise ValueError('error can only be calculated if mc steps is a multiple of 10')
    result_path = argv[7]
    job_array_nr = argv[8]

    eps = beta / time_steps
    if eps > 0.15:
        return

    w_a = np.exp(- eps * U / 4)
    w_b = np.exp(eps * U / 4) * np.sinh(eps * t)

    # Initialise to reference configuation
    algorithm = MeronAlgorithmSpinHalfBruteForce(lattice_width, 2 * time_steps, w_a, w_b, mc_steps)

    # Thermalise
    algorithm.correlation_function(1000)

    # calculate two point function
    algorithm.correlation_function(mc_steps)

    # output result to csv file
    output_path = os.path.join(result_path,
                               f'correlation_function_U={U}_t={t}_beta={beta}_L={lattice_width}_T={time_steps}.csv')
    columns = np.array(['n_steps'] + [f'corr_{i}' for i in range(lattice_width)])
    data = np.concatenate((np.array([mc_steps]), algorithm.correlation))
    data = pd.DataFrame(np.array([data]), columns=columns)
    data.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


if __name__ == "__main__":
    main(sys.argv)
