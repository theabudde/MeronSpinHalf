import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from astropy.stats import jackknife_stats


def main(argv):
    # argv parameters: U, t, beta, lattice_width, time_steps, mc_steps, result_path, N_files
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
    N_files = int(argv[8])

    file = os.path.join(result_path,
                        f'correlation_function_U={U}_t={t}_beta={beta}_L={lattice_width}_T={time_steps}_mcsteps={mc_steps}_job_id=')
    result = []
    corr_labels = ['corr_' + str(i) for i in range(lattice_width)]
    column_names = ['sign'] + corr_labels
    for i in range(N_files):
        data = pd.read_csv(file + str(i) + '.csv')
        sign = data['sign'].sum()
        correlations = np.zeros(lattice_width)
        for j in range(lattice_width):
            correlations[j] = data['corr_' + str(j)].sum()
        result.append(np.concatenate((np.array([sign]), correlations)))

    sums = pd.DataFrame(result, columns=column_names)
    sums.to_csv(os.path.join(result_path,
                             f'correlation_function_U={U}_t={t}_beta={beta}_L={lattice_width}_T={time_steps}_mcsteps={mc_steps * N_files}_sums.csv'))
    correlations = np.array(sums[corr_labels])
    sign = np.array(sums['sign'])
    correlations = correlations / np.transpose(np.array([sign for i in range(lattice_width)]))
    average = np.zeros(lattice_width)
    error = np.zeros(lattice_width)
    for i in range(lattice_width):
        jacknife_result = jackknife_stats(correlations[:, i], np.mean)
        average[i] = jacknife_result[0]
        error[i] = jacknife_result[2]
    averages = pd.DataFrame(np.transpose(np.array([average, error])), columns=['average', 'error'])
    averages.to_csv(os.path.join(result_path,
                                 f'correlation_function_U={U}_t={t}_beta={beta}_L={lattice_width}_T={time_steps}_mcsteps={mc_steps * N_files}_result.csv'))


if __name__ == "__main__":
    main(sys.argv)
