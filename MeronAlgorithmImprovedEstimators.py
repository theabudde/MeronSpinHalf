from MeronAlgorithm import MeronAlgorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class MeronAlgorithmImprovedEstimators(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, mc_steps)
        self.is_meron = np.array([])
        self.n_merons = -1
        self.improved_two_point_function = np.zeros(n)
        self.improved_sign = 0

    def _calculate_merons(self):
        self.is_meron = np.zeros(self.n_clusters)
        for cluster in range(self.n_clusters):
            position = self.cluster_positions[cluster]
            start_position = position
            meronness = 0
            while True:
                previous_position = position
                position = self._cluster_loop_step(position)
                if self.fermion[position] and position[1] == previous_position[1]:
                    meronness += 1
                elif position[1] == 0 and previous_position[1] == (-1) % self.t:
                    meronness += 1
                elif position[1] == (-1) % self.t and previous_position[1] == 0:
                    meronness += 1
                if start_position == position:
                    break
            if meronness % 2 == 0:
                self.is_meron[cluster] = 1
        self.n_merons = np.sum(self.is_meron)

    def _two_point_functions(self):
        weight = np.zeros(self.n)
        for site in range(self.n):
            if self.n_merons == 2 and self.is_meron[self.cluster_id[0, 0]] \
                    and self.is_meron[self.cluster_id[site, 0]] and not self.cluster_id[0, 0] == self.cluster_id[
                site, 0]:
                weight[site] = 1
            elif self.n_merons == 0 and self.cluster_id[0, 0] == self.cluster_id[site, 0]:
                weight[site] = 1
        return weight

    def _config_sign(self):
        if self.n_merons == 0:
            return 1
        return 0

    def calculate_improved_two_point_function(self, mc_steps):
        self.improved_two_point_function = np.zeros(self.n)
        self.improved_sign = 0
        for i in range(mc_steps):
            self.mc_step()
            self._calculate_merons()
            self.improved_two_point_function += self._two_point_functions()
            self.improved_sign += self._config_sign()
        return self.improved_sign, self.improved_two_point_function

    def produce_data(self, U, t, beta, n, N, mc_steps, output_path, job_array_id):
        for i in range(1000):
            self.mc_step()
        result = []
        for i in range(100):
            sign, two_point = self.calculate_improved_two_point_function(mc_steps // 100)
            result.append(np.concatenate((np.array([sign]), two_point)))
            data = pd.DataFrame(result,
                                columns=['sign', 'corr_0', 'corr_1', 'corr_2', 'corr_3', 'corr_4', 'corr_5', 'corr_6',
                                         'corr_7'])
            data.to_csv(os.path.join(output_path,
                                     f'correlation_function_U={U}_t={t}_beta={beta}_L={n}_T={N}_mcsteps={mc_steps}_job_id={job_array_id}.csv'))

    def plot_data(self, U, t, beta, n, N, mc_steps):
        data = pd.read_csv(f'correlation_function_U={U}_t={t}_beta={beta}_L={n}_T={N}_mcsteps={mc_steps}.csv')
        plt.errorbar(data.loc[:, 'site'], data.loc[:, 'average'], yerr=data.loc[:, 'error'], fmt='o--', capsize=3,
                     markersize=5,
                     markerfacecolor='none')
        plt.title(f'correlation_function_U={U}_t={t}_beta={beta}_L={n}_T={N}_mcsteps={mc_steps}.csv')
        plt.xlabel('distance')
        plt.ylabel('correlation')
        plt.show()
