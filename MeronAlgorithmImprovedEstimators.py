from MeronAlgorithm import MeronAlgorithm
import numpy as np


class MeronAlgorithmImprovedEstimators(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, mc_steps)
        self.is_meron = np.array([])
        self.n_merons = -1
        self.improved_two_point_function = 0
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

    def _two_point_function(self, site_1, site_2):
        if self.n_merons == 2 and self.is_meron[self.cluster_id[site_1, 0]] \
                and self.is_meron[self.cluster_id[site_2, 0]]:
            weight = 1 / 4
        elif self.n_merons == 0 and self.cluster_id[site_1, 0] == self.cluster_id[site_2, 0]:
            weight = 1 / 4
        else:
            weight = 0
        return weight

    def _config_sign(self):
        if self.n_merons == 0:
            return 1
        return 0

    def calculate_improved_two_point_function(self, site_1, site_2):
        for i in range(self.mc_steps):
            self.mc_step()
            self._calculate_merons()
            self.improved_two_point_function += self._two_point_function(site_1, site_2)
            self.improved_sign += self._config_sign()
        return self.improved_two_point_function / self.improved_sign
