from MeronAlgorithm import MeronAlgorithm
import numpy as np


class MeronAlgorithmImprovedEstimators(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, mc_steps)
        self.is_meron = np.zeros(self.n_clusters)
        self.n_merons = -1

    def _calculate_merons(self):
        for cluster in range(self.n_clusters):
            position = self.cluster_positions[cluster]
            start_position = position
            previous_position = position
            meronness = 0
            while True:
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
        if self.n_merons == 2 and self.is_meron[self.cluster_id[site_1]] and self.is_meron[self.cluster_id[site_2]]:
            weight = 1 / 4
        elif self.n_merons == 0 and self.cluster_id[site_1] == self.cluster_id[site_2]:
            weight = 1 / 4
        else:
            weight = 0
        return weight

    def _config_sign(self):
        if self.n_merons == 0:
            return 1
        return 0

    def calculate_two_point_function(self, site_1, site_2):
        two_point_function = 0
        sign = 0
        for i in range(self.mc_steps):
            self.mc_step()
            self._calculate_merons()
            two_point_function += self._two_point_function(site_1, site_2)
            sign += self._config_sign()
            if i % 1024 == 0:
                print('<(n_0-1/2)(n_i-1/2)(-1)^i> =', two_point_function / sign, ' sign =', sign)


