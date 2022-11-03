from MeronAlgorithm import MeronAlgorithm
import numpy as np


class MeronAlgorithmImprovedEstimators(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, beta, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, beta, mc_steps)
        self.is_meron = np.zeros(self.n_clusters)

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

    # Does this work here? 0 for j odd k even more interesting for both same.
    def _two_point_function(self, site_1, site_2):
        if self.cluster_id[site_1] == self.cluster_id[site_2]:
            return self.fermion[site_1] * self.fermion[site_2]
