from MeronAlgorithmSpinHalfMassless import MeronAlgorithmSpinHalfMassless
import numpy as np
from itertools import product


class MeronAlgorithmSpinHalfMassive(MeronAlgorithmSpinHalfMassless):

    # w_a/(w_a + w_b) is the probability with which a vertical plaquette is placed
    # mass factor is the weight each cluster flip gets multiplied with per horizontal bond
    def __int__(self, n, t, w_a, w_b, mass_factor, mc_steps):
        self.mass_factor_per_bond = mass_factor
        # Weight of cluster flip for each cluster
        # will be numpy array after _find_clusters
        self.mass_factor = []
        MeronAlgorithmSpinHalfMassless.__init__(self, n, t, w_a, w_b, mc_steps)

    def _find_clusters(self):
        visited = np.full((self.n, self.t), False)  # record if site has been visited
        # counter for how many clusters there are -1 and the ID given to each of the clusters
        cluster_nr = 0
        # Order like this to ensure top left position is encountered first
        for y, x in product(range(self.t), range(self.n)):
            if not visited[x, y]:  # if you haven't seen the loop before
                new_coordinates = (x, y)
                start_coordinates = new_coordinates
                self.cluster_positions[cluster_nr] = new_coordinates
                leftness = 0
                max_leftness = 0
                self.mass_factor.append(0)
                while True:
                    visited[new_coordinates] = True
                    self.cluster_id[new_coordinates] = cluster_nr  # give cluster its ID
                    previous_coordinates = new_coordinates
                    new_coordinates = self._cluster_loop_step(new_coordinates)

                    if (previous_coordinates[0] - 1) % self.n == new_coordinates[0]:
                        leftness += 1
                        if leftness > max_leftness:
                            max_leftness = leftness
                            self.cluster_positions[cluster_nr] = new_coordinates
                    elif (previous_coordinates[0] + 1) % self.n == new_coordinates[0]:
                        leftness -= 1
                    elif (previous_coordinates[1] - 1) % self.t == new_coordinates[1] and leftness == max_leftness and \
                            new_coordinates[1] == (self.cluster_positions[cluster_nr][1] - 1) % self.t:
                        self.cluster_positions[cluster_nr] = new_coordinates

                    if (previous_coordinates[1] - 1) % self.t == new_coordinates[1] or (
                            previous_coordinates[1] + 1) % self.t == new_coordinates[1]:
                        self.mass_factor[cluster_nr] += 1

                    if new_coordinates == (x, y):
                        break
                # look where to find next cluster
                cluster_nr += 1
        self.n_clusters = cluster_nr
        self.flip = np.zeros(self.n_clusters)
        self.mass_factor = self.mass_factor_per_bond * np.array(self.mass_factor)

    def charge_automaton(self, row, charge_index, case_character):
        next_row, arrow_weight = MeronAlgorithmSpinHalfMassless.charge_automaton(row, charge_index, case_character)
        if case_character == 3:
            arrow_weight *= self.mass_factor[self.charged_cluster_order[charge_index]]
        return next_row, arrow_weight

    def _calculate_neutral_combinations(self, start_cluster, plus_minus):
        result = np.array([0.0, 0.0])
        # multiply out product for clusters in the same level
        for i in range(len(self.cluster_order[start_cluster])):
            result = (result + 1.0) * (
                    self._calculate_neutral_combinations(self.cluster_order[start_cluster][i],
                                                         not plus_minus) * self.mass_factor[
                        self.cluster_order[start_cluster][i]] + 1.0) - 1.0
        # calculate the effect of the loop
        if self.cluster_charge[start_cluster] == 0:
            if plus_minus:
                result = np.array([result[0] + result[1] + 1.0, result[1]])
            else:
                result = np.array([result[0], result[0] + result[1] + 1.0])
        # save result
        if start_cluster >= 0:
            self.cluster_combinations[start_cluster] = result
        return result
