from MeronAlgorithm import MeronAlgorithm
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random


class MeronAlgorithmNoCondition(MeronAlgorithm):

    def __init__(self, n, t, w_a, w_b, w_c, beta, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, w_c, beta, mc_steps)
        # number of times a neutral cluster wraps horizontally
        self.horizontal_winding = np.array([0])
        self.horizontal_winding_order = []
        self.horizontally_winding_clusters_exist = False

        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

    def _identify_horizontal_winding(self):
        # determine cluster's winding
        self.horizontal_winding = np.zeros(self.n_clusters)
        for i in range(self.t):
            if i % 2:
                self.horizontal_winding[self.cluster_id[0, i]] += 1
            else:
                self.horizontal_winding[self.cluster_id[0, i]] -= 1

        # determine order of winding clusters
        for i in range(self.t):
            if self.horizontal_winding[self.cluster_id[0, i]] != 0 and not self.cluster_id[
                                                                               0, i] in self.charged_cluster_order:
                self.horizontal_winding_order.append(self.cluster_id[0, i])
                self.cluster_positions[self.cluster_id[0, i]] = (0, i)
                self.horizontally_winding_clusters_exist = True

    def correct_top_position_of_horizontally_winding_clusters(self):
        # correct the topmost position of topmost horizontally winding cluster
        y = self.cluster_positions[self.charged_cluster_order[-1]][1]
        while self.cluster_id[0, y] != self.charged_cluster_order[0]:
            y = (y + 1) % self.t
        self.cluster_positions[self.charged_cluster_order[1]] = (0, y)

    def _assign_groups_only_neutrals(self):
        self.cluster_order[-2] = []
        start_cluster = 0
        while True:
            outer_cluster = -2
            left_neighbors = self._find_left_neighbors(start_cluster)
            for neighbor in left_neighbors:
                if self._has_as_right_neighbor(neighbor, start_cluster):
                    outer_cluster = neighbor
            for neighbor in left_neighbors:
                if not neighbor == outer_cluster:
                    if self.cluster_group[neighbor] == - 1:
                        self.cluster_group[neighbor] = outer_cluster
                        self.cluster_order[outer_cluster].append(neighbor)
                        self._order_neighboring_clusters(neighbor, outer_cluster, neighbor)
            if outer_cluster == - 2:
                break
            start_cluster = outer_cluster

    def _find_left_neighbors(self, start_cluster_id):
        left_neighbors = []
        start_position = self.cluster_positions[start_cluster_id]
        position = start_position
        while True:
            position = self._cluster_loop_step_new(position)

            x = position[0]
            y = position[1]
            cluster_up = self.cluster_id[x, (y - 1) % self.t]
            cluster_right = self.cluster_id[(x + 1) % self.n, y]
            cluster_down = self.cluster_id[x, (y + 1) % self.t]
            cluster_left = self.cluster_id[(x - 1) % self.n, y]

            for neighbor in [cluster_up, cluster_right, cluster_down, cluster_left]:
                if neighbor != start_cluster_id:
                    if self.cluster_positions[neighbor][0] % 2 == self.cluster_positions[start_cluster_id][0] % 2:
                        # is a left neighbor
                        if neighbor not in left_neighbors:
                            left_neighbors.append(neighbor)
            if position == start_position:
                break
        return left_neighbors

    # Has to be direct neighbor already, this only checks whether this is a right and not a left neighbor
    def _has_as_right_neighbor(self, cluster, potential_right_neighbor_of_cluster):
        if cluster == potential_right_neighbor_of_cluster:
            return False
        if self.cluster_positions[cluster][0] % 2 != self.cluster_positions[potential_right_neighbor_of_cluster][0] % 2:
            return True
        return False

    def _charge_automaton(self, row, charge_index, case_character):
        next_row = -1
        arrow_weight = 0
        charge = self.cluster_charge[self.charged_cluster_order[charge_index]]
        pm_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][0]
        mp_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][1]
        match row:
            case 0:
                if case_character == 0:
                    if charge_index == len(self.charged_cluster_order) - 1 or charge_index == 0:
                        next_row = -1
                    else:
                        next_row = 1
                    if charge > 0:
                        arrow_weight = pm_combinations + 1
                    else:
                        arrow_weight = mp_combinations + 1
                elif case_character == 1:
                    if charge_index == 0:
                        next_row = -1
                    elif charge > 0:
                        next_row = 0
                        arrow_weight = mp_combinations + 1
                    else:
                        next_row = 0
                        arrow_weight = pm_combinations + 1
            case 1:
                if charge_index == 0:
                    next_row = -1
                elif case_character == 0:
                    next_row = 0
                    if charge > 0:
                        arrow_weight = mp_combinations + 1
                    else:
                        arrow_weight = pm_combinations + 1
                else:
                    next_row = -1
            case 2:
                match case_character:
                    case 0:
                        next_row = 2
                        arrow_weight = 1
                    case 1:
                        if charge_index == len(self.charged_cluster_order) - 1:
                            next_row = 2
                            arrow_weight = pm_combinations
                        elif charge > 0:
                            next_row = 1
                            arrow_weight = pm_combinations
                        else:
                            next_row = 0
                            arrow_weight = pm_combinations
                    case 2:
                        if charge_index == len(self.charged_cluster_order) - 1:
                            next_row = 2
                            arrow_weight = mp_combinations
                        elif charge > 0:
                            next_row = 3
                            arrow_weight = mp_combinations
                        else:
                            next_row = 4
                            arrow_weight = mp_combinations
                    case 3:
                        if charge_index == len(self.charged_cluster_order) - 1:
                            next_row = -1
                        elif charge > 0:
                            next_row = 0
                            arrow_weight = mp_combinations + 1
                        else:
                            next_row = 3
                            arrow_weight = pm_combinations + 1
            case 3:
                if charge_index == 0:
                    next_row = -1
                elif case_character == 0:
                    if charge_index == len(self.charged_cluster_order) - 2:
                        next_row = -1
                    elif charge_index == len(self.charged_cluster_order) - 1:
                        next_row = 3
                        arrow_weight = mp_combinations + 1
                    elif charge > 0:
                        next_row = 4
                        arrow_weight = pm_combinations + 1
                    else:
                        next_row = 4
                        arrow_weight = mp_combinations + 1
                else:
                    if charge_index == len(self.charged_cluster_order) - 1:
                        next_row = -1
                    elif charge > 0:
                        next_row = 3
                        arrow_weight = mp_combinations + 1
                    else:
                        next_row = 3
                        arrow_weight = pm_combinations + 1
            case 4:
                if charge_index == 0 or charge_index == 0 or charge_index == len(self.charged_cluster_order) - 1:
                    next_row = -1
                elif case_character == 0:
                    next_row = 3
                    if charge > 0:
                        arrow_weight = mp_combinations + 1
                    else:
                        arrow_weight = pm_combinations + 1
                else:
                    next_row = -1
        return next_row, arrow_weight

    def mc_step(self):
        n_flip_configs = 1
        seed = 1
        # for seed in range(0, 10000):
        # random.seed(seed)

        # reset to reference config
        self._reset()

        # place new bonds
        self._assign_bonds()

        self.draw_bonds()

        # find clusters
        self._find_clusters()

        self.draw_bonds()

        self._identify_charged_clusters()
        self._identify_horizontal_winding()

        # optional: run tests to verify hypothesis of cluster structure (very slow)
        # self.tests(seed)

        histogram = np.zeros(2 ** self.n_clusters)

        if self.charged_clusters_exist:
            self._correct_positions()

            self._assign_groups()
            # draw config for debug
            self.draw_bonds()
            # calculate the cluster combinations
            for charge in self.charged_cluster_order:
                if self.cluster_charge[charge] > 0:
                    self._calculate_neutral_combinations(charge, True)
                else:
                    self._calculate_neutral_combinations(charge, False)
            self._calculate_charge_combinations()
            for i in range(n_flip_configs):
                self.flip = np.zeros(self.n_clusters)
                self._generate_flips()
                self._flip()
                self.draw_bonds()
                self._calculate_gauge_field()
                histogram[int("".join(str(int(k)) for k in self.flip), 2)] += 1

        elif self.horizontally_winding_clusters_exist:
            self.draw_bonds()
            pass
            raise NotImplementedError('horizontal winding')
            # TODO: look at ordering, what constraints does a horizontal charge give?
        else:
            self._correct_positions()
            self._assign_groups_only_neutrals()
            self.draw_bonds()
            plus_minus = self.cluster_positions[self.cluster_order[-2][0]][0] % 2
            total_combinations = self._calculate_neutral_combinations(-2, not plus_minus)
            if plus_minus:
                total_combinations[1] -= total_combinations[0] + 1
            else:
                total_combinations[0] -= total_combinations[1] + 1
            p_plus_minus = (total_combinations[0] + 1) / (total_combinations[0] + total_combinations[1] + 2)
            for i in range(n_flip_configs):
                while True:
                    self.flip = np.zeros(self.n_clusters)
                    if random.random() < p_plus_minus:
                        charge = -1
                        self._generate_neutral_flips(-2, charge, plus_minus)
                    else:
                        charge = 1
                        self._generate_neutral_flips(-2, charge, plus_minus)
                    if not int("".join(str(int(k)) for k in self.flip)) == 0 or random.random() < 0.5:
                        break
                histogram[int("".join(str(int(k)) for k in self.flip), 2)] += 1
        plt.plot(histogram, ".")
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid()
        # plt.show()

        # draw config for debug
        self.draw_bonds()

        pass
