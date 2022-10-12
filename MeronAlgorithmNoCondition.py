from MeronAlgorithm import MeronAlgorithm
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random


class MeronAlgorithmNoCondition(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, beta, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, beta, mc_steps)
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
            same_cluster_level = []
            neighbors = self._find_left_neighbors(start_cluster)
            for neighbor in neighbors:
                if self._has_as_right_neighbor(neighbor, start_cluster):
                    outer_cluster = neighbor
                else:
                    same_cluster_level.append(neighbor)
            for neighbor in same_cluster_level:
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

        closed_loop = False
        start_position = self.cluster_positions[start_cluster_id]
        position = start_position
        previous_position = start_position

        while not closed_loop:
            position, closed_loop, direction, previous_position = self._cluster_loop_step(position, previous_position,
                                                                                          start_position)
            x = position[0]
            y = position[1]
            cluster_up = self.cluster_id[x, (y - 1) % self.t]
            cluster_right = self.cluster_id[(x + 1) % self.n, y]
            cluster_down = self.cluster_id[x, (y + 1) % self.t]
            cluster_left = self.cluster_id[(x - 1) % self.n, y]

            # check left (right) neighbors of cluster and mark them as being in the same (own_id) group
            # and recursively check their neighbors too
            if direction == 0:
                if cluster_left not in left_neighbors and cluster_left != start_cluster_id:
                    left_neighbors.append(cluster_left)
            elif direction == 1:
                if cluster_up not in left_neighbors and cluster_up != start_cluster_id:
                    left_neighbors.append(cluster_up)
            elif direction == 2:
                if cluster_right not in left_neighbors and cluster_right != start_cluster_id:
                    left_neighbors.append(cluster_right)
            elif direction == 3:
                if cluster_down not in left_neighbors and cluster_down != start_cluster_id:
                    left_neighbors.append(cluster_down)
        return left_neighbors

    def _has_as_right_neighbor(self, cluster, potential_right_neighbor_of_cluster):
        is_neighbor = False

        if cluster == potential_right_neighbor_of_cluster:
            return False

        closed_loop = False
        start_position = self.cluster_positions[cluster]
        position = start_position
        previous_position = start_position

        while not closed_loop:
            position, closed_loop, direction, previous_position = self._cluster_loop_step(position, previous_position,
                                                                                          start_position)
            x = position[0]
            y = position[1]
            cluster_up = self.cluster_id[x, (y - 1) % self.t]
            cluster_right = self.cluster_id[(x + 1) % self.n, y]
            cluster_down = self.cluster_id[x, (y + 1) % self.t]
            cluster_left = self.cluster_id[(x - 1) % self.n, y]

            # check left (right) neighbors of cluster and mark them as being in the same (own_id) group
            # and recursively check their neighbors too
            if direction == 0:
                if cluster_right == potential_right_neighbor_of_cluster:
                    is_neighbor = True
            elif direction == 1:
                if cluster_down == potential_right_neighbor_of_cluster:
                    is_neighbor = True
            elif direction == 2:
                if cluster_left == potential_right_neighbor_of_cluster:
                    is_neighbor = True
            elif direction == 3:
                if cluster_up == potential_right_neighbor_of_cluster:
                    is_neighbor = True
        return is_neighbor

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
        n_flip_configs = 100000
        seed = 1
        # for seed in range(0, 10000):
        random.seed(seed)

        # reset to reference config
        self._reset()

        # place new bonds
        self._assign_bonds()

        # find clusters
        self._find_clusters()

        self._identify_charged_clusters()
        self._identify_horizontal_winding()

        # optional: run tests to verify hypothesis of cluster structure (very slow)
        # self.tests(seed)

        histogram = np.zeros(2 ** self.n_clusters)

        if self.charged_clusters_exist:
            self.correct_left_position_of_charged_clusters()
            self._correct_left_positions_of_boundary_clusters()

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
                histogram[int("".join(str(int(k)) for k in self.flip), 2)] += 1

        elif self.horizontally_winding_clusters_exist:
            raise NotImplementedError('horizontal winding')
            # TODO: look at ordering, what constraints does a horizontal charge give?
        else:
            self._correct_positions()
            self._assign_groups_only_neutrals()
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
        plt.show()

        # draw config for debug
        self.draw_bonds()

        pass
