from MeronAlgorithm import MeronAlgorithm
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random


class MeronAlgorithmSpinHalfMassless(MeronAlgorithm):

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
        self.horizontal_winding_order = []
        self.horizontally_winding_clusters_exist = False
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
                                                                               0, i] in self.horizontal_winding_order:
                self.horizontal_winding_order.append(self.cluster_id[0, i])
                self.cluster_positions[self.cluster_id[0, i]] = (0, i)
                self.horizontally_winding_clusters_exist = True

    def correct_top_position_of_horizontally_winding_clusters(self):
        # correct the topmost position of topmost horizontally winding cluster
        y = self.cluster_positions[self.charged_cluster_order[-1]][1]
        while self.cluster_id[0, y] != self.charged_cluster_order[0]:
            y = (y + 1) % self.t
        self.cluster_positions[self.charged_cluster_order[1]] = (0, y)

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
        # seed = 1
        # for seed in range(0, 10000):
        # random.seed(seed)

        # reset to reference config
        self._reset()

        # place new bonds
        self._assign_bonds()

        # find clusters
        self._find_clusters()

        self._identify_charged_clusters()
        self._identify_horizontal_winding()

        if self.charged_clusters_exist or self.horizontally_winding_clusters_exist:
            if self.horizontally_winding_clusters_exist and not self.charged_clusters_exist:
                self.charged_cluster_order = self.horizontal_winding_order
                self.cluster_charge = self.horizontal_winding
            self._assign_groups_with_charges()

            # calculate the cluster combinations
            for charge in self.charged_cluster_order:
                if self.cluster_charge[charge] > 0:
                    self._calculate_neutral_combinations(charge, True)
                else:
                    self._calculate_neutral_combinations(charge, False)
            self._calculate_charge_combinations()
            self.flip = np.zeros(self.n_clusters)
            self._generate_flips()
            self._flip()
            self._calculate_gauge_field()

        elif not self.horizontally_winding_clusters_exist and not self.charged_clusters_exist:
            self._assign_groups_only_neutrals(0)
            if self.n_clusters > 1:
                plus_minus = self.cluster_positions[self.cluster_order[-2][0]][0] % 2
                total_combinations = self._calculate_neutral_combinations(-2, not plus_minus)
                if plus_minus:
                    total_combinations[1] -= total_combinations[0] + 1
                else:
                    total_combinations[0] -= total_combinations[1] + 1
                p_plus_minus = (total_combinations[0] + 1) / (total_combinations[0] + total_combinations[1] + 2)
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
            else:
                if random.random() < 0.5:
                    self.flip[0] = 1
                else:
                    self.flip[0] = 0

        # plt.plot(histogram, ".")
        # plt.ylim(bottom=0)
        # plt.xlim(left=0)
        # plt.grid()
        # plt.show()

        self._flip()
        self._calculate_gauge_field()
        self._test_gauss_law()

        pass
