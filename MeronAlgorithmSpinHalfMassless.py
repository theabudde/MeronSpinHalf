import os
import pickle

from MeronAlgorithmWithAGaussLaw import MeronAlgorithmWithAGaussLaw
import numpy as np
from itertools import product
import random
import matplotlib.pyplot as plt


class MeronAlgorithmSpinHalfMassless(MeronAlgorithmWithAGaussLaw):
    def __init__(self, n, t, w_a, w_b, mc_steps, data_file_path, job_array_nr):
        MeronAlgorithmWithAGaussLaw.__init__(self, n, t, w_a, w_b, mc_steps)
        self.data_file_path = data_file_path
        self.job_array_nr = job_array_nr

    def _reset(self):
        MeronAlgorithmWithAGaussLaw._reset(self)
        self.horizontal_winding = np.array([0])
        self.horizontal_winding_order = []
        self.horizontally_winding_clusters_exist = False
        # reset fermions
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True
                self.fermion[2 * i + 1, j] = False

    def _charge_automaton(self, row, charge_index, case_character):
        next_row = -1
        arrow_weight = 0
        charge = self.cluster_charge[self.charged_cluster_order[charge_index]]
        pm_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][0]
        mp_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][1]
        match row:
            case 0:
                if case_character == 0:
                    if charge_index == 0:
                        next_row = -1
                    else:
                        next_row = 1
                    if charge > 0:
                        arrow_weight = mp_combinations + 1
                    else:
                        arrow_weight = pm_combinations + 1
            case 1:
                if charge_index == 0:
                    next_row = -1
                elif case_character == 0 and not charge_index == len(self.charged_cluster_order) - 1:
                    next_row = 0
                    if charge > 0:
                        arrow_weight = pm_combinations + 1
                    else:
                        arrow_weight = mp_combinations + 1
                elif case_character == 1:
                    next_row = 1
                    if charge > 0:
                        arrow_weight = mp_combinations + 1
                    else:
                        arrow_weight = pm_combinations + 1
            case 2:
                match case_character:
                    case 0:
                        next_row = 2
                        arrow_weight = 1
                    case 1:
                        if charge > 0:
                            next_row = 4
                        else:
                            next_row = 3
                        arrow_weight = pm_combinations
                    case 2:
                        if charge > 0:
                            next_row = 1
                        else:
                            next_row = 0
                        arrow_weight = mp_combinations
                    case 3:
                        if charge_index == len(self.charged_cluster_order) - 1:
                            next_row = -1
                        elif charge > 0:
                            next_row = 3
                            arrow_weight = mp_combinations + 1
                        else:
                            next_row = 1
                            arrow_weight = pm_combinations + 1
            case 3:
                if charge_index == 0:
                    next_row = -1
                elif case_character == 0:
                    if charge_index == len(self.charged_cluster_order) - 2:
                        next_row = -1
                    elif charge > 0:
                        next_row = 4
                        arrow_weight = pm_combinations + 1
                    else:
                        next_row = 4
                        arrow_weight = mp_combinations + 1
                elif case_character == 1:
                    if charge_index == len(self.charged_cluster_order) - 1:
                        next_row = -1
                    elif charge > 0:
                        next_row = 3
                        arrow_weight = mp_combinations + 1
                    else:
                        next_row = 3
                        arrow_weight = pm_combinations + 1
            case 4:
                if charge_index == 0 or charge_index == 1 or charge_index == len(self.charged_cluster_order) - 1:
                    next_row = -1
                elif case_character == 0:
                    next_row = 3
                    if charge > 0:
                        arrow_weight = mp_combinations + 1
                    else:
                        arrow_weight = pm_combinations + 1
        return next_row, arrow_weight

    def _calculate_neutral_combinations(self, start_cluster, plus_minus):
        result = np.array([0.0, 0.0])
        # multiply out product for clusters in the same level
        for i in range(len(self.cluster_order[start_cluster])):
            result = (result + 1.0) * (
                    self._calculate_neutral_combinations(self.cluster_order[start_cluster][i],
                                                         not plus_minus) + 1.0) - 1.0
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

    def _generate_neutral_flips(self, charged_cluster, boundary_charge, plus_minus):
        for cluster in self.cluster_order[charged_cluster]:
            if np.array_equal(self.cluster_combinations[cluster], np.array([1.0, 0.0])):
                if boundary_charge < 0:
                    self.flip[cluster] = random.random() < 0.5
            elif np.array_equal(self.cluster_combinations[cluster], np.array([0.0, 1.0])):
                if boundary_charge > 0:
                    self.flip[cluster] = random.random() < 0.5
            elif boundary_charge < 0:
                if random.random() < (self.cluster_combinations[cluster][1] + 1) / (
                        self.cluster_combinations[cluster][0] + 1) and plus_minus:
                    self.flip[cluster] = 1
                    self._generate_neutral_flips(cluster, - boundary_charge, not plus_minus)
                else:
                    self._generate_neutral_flips(cluster, boundary_charge, not plus_minus)
            else:
                if random.random() < (self.cluster_combinations[cluster][0] + 1) / (
                        self.cluster_combinations[cluster][1] + 1) and not plus_minus:
                    self.flip[cluster] = 1
                    self._generate_neutral_flips(cluster, - boundary_charge, not plus_minus)
                else:
                    self._generate_neutral_flips(cluster, boundary_charge, not plus_minus)

    def _flips_are_zero(self, charged_cluster):
        result = 0
        for cluster in self.cluster_order[charged_cluster]:
            if self.flip[cluster] == 1:
                return 1
            result += self._flips_are_zero(cluster)
        return result > 0

    def _generate_neutral_flips_no_zero(self, charged_cluster, boundary_charge, plus_minus):
        self._generate_neutral_flips(charged_cluster, boundary_charge, plus_minus)
        if not self._flips_are_zero(charged_cluster) and not np.any(self.flip[self.cluster_order[charged_cluster]]):
            self._generate_neutral_flips_no_zero(charged_cluster, boundary_charge, plus_minus)

    def _calculate_charge_combinations(self):
        self.charge_combinations = np.full((5, len(self.charged_cluster_order) + 1, 4), 0.0, dtype=np.float64)

        # indexed by: [row, charged_cluster_order_id, case]

        # case: 0: don't flip anything
        #       1: flip only +- neutrals
        #       2: flip only -+ neutrals
        #       3: flip charge and optionally the neutrals

        # row 0 is line starting with +
        # row 1 is skipping line for stuff starting with +
        # row 2 have not flipped anything yet, can go either way
        # row 3 is line starting with -
        # row 4 is skipping line starting with -

        # define legal final states
        self.charge_combinations[1, -1, 0] = 1.0
        self.charge_combinations[2, -1, 0] = 1.0
        self.charge_combinations[4, -1, 0] = 1.0

        for charge_index in range(len(self.charged_cluster_order) - 1, -1, -1):
            for row in range(5):
                if row == 2:
                    for case_character in range(4):
                        next_row, weight = self._charge_automaton(row, charge_index, case_character)
                        if not next_row == -1:
                            self.charge_combinations[row, charge_index, case_character] += float(weight) * float(np.sum(
                                self.charge_combinations[next_row, charge_index + 1]))
                else:
                    for case_character in range(2):
                        next_row, weight = self._charge_automaton(row, charge_index, case_character)
                        if not next_row == -1:
                            self.charge_combinations[row, charge_index, case_character] += weight * np.sum(
                                self.charge_combinations[next_row, charge_index + 1])

    def _generate_flips(self):
        row = 2
        for charge_idx in range(len(self.charged_cluster_order)):
            charge = self.charged_cluster_order[charge_idx]
            case_character = random.choices(range(4), weights=self.charge_combinations[row, charge_idx])[0]
            if row == 2:
                match case_character:
                    case 0:
                        self.flip[charge] = 0  # dont flip anything
                    case 1:
                        self.flip[charge] = 0
                        if self.cluster_combinations[charge][0] > 0:
                            self._generate_neutral_flips_no_zero(charge, -1, self.cluster_charge[charge] < 0)
                        else:
                            raise NotImplementedError
                    case 2:
                        self.flip[charge] = 0
                        if self.cluster_combinations[charge][1] > 0:
                            self._generate_neutral_flips_no_zero(charge, 1, self.cluster_charge[charge] < 0)
                        else:
                            raise NotImplementedError
                    case 3:
                        self.flip[charge] = 1
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
            elif row == 0 or row == 4:
                match case_character:
                    case 0:
                        self.flip[charge] = 0
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                    case 1:
                        self.flip[charge] = 1
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
            elif row == 1 or row == 3:
                match case_character:
                    case 0:
                        self.flip[charge] = 0
                        self._generate_neutral_flips(charge, - self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                    case 1:
                        self.flip[charge] = 1
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)

            row, weight = self._charge_automaton(row, charge_idx, case_character)

    def _generate_flips_no_charges(self, plus_minus, p_plus_minus):
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

    def _calculate_gauge_field(self):
        for x in range(self.n - 1):
            for y in range(1, self.t):
                if (y + 1) % 2 != x % 2 or self.fermion[x, y] == self.fermion[x, y - 1]:
                    self.gauge_field[x, y] = self.gauge_field[x, y - 1]
                elif not self.fermion[x, y]:
                    self.gauge_field[x, y] = self.gauge_field[x, y - 1] - 1
                else:
                    self.gauge_field[x, y] = self.gauge_field[x, y - 1] + 1
            if self.fermion[x + 1, 0] == x % 2:
                self.gauge_field[x + 1, 0] = self.gauge_field[x, 0]
            elif x % 2:
                self.gauge_field[x + 1, 0] = self.gauge_field[x, 0] - 1
            else:
                self.gauge_field[x + 1, 0] = self.gauge_field[x, 0] + 1

    def _test_gauss_law(self):
        if np.amax(self.gauge_field) - np.amin(self.gauge_field) > 1:
            with open(os.path.join(self.data_file_path, 'ErrorObjects/' + self.job_array_nr + '.pkl'),
                      'wb') as outp:  # Overwrites any existing file.
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            raise ValueError('gauss law broken, pkl should be in',
                             os.path.join(self.data_file_path, 'ErrorObjects/' + self.job_array_nr + '.pkl'))

    def reweight_factor_vertical_bonds(self):
        reweight_factor = 1
        normalizing_factor = self.w_a
        for x, y in product(range(self.n), range(self.t)):
            if self.bond[x, y] == 0:
                if self.fermion[x, y] == self.fermion[(x + 1) % self.n, y] \
                        and self.fermion[x, (y + 1) % self.t] == self.fermion[(x + 1) % self.n, (y + 1) % self.t]:
                    reweight_factor *= self.w_a / normalizing_factor
                elif self.fermion[x, y] == self.fermion[x, (y + 1) % self.t]:
                    if (self.fermion[x, y] and y % 2 == 0) or ((not self.fermion[x, y]) and y % 2 == 1):
                        # reweight_factor *= (self.w_a - 2 * self.w_c) / normalizing_factor
                        pass
                    else:
                        # reweight_factor *= (self.w_a + 2 * self.w_c) / normalizing_factor
                        pass
                else:
                    raise ("fermion got flipped wrong")
        return reweight_factor

    def improved_two_point_function(self, n_steps):
        result = np.zeros(self.n)
        for i in range(n_steps):
            self.mc_step()
            for site in range(self.n):
                if self.fermion[0, 0] == self.fermion[site, 0]:
                    result[site] += 1
                else:
                    result[site] -= 1
            if i % (n_steps // 100) == 0:
                print(f'i = {i}')
        for site in range(self.n):
            if site % 2:
                result[site] *= -1
        return result, n_steps

    def flip_histogram(self):
        if self.n_clusters > 10:
            return
        histogram = np.zeros(2 ** self.n_clusters)
        for i in range(2 ** self.n_clusters * 500):
            self.flip = np.zeros(self.n_clusters)
            self._generate_flips()
            histogram[int("".join(str(k) for k in self.flip.astype(int)), 2)] += 1
            self._flip()
            self._calculate_gauge_field()
            self._test_gauss_law()
            # reset fermions
            for i in range(self.n // 2):
                for j in range(self.t):
                    self.fermion[2 * i, j] = True
                    self.fermion[2 * i + 1, j] = False
        plt.plot(histogram, 'x')
        plt.savefig('./Histograms/hist' + str(np.random.randint(0, 10000 * 2 ** self.n_clusters * 500)))
        plt.clf()
        self.flip = np.zeros(self.n_clusters)
        print('charged')
        pass

    def neutral_flip_histogram(self, plus_minus, p_plus_minus):
        if self.n_clusters > 10:
            return
        histogram = np.zeros(2 ** self.n_clusters)
        for i in range(2 ** self.n_clusters * 500):
            self.flip = np.zeros(self.n_clusters)
            self._generate_flips_no_charges(plus_minus, p_plus_minus)
            histogram[int("".join(str(k) for k in self.flip.astype(int)), 2)] += 1
            self._flip()
            self._calculate_gauge_field()
            self._test_gauss_law()
            # reset fermions
            for i in range(self.n // 2):
                for j in range(self.t):
                    self.fermion[2 * i, j] = True
                    self.fermion[2 * i + 1, j] = False
        plt.plot(histogram, 'x')
        plt.savefig('./Histograms/hist_' + str(np.random.randint(0, 10000 * 2 ** self.n_clusters * 500)))
        plt.clf()
        self.flip = np.zeros(self.n_clusters)
        print('neutral')
        pass

    def mc_step(self):
        # reset to reference config
        self._reset()

        # place new bonds
        self._assign_bonds()

        # find clusters
        self._find_clusters()
        self._set_sizes_of_arrays()

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
            # self.flip_histogram()
            self.flip = np.zeros(self.n_clusters)
            self._generate_flips()

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
                # self.neutral_flip_histogram(plus_minus, p_plus_minus)
                self._generate_flips_no_charges(plus_minus, p_plus_minus)
            else:
                if random.random() < 0.5:
                    self.flip[0] = 1
                else:
                    self.flip[0] = 0

        self._flip()
        self._calculate_gauge_field()
        self._test_gauss_law()
        # self.draw_bonds()
        # input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
