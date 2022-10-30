import random
from MeronAlgorithm import MeronAlgorithm
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


class MeronAlgorithmOneFixedCharge(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, w_c, beta, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, w_c, beta, mc_steps)

    # Places vertical and horizontal bonds with probability corresponding to wa_ and  w_b
    def _assign_bonds(self):
        for x, y in product(range(self.n), range(self.t)):
            if y % 2 != x % 2:
                self.bond[x, y] = - 1
            elif x == self.n - 1 or x == 0 or x == 1:
                self.bond[x, y] = 0
            # all occupied or all unoccupied
            elif self.fermion[x, y] == self.fermion[(x + 1) % self.n, y] \
                    and self.fermion[x, (y + 1) % self.t] == self.fermion[(x + 1) % self.n, (y + 1) % self.t]:
                self.bond[x, y] = False
            # diagonal occupation
            elif self.fermion[x, y] != self.fermion[x, (y + 1) % self.t]:
                self.bond[x, y] = True
            # parallel occupation
            else:
                self.bond[x, y] = False if random.random() < self.w_a / (self.w_a + self.w_b) else True

    def _charge_automaton_depricated(self, row, charge_index, case_character):
        next_row = -1
        arrow_weight = 0
        charge = self.cluster_charge[self.charged_cluster_order[charge_index]]
        pm_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][0]
        mp_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][1]
        last_charge_index = len(self.charged_cluster_order) - 1

        match row:
            case 0:
                if charge_index < last_charge_index - 1:
                    match case_character:
                        case 0:
                            next_row = 0
                            arrow_weight = 1
                        case 1:
                            next_row = -1
                            arrow_weight = pm_combinations
                        case 2:
                            next_row = 1
                            arrow_weight = mp_combinations
                        case 3:
                            if charge > 1:
                                next_row = -2
                                arrow_weight = mp_combinations + 1
                            else:
                                next_row = 2
                                arrow_weight = pm_combinations + 1
                elif charge_index == last_charge_index - 1 and case_character == 3:
                    next_row = -2
                    arrow_weight = mp_combinations + 1
            case -1:
                if charge_index < last_charge_index - 1:
                    match case_character:
                        case 1:
                            next_row = -1
                            arrow_weight = pm_combinations + 1
                        case 3:
                            if charge > 1:
                                next_row = -2
                                arrow_weight = mp_combinations + 1
                elif charge_index == last_charge_index - 1 and case_character == 3:
                    next_row = -2
                    arrow_weight = mp_combinations + 1
            case 1:
                if charge_index <= last_charge_index - 2:
                    match case_character:
                        case 2:
                            next_row = 1
                            arrow_weight = mp_combinations + 1
                        case 3:
                            if charge < 1:
                                next_row = 2
                                arrow_weight = pm_combinations + 1
                elif charge_index == last_charge_index - 2 and case_character == 3:
                    next_row = 2
                    arrow_weight = pm_combinations + 1
            case -2:
                if charge_index <= last_charge_index - 1:
                    match case_character:
                        case 2:
                            next_row = -2
                            arrow_weight = mp_combinations + 1
                        case 3:
                            if charge < 1:
                                next_row = - 3
                                arrow_weight = pm_combinations + 1
                elif charge_index == last_charge_index and case_character == 3:
                    next_row = -3
                    arrow_weight = pm_combinations + 1
            case 2:
                if charge_index <= last_charge_index - 2:
                    match case_character:
                        case 1:
                            next_row = 2
                            arrow_weight = pm_combinations + 1
                        case 3:
                            if charge > 0:
                                next_row = 3
                                arrow_weight = mp_combinations + 1
                elif charge_index == last_charge_index - 1 and case_character == 3:
                    next_row = 3
                    arrow_weight = mp_combinations + 1
            case -3:
                if case_character == 1:
                    next_row = -3
                    arrow_weight = pm_combinations + 1
            case 3:
                if case_character == 2:
                    next_row = 3
                    arrow_weight = mp_combinations + 1
        return next_row, arrow_weight

    def _charge_automaton(self, row, charge_index, case_character):
        next_row = -1
        arrow_weight = 0
        charge = self.cluster_charge[self.charged_cluster_order[charge_index]]
        pm_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][0]
        mp_combinations = self.cluster_combinations[self.charged_cluster_order[charge_index]][1]
        last_charge_index = len(self.charged_cluster_order) - 1
        if row == 0:
            if case_character == 1:
                next_row = 0
                if charge > 0:
                    arrow_weight = mp_combinations + 1
                else:
                    arrow_weight = pm_combinations + 1
            elif case_character == 0 and not charge_index == last_charge_index:
                next_row = 1
                if charge > 0:
                    arrow_weight = pm_combinations + 1
                else:
                    arrow_weight = mp_combinations + 1
        elif row == 1:
            if case_character == 0:
                next_row = 0
                if charge > 0:
                    arrow_weight = mp_combinations + 1
                else:
                    arrow_weight = pm_combinations + 1
        return next_row, arrow_weight

    def _calculate_charge_combinations(self):
        self.charge_combinations = np.full((2, len(self.charged_cluster_order) + 1, 2), 0)

        # define legal final states
        self.charge_combinations[0, -1, 0] = 1

        for charge_index in range(len(self.charged_cluster_order) - 1, -1, -1):
            for row in range(2):
                for case_character in range(2):
                    next_row, weight = self._charge_automaton(row, charge_index, case_character)
                    if not weight == 0:
                        self.charge_combinations[row, charge_index, case_character] += weight * np.sum(
                            self.charge_combinations[next_row, charge_index + 1])

    # TODO
    def _generate_flips_depricated(self):
        row = 0
        for charge_idx in range(len(self.charged_cluster_order)):
            charge = self.charged_cluster_order[charge_idx]
            case_character = random.choices(range(4), weights=self.charge_combinations[row, charge_idx])[0]
            match case_character:
                case 0:
                    pass  # dont flip anything
                case 1:
                    if row == 0:
                        self._generate_neutral_flips_no_zero(charge, -1, self.cluster_charge[charge] < 0)
                    else:
                        self._generate_neutral_flips(charge, -1, self.cluster_charge[charge] < 0)
                case 2:
                    if row == 0:
                        self._generate_neutral_flips_no_zero(charge, 1, self.cluster_charge[charge] < 0)
                    else:
                        self._generate_neutral_flips(charge, 1, self.cluster_charge[charge] < 0)
                case 3:
                    self.flip[charge] = 1
                    self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                 self.cluster_charge[charge] < 0)

            row, weight = self._charge_automaton(row, charge_idx, case_character)

    def _generate_flips(self):
        row = 1
        self._generate_neutral_flips(1, -1, False)
        self.flip[0] = 1
        self.flip[1] = 0
        for charge_idx in range(len(self.charged_cluster_order)):
            charge = self.charged_cluster_order[charge_idx]
            case_character = random.choices(range(2), weights=self.charge_combinations[row, charge_idx])[0]
            match case_character:
                case 0:
                    if row == 0:
                        self._generate_neutral_flips(charge, - self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                    elif row == 1:
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                case 1:
                    if row == 0:
                        self.flip[charge] = 1
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                    else:
                        raise ('case 1 in row one')

            row, weight = self._charge_automaton(row, charge_idx, case_character)

    def mc_step(self):
        self._assign_bonds()
        self.draw_bonds()
        self._reset()
        self._find_clusters()
        self._identify_charged_clusters()
        self._correct_positions()
        self._assign_groups()
        self.draw_bonds()
        for charge in self.charged_cluster_order:
            self.cluster_combinations[charge] = self._calculate_neutral_combinations(charge,
                                                                                     self.cluster_charge[charge] > 0)
        self.charged_cluster_order = self.charged_cluster_order[2:]
        self._calculate_charge_combinations()

        n_flip_configs = 100000
        histogram = np.zeros(2 ** self.n_clusters)
        for i in range(n_flip_configs):
            self.flip = np.zeros(self.n_clusters)
            self._generate_flips()
            histogram[int("".join(str(int(k)) for k in self.flip), 2)] += 1

        self._flip()
        self.draw_bonds()

        plt.plot(histogram, ".")
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid()
        plt.show()
        pass
