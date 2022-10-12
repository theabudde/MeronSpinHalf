import random
from MeronAlgorithm import MeronAlgorithm
import numpy as np
from itertools import product


class MeronAlgorithmOneFixedCharge(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, beta, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, beta, mc_steps)
        for y in range(self.t):
            self.fermion[0, y] = False
            self.fermion[3, y] = True

    # Places vertical and horizontal bonds with probability corresponding to wa_ and  w_b
    def _assign_bonds(self):
        for x, y in product(range(self.n), range(self.t)):
            if y % 2 != x % 2:
                self.bond[x, y] = - 1
            elif x == self.n - 1 or x == 0:
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

    def _charge_automaton(self, row, charge_index, case_character):
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

    def _calculate_charge_combinations(self):
        self.charge_combinations = np.full((7, len(self.charged_cluster_order) + 1, 4), 0)

        # define legal final states
        self.charge_combinations[-3, -1, 0] = 1
        self.charge_combinations[3, -1, 0] = 1

        for charge_index in range(len(self.charged_cluster_order) - 1, -1, -1):
            for row in range(-3, 4, 1):
                for case_character in range(4):
                    next_row, weight = self._charge_automaton(row, charge_index, case_character)
                    if not weight == 0:
                        self.charge_combinations[row, charge_index, case_character] += weight * np.sum(
                            self.charge_combinations[next_row, charge_index + 1])

    def _generate_flips(self):
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
