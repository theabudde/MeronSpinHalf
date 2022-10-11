from spinHalf import MeronAlgorithm
import numpy as np
from itertools import product


class MeronAlgorithmTwoCharges(MeronAlgorithm):
    def __init__(self, n, t, w_a, w_b, beta, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, beta, mc_steps)

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
                if charge_index <= last_charge_index - 1:
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
                elif charge_index == last_charge_index - 2 and case_character == 3:
                    next_row = 2
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
