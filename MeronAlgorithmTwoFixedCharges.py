import random
from MeronAlgorithmOneFixedCharge import MeronAlgorithmOneFixedCharge
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


class MeronAlgorithmTwoFixedCharges(MeronAlgorithmOneFixedCharge):
    def __init__(self, n, t, w_a, w_b, w_c, beta, mc_steps):
        MeronAlgorithmOneFixedCharge.__init__(self, n, t, w_a, w_b, w_c, beta, mc_steps)

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

    def _generate_flips(self):
        self._generate_neutral_flips(1, 1, False)
        self.flip[0] = 1
        self.flip[1] = 1
        MeronAlgorithmOneFixedCharge._generate_flips(self)

    def mc_step(self):
        self._assign_bonds()
        self.draw_bonds()
        self._reset()
        self._find_clusters()
        self._identify_charged_clusters()
        self._correct_positions()
        self._assign_groups_with_charges()
        self.draw_bonds()
        for charge in self.charged_cluster_order:
            self.cluster_combinations[charge] = self._calculate_neutral_combinations(charge,
                                                                                     self.cluster_charge[charge] > 0)
        self.charged_cluster_order = self.charged_cluster_order[2:]
        self._calculate_charge_combinations()

        n_flip_configs = 1
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
        # plt.show()
        pass
