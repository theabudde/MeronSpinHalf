import random

import numpy as np
from MeronAlgorithm import MeronAlgorithm


class SpinHalfBruteForce(MeronAlgorithm):

    def __init__(self, n, t, w_a, w_b, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, mc_steps)
        self.gauge_field = np.array([])
        self.result = np.zeros(n)

    def generate_flips(self):
        result = np.zeros(self.n)
        n_legal_configs = 0
        legal_configs = []
        if self.n_clusters > 20:
            print(self.n_clusters)
        for flip in range(2 ** self.n_clusters):
            flip = format(flip, f"0{self.n_clusters}b")
            self.flip = [int(x) for x in list(flip)]
            self.fermion = np.full((self.n, self.t), False)
            for i in range(self.n // 2):
                for j in range(self.t):
                    self.fermion[2 * i, j] = True
            self._flip()
            self._calculate_gauge_field()
            if self._test_gauss_law():
                n_legal_configs += 1
                legal_configs.append(flip)
                for site in range(self.n):
                    if self.fermion[0, 0] == self.fermion[site, 0]:
                        result[site] += 1
                    else:
                        result[site] -= 1
        flip = random.choice(legal_configs)
        self.flip = [int(x) for x in list(flip)]
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True
        self._flip()
        self.result += result / n_legal_configs

    def _calculate_gauge_field(self):
        self.gauge_field = np.zeros((self.n, self.t))
        for x in range(self.n):
            for y in range(1, self.t):
                if (y + 1) % 2 != x % 2 or self.fermion[x, y] == self.fermion[x, y - 1]:
                    self.gauge_field[x, y] = self.gauge_field[x, y - 1]
                elif not self.fermion[x, y]:
                    self.gauge_field[x, y] = self.gauge_field[x, y - 1] - 1
                else:
                    self.gauge_field[x, y] = self.gauge_field[x, y - 1] + 1
            if not x == self.n - 1:
                if self.fermion[x + 1, 0] == x % 2:
                    self.gauge_field[x + 1, 0] = self.gauge_field[x, 0]
                elif x % 2:
                    self.gauge_field[x + 1, 0] = self.gauge_field[x, 0] - 1
                else:
                    self.gauge_field[x + 1, 0] = self.gauge_field[x, 0] + 1

    def _test_gauss_law(self):
        if np.amax(self.gauge_field) - np.amin(self.gauge_field) > 1:
            return False
        for i in range(self.t):
            if self.fermion[0, i] == 1:
                if self.gauge_field[- 1, i] != self.gauge_field[0, i]:
                    return False
            else:
                if self.gauge_field[- 1, i] != self.gauge_field[0, i] + 1:
                    return False
        return True

    def corr_function(self):
        for i in range(self.mc_steps):
            self._assign_bonds()
            self._reset()
            self._find_clusters()
            self.generate_flips()
            if i % 100 == 0:
                print(i)
        self.result /= self.mc_steps