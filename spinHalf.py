import random
import numpy as np
from itertools import product
from PIL import Image, ImageDraw


class MeronAlgorithm:
    def __init__(self, n, t, w_a, w_b, beta, mc_steps):

        # constants
        self.n = n
        self.t = t
        self.w_a = w_a
        self.w_b = w_b
        self.beta = beta
        self.mc_steps = mc_steps

        self.cluster_positions = []     # top left most position of each cluster in order cluster_id
        self.cluster_charge = np.array([0])    # charge of each cluster in order cluster_id
        self.cluster_nr = 0     # number of clusters
        self.cluster_group = np.array([0])  # closest charged left neighbor of every neutral cluster

        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

        self.cluster_id = np.full((self.n, self.t), -1)

        # bond lattice is squashed down and initalized to vertical plaquettes
        self.bond = np.full((self.n // 2, self.t), False)  # bond lattice, 0 is vertical plaquette A, 1 is horizontal plaquette B

        self.bond_debug = np.full((self.n, self.t), - 1)  # only for debugging purposes

    def _cluster_loop_step(self, x, y, visited):
        loop_closed = False
        if x % 2 == 0 and y % 2 == 0:
            # top
            if not self.bond[(x // 2 - 1) % (self.n // 2), (y - 1) % self.t] and not visited[x, (y - 1) % self.t]:
                y -= 1
            # right
            elif self.bond[x // 2, y] and not visited[(x + 1) % self.n, y]:
                x += 1
            # left
            elif self.bond[(x // 2 - 1) % (self.n // 2), (y - 1) % self.t] and not visited[(x - 1) % self.n, y]:
                x -= 1
            # bottom
            elif not self.bond[x // 2, y] and not visited[x, (y + 1) % self.t]:
                y += 1
            # closed loop
            else:
                loop_closed = True
        elif x % 2 == 1 and y % 2 == 0:
            # top
            if not self.bond[x // 2, (y - 1) % self.t] and not visited[x, (y - 1) % self.t]:
                y -= 1
            # right
            elif self.bond[x // 2, (y - 1) % self.t] and not visited[(x + 1) % self.n, y]:
                x += 1
            # left
            elif self.bond[x // 2, y] and not visited[(x - 1) % self.n, y]:
                x -= 1
            # bottom
            elif not self.bond[x // 2, y] and not visited[x, (y + 1) % self.t]:
                y += 1
            # closed loop
            else:
                loop_closed = True
        elif x % 2 == 0 and y % 2 == 1:
            # top
            if not self.bond[x // 2, (y - 1) % self.t] and not visited[x, (y - 1) % self.t]:
                y -= 1
            # right
            elif self.bond[x // 2, (y - 1) % self.t] and not visited[(x + 1) % self.n, y]:
                x += 1
            # left
            elif self.bond[(x // 2 - 1) % (self.n // 2), y] and not visited[(x - 1) % self.n, y]:
                x -= 1
            # bottom
            elif not self.bond[(x // 2 - 1) % (self.n // 2), y] and not visited[x, (y + 1) % self.t]:
                y += 1
            # closed loop
            else:
                loop_closed = True
        elif x % 2 == 1 and y % 2 == 1:
            # top
            if not self.bond[x // 2, (y - 1) % self.t] and not visited[x, (y - 1) % self.t]:
                y -= 1
            # right
            elif self.bond[x // 2, y] and not visited[(x + 1) % self.n, y]:
                x += 1
            # left
            elif self.bond[x // 2, (y - 1) % self.t] and not visited[(x - 1) % self.n, y]:
                x -= 1
            # bottom
            elif not self.bond[x // 2, y] and not visited[x, (y + 1) % self.t]:
                y += 1
            # closed loop
            else:
                loop_closed = True
        x = x % self.n  # boundary conditions
        y = y % self.t
        return x, y, loop_closed

    def tests(self):
        charge = np.zeros(self.cluster_id.max() + 1)
        for j in range(self.t):
            for c in range(self.cluster_id.max() + 1):
                rowcharge = 0
                for i in range(self.n):
                    if self.cluster_id[i, j] == c:
                        if i % 2:
                            rowcharge += 1
                        else:
                            rowcharge -= 1
                if charge[c] != rowcharge and j > 0:
                    raise ('Charge varies over different rows')
                charge[c] = rowcharge
        # print(charge)
        if charge.sum() != 0:
            raise ('Total charge not zero')

        if charge.max() > 1:
            if np.count_nonzero(charge == charge.max()) > 1:
                print('multiple 2 windings')
            for c in charge:
                if abs(c) != charge.max() and c != 0:
                    raise ('clusters of different charges mixed')
        if charge.max() > 1:
            print(charge.max())

        for j in range(self.t):
            for i in range(self.n - 2):
                if self.cluster_id[i, j] != self.cluster_id[i + 1, j] and abs(charge[self.cluster_id[i, j]]) == abs(charge[self.cluster_id[i + 1, j]]) == 1:
                    assert (charge[self.cluster_id[i, j]] != charge[self.cluster_id[i + 1, j]])

    # Places vertical and horizontal bonds with probability corresponding to wa_ and  w_b
    def _bond_assignment(self):
        for x, y in product(range(self.n), range(self.t)):
            if y % 2 != x % 2:
                continue
            # all occupied or all unoccupied
            if self.fermion[x, y] == self.fermion[(x + 1) % self.n, y] and self.fermion[x, (y + 1) % self.t] == self.fermion[(x + 1) % self.n, (y + 1) % self.t]:
                self.bond[x // 2, y] = False
            # diagonal occupation
            elif self.fermion[x, y] != self.fermion[x, (y + 1) % self.t]:
                self.bond[x // 2, y] = True
            # parallel occupation
            else:
                self.bond[x // 2, y] = False if random.random() < self.w_a / (self.w_a + self.w_b) else True
            # calculate bond config in nicer lattice for debugging purposes
            self.bond_debug[x, y] = self.bond[x // 2, y]

    # reset to reference config
    def _reset(self):
        self.cluster_positions = []  # one position of each cluster in order cluster_id
        self.cluster_charge = np.array([0])  # charge of each cluster in order cluster_id
        self.cluster_nr = 0  # number of clusters

        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

        self.cluster_id = np.full((self.n, self.t), -1)

        # bond lattice is squashed down and initalized to vertical plaquettes
        self.bond = np.full((self.n // 2, self.t),
                            False)  # bond lattice, 0 is vertical plaquette A, 1 is horizontal plaquette B

        self.bond_debug = np.full((self.n, self.t), - 1)  # only for debugging purposes

    def _find_clusters(self):
        visited = np.full((self.n, self.t), False)  # record if site has been visited
        cluster_nr = 0  # counter for how many clusters there are -1 and the ID given to each of the clusters
        self.cluster_positions = []  # keeps one position of the cluster in order of cluster_id
        for i, j in product(range(self.n), range(self.t)):  # check for a new cluster in all positions
            if not visited[i, j]:  # if you haven't seen the loop before
                x = i
                y = j
                self.cluster_positions.append([x, y])
                # Go around a cluster loop
                loop_closed = False
                while not loop_closed:
                    self.cluster_id[x, y] = cluster_nr  # give cluster its ID
                    visited[x, y] = True  # Save where algorithm has been, so you don't go backwards around the loop

                    # update x and y to next position in cluster loop
                    x, y, loop_closed = self._cluster_loop_step(x, y, visited)

                # look where to find next cluster
                cluster_nr += 1
        self.cluster_nr = cluster_nr + 1

    def mc_step(self):
        # reset to reference config
        self._reset()

        # place new bonds
        self._bond_assignment()

        # find clusters
        self._find_clusters()

        # optional: run tests to verify hypothesis of cluster structure (very slow)
        self.tests()

        # determine cluster's charges
        self.cluster_charge = np.zeros(self.cluster_nr)
        for i in range(self.n):
            if i % 2:
                self.cluster_charge[self.cluster_id[i, 0]] += 1
            else:
                self.cluster_charge[self.cluster_id[i, 0]] -= 1

                # determine order of charged clusters
        charged_cluster_order = []
        for i in range(self.n):
            if self.cluster_charge[self.cluster_id[i, 0]] != 0:
                charged_cluster_order.append(self.cluster_id[i, 0])

        

    def draw_bonds(self):
        scale = 40
        image = Image.new("RGB", (scale*self.n + 2, scale*self.t+2), "white")
        draw = ImageDraw.Draw(image)
        for x in range(self.n):
            for y in range(self.t):
                if self.bond_debug[x, y] == 1:
                    draw.line([(x*scale, y*scale), ((x+1) * scale, y * scale)], width=scale//10, fill="green", joint="curve")
                    draw.line([(x*scale, (y+1)*scale), ((x+1) * scale, (y+1) * scale)], width=scale//10, fill="green", joint="curve")
                elif self.bond_debug[x, y] == 0:
                    draw.line([(x * scale, y * scale), (x * scale, (y+1) * scale)], width=scale // 10, fill="green",
                              joint="curve")
                    draw.line([((x+1) * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], width=scale // 10,
                              fill="green", joint="curve")
                np.random.seed(self.cluster_id[x, y] + 30)
                color = tuple(np.random.choice(range(256), size=3))
                draw.ellipse((x*scale - 10, y*scale-10, x*scale+10, y*scale+10), fill=color, outline='black')
                if x % 2:
                    draw.text((x*scale-4, y*scale-4), "+", fill=(0, 0, 0))
                else:
                    draw.text((x*scale-4, y*scale-4), "-", fill=(0, 0, 0))

        image.save("config.jpg")


def main():
    n = 30  # number of lattice points
    t = 18  # number of half timesteps (#even + #odd)
    beta = 1   # beta
    mc_steps = 1   # number of mc steps
    initial_mc_steps = 5000
    w_a = 3/4  # np.exp(b/t)  # weight of a plaquettes U = t = 1
    w_b = 1/4  # np.sinh(b/t)  # weight of b plaquettes

    Algorithm = MeronAlgorithm(n, t, w_a, w_b, beta, mc_steps)

    for mc in range(mc_steps):
        Algorithm.mc_step()
    Algorithm.draw_bonds()




if __name__ == "__main__":
    main()