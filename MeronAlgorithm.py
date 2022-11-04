import random
import numpy as np
from PIL import Image, ImageDraw
from itertools import product


class MeronAlgorithm:
    def __init__(self, n, t, w_a, w_b, mc_steps):
        # constants
        self.n = n
        self.t = t
        self.w_a = w_a
        self.w_b = w_b
        self.mc_steps = mc_steps

        # positions of fermions
        self.fermion = np.full((self.n, self.t), False)
        # bond lattice, 0 is vertical plaquette A, 1 is horizontal plaquette B
        self.bond = np.zeros((self.n, self.t))
        # number of clusters in configuration
        self.n_clusters = -1
        # cluster_id of the cluster in a given position
        self.cluster_id = np.full((self.n, self.t), -1)
        # top left most position of each cluster indexed by cluster_id
        self.cluster_positions = {}
        # Bool whether to flip each cluster indexed by cluster_id
        self.flip = []

    def _reset(self):
        # cluster_id of the cluster in a given position
        self.cluster_id = np.full((self.n, self.t), -1)
        # top left most position of each cluster indexed by cluster_id
        self.cluster_positions = {}
        self.flip = []
        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

    # Places vertical and horizontal bonds with probability corresponding to wa_ and  w_b
    def _assign_bonds(self):
        for x, y in product(range(self.n), range(self.t)):
            if y % 2 != x % 2:
                self.bond[x, y] = - 1
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

    def draw_bonds(self):
        scale = 40
        image = Image.new("RGB", (scale * self.n + 2, scale * self.t + 2), "white")
        draw = ImageDraw.Draw(image)

        for x, y in product(range(self.n), range(self.t)):
            if self.bond[x, y] == 1:
                draw.line([(x * scale, y * scale), ((x + 1) * scale, y * scale)], width=scale // 10, fill="green",
                          joint="curve")
                draw.line([(x * scale, (y + 1) * scale), ((x + 1) * scale, (y + 1) * scale)], width=scale // 10,
                          fill="green", joint="curve")
            elif self.bond[x, y] == 0:
                draw.line([(x * scale, y * scale), (x * scale, (y + 1) * scale)], width=scale // 10, fill="green",
                          joint="curve")
                draw.line([((x + 1) * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], width=scale // 10,
                          fill="green", joint="curve")
            color = self._get_random_color(self.cluster_id[x, y])
            draw.ellipse((x * scale - 10, y * scale - 10, x * scale + 10, y * scale + 10), fill=color, outline='black')
            if self.fermion[x, y]:
                draw.text((x * scale - 4, y * scale - 4), "x", fill=(0, 0, 0))
            # if x % 2:
            #    draw.text((x * scale - 4, y * scale - 4), "+", fill=(0, 0, 0))
            # else:
            #    draw.text((x * scale - 4, y * scale - 4), "-", fill=(0, 0, 0))

        image.save("config.jpg")

    @staticmethod
    def _get_random_color(index):
        np.random.seed(index + 100)
        color = tuple(np.append(np.random.choice(range(256), size=3), 127))
        return color

    def _find_clusters(self):
        visited = np.full((self.n, self.t), False)  # record if site has been visited
        # counter for how many clusters there are -1 and the ID given to each of the clusters
        cluster_nr = 0
        # Order like this to ensure top left position is encountered first
        for y, x in product(range(self.t), range(self.n)):
            if not visited[x, y]:  # if you haven't seen the loop before
                new_coordinates = (x, y)
                start_coordinates = new_coordinates
                self.cluster_positions[cluster_nr] = new_coordinates
                leftness = 0
                max_leftness = 0
                while True:
                    visited[new_coordinates] = True
                    self.cluster_id[new_coordinates] = cluster_nr  # give cluster its ID
                    previous_coordinates = new_coordinates
                    new_coordinates = self._cluster_loop_step(new_coordinates)

                    if (previous_coordinates[0] - 1) % self.n == new_coordinates[0]:
                        leftness += 1
                        if leftness > max_leftness:
                            max_leftness = leftness
                            self.cluster_positions[cluster_nr] = new_coordinates
                    elif (previous_coordinates[0] + 1) % self.n == new_coordinates[0]:
                        leftness -= 1
                    elif (previous_coordinates[1] - 1) % self.t == new_coordinates[1] and leftness == max_leftness and \
                            new_coordinates[1] == (self.cluster_positions[cluster_nr][1] - 1) % self.t:
                        self.cluster_positions[cluster_nr] = new_coordinates

                    if new_coordinates == (x, y):
                        break
                # look where to find next cluster
                cluster_nr += 1
        self.n_clusters = cluster_nr
        self.flip = np.zeros(self.n_clusters)

    def _test_cluster_assignment(self):
        if -1 in self.cluster_id:
            raise 'not all clusters have a group'

        for cluster in range(self.n_clusters):
            if self.cluster_id[self.cluster_positions[cluster]] != cluster:
                raise 'cluster_position does not point to correct cluster'

    # works only in reference configuration!
    def _cluster_loop_step(self, current_position):
        x = current_position[0]
        y = current_position[1]

        coord_bond_up_left = (x - 1) % self.n, (y - 1) % self.t
        coord_bond_down_left = (x - 1) % self.n, y % self.t
        coord_bond_down_right = x % self.n, y % self.t
        coord_bond_up_right = x % self.n, (y - 1) % self.t

        up = x, (y - 1) % self.t
        right = (x + 1) % self.n, y
        down = x, (y + 1) % self.t
        left = (x - 1) % self.n, y

        current_cluster = self.cluster_id[current_position]

        # if occupied
        if current_position[0] % 2 == 0:
            if current_position[1] % 2 == 0:
                if self.bond[coord_bond_up_left] == 0:
                    next_position = up
                else:
                    next_position = left
            else:
                if self.bond[coord_bond_up_right] == 0:
                    next_position = up
                else:
                    next_position = right
        else:
            if current_position[1] % 2 == 0:
                if self.bond[coord_bond_down_left] == 0:
                    next_position = down
                else:
                    next_position = left
            else:
                if self.bond[coord_bond_down_right] == 0:
                    next_position = down
                else:
                    next_position = right
        return next_position

    def _test_cluster_loop_step(self):
        position = (random.randint(0, self.n), random.randint(0, self.t))
        if self.cluster_id[position] != self.cluster_id[self._cluster_loop_step(position)]:
            raise 'loop step went to wrong cluster'

    def _flip(self):
        for i in range(self.n_clusters):
            if self.flip[i]:
                start_position = self.cluster_positions[i]
                position = start_position
                # self.fermion[start_position] = not self.fermion[start_position]
                while True:
                    position = self._cluster_loop_step(position)
                    self.fermion[position] = not self.fermion[position]
                    if position == start_position:
                        break

    def mc_step(self):
        self._assign_bonds()
        self._reset()
        self._find_clusters()
        self.flip = [random.random() < 0.5 for i in range(self.n_clusters)]
        self._flip()
