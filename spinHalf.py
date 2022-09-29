import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import deque
from itertools import product


class MeronAlgorithm:
    def __init__(self, n, t, w_a, w_b, beta, mc_steps):
        # constants
        self.n = n
        self.t = t
        self.w_a = w_a
        self.w_b = w_b
        self.beta = beta
        self.mc_steps = mc_steps

        # positions of fermions
        self.fermion = np.full((self.n, self.t), False)
        # bond lattice, 0 is vertical plaquette A, 1 is horizontal plaquette B
        self.bond = np.full((self.n, self.t), - 1)
        # number of clusters in configuration
        self.n_clusters = -1
        # cluster_id of the cluster in a given position
        self.cluster_id = np.full((self.n, self.t), -1)
        # top left most position of each cluster indexed by cluster_id
        self.cluster_positions = {}
        # charge of each cluster in order cluster_id
        self.cluster_charge = np.array([0])
        # order of charged clusters only or if only neutrals exist, the horizontally winding clusters
        self.charged_cluster_order = []
        # number of times a neutral cluster wraps horizontally
        self.horizontal_winding = np.array([0])
        # if only non winding neutrals these are the left neighbors of the 0 cluster
        # the first one will be an outermost cluster
        self.left_neighbors = []
        # left neighbor going counterclockwise of every neutral cluster
        self.cluster_group = np.array([0])
        # order of nested neutral clusters right of the indexed charge
        self.cluster_order = {}
        # nesting patterns of the cluster indexed by the charge group, +1 (+2) for the opening (closing) of a +- cluster and
        # -1 (-2) for the opening (closing) of a -+ cluster and
        self.nesting_brackets = {}
        # saves the nr of flip possibilites for +- and -+ starting from the corresponding cluster
        self.cluster_combinations = np.array([])

        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

    # todo more efficient way to save bond
    # this in init: self._bond = np.full((self.n // 2, self.t), -1)
    # @property
    # def bond(self):
    #     class Bond:
    #         def __getitem__(slf, item):
    #             return self._bond[item[0] // 2, item[1]]
    #
    #         def __setitem__(slf, key, value):
    #             self._bond[key[0] // 2, key[1]] = value
    #
    #     return Bond()

    def _reset(self):
        # positions of fermions
        self.fermion = np.full((self.n, self.t), False)
        # bond lattice, 0 is vertical plaquette A, 1 is horizontal plaquette B
        self.bond = np.full((self.n, self.t), - 1)
        # number of clusters in configuration
        self.n_clusters = -1
        # cluster_id of the cluster in a given position
        self.cluster_id = np.full((self.n, self.t), -1)
        # top left most position of each cluster indexed by cluster_id
        self.cluster_positions = {}
        # charge of each cluster in order cluster_id
        self.cluster_charge = np.array([0])
        # order of charged clusters only
        self.charged_cluster_order = []
        # number of times a neutral cluster wraps horizontally
        self.horizontal_winding = np.array([0])
        # order of the horizontally wrapping clusters
        self.horizontal_winding_order = []
        # left neighbor going counterclockwise of every neutral cluster
        self.cluster_group = np.array([0])
        # order of clusters for automaton to be able to process
        self.cluster_order = {}
        # nesting patterns of the cluster. 0 for a charge, +1 (+2) for the opening (closing) of a +- cluster and
        # -1 (-2) for the opening (closing) of a -+ cluster and
        self.nesting_brackets = {}
        # saves the nr of flip possibilites for +- and -+ starting from the corresponding cluster
        self.cluster_combinations = np.array([])

        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

    # Places vertical and horizontal bonds with probability corresponding to wa_ and  w_b
    def _bond_assignment(self):
        for x, y in product(range(self.n), range(self.t)):
            if y % 2 != x % 2:
                continue
            # all occupied or all unoccupied
            if self.fermion[x, y] == self.fermion[(x + 1) % self.n, y] \
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
        # TODO: also for neutrals when implemented
        if len(self.cluster_group) != 1:
            for x, y in product(range(self.n), range(self.t)):
                color = self._get_random_color(self.cluster_group[self.cluster_id[x, y]])
                draw.rectangle(((x - 0.5) * scale, (y - 0.5) * scale, (x + 0.5) * scale, (y + 0.5) * scale),
                               fill=color)

        for x, y in product(range(self.n), range(self.t)):
            # TODO: don't use debug
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
            if x % 2:
                draw.text((x * scale - 4, y * scale - 4), "+", fill=(0, 0, 0))
            else:
                draw.text((x * scale - 4, y * scale - 4), "-", fill=(0, 0, 0))

        image.save("config.jpg")

    def _get_random_color(self, index):
        np.random.seed(index + 10)
        color = tuple(np.append(np.random.choice(range(256), size=3), 127))
        return color

    def _find_clusters(self):
        visited = np.full((self.n, self.t), False)  # record if site has been visited
        # counter for how many clusters there are -1 and the ID given to each of the clusters
        cluster_nr = 0
        # Order like this to ensure top left position is encountered first
        for j, i in product(range(self.t), range(self.n)):
            if not visited[i, j]:  # if you haven't seen the loop before
                x = i
                y = j
                self.cluster_positions[cluster_nr] = (x, y)
                # Go around a cluster loop
                loop_closed = False
                while not loop_closed:
                    self.cluster_id[x, y] = cluster_nr  # give cluster its ID
                    visited[x, y] = True  # Save where algorithm has been, so you don't go backwards around the loop
                    # update x and y to next position in cluster loop
                    x, y, loop_closed, direction = self._cluster_loop_step(x, y, visited)

                # look where to find next cluster
                cluster_nr += 1
        self.n_clusters = cluster_nr

    def _cluster_loop_step(self, x, y, visited):
        loop_closed = False

        up = 0
        right = 1
        down = 2
        left = 3

        direction = -1

        coord_bond_up_left = (x - 1) % self.n, (y - 1) % self.t
        coord_bond_down_left = (x - 1) % self.n, y % self.t
        coord_bond_down_right = x % self.n, y % self.t
        coord_bond_up_right = x % self.n, (y - 1) % self.t

        coord_up = x, (y - 1) % self.t
        coord_right = (x + 1) % self.n, y
        coord_down = x, (y + 1) % self.t
        coord_left = (x - 1) % self.n, y

        if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
            if not self.bond[coord_bond_up_left] and not visited[coord_up]:
                y -= 1
                direction = up
            elif self.bond[coord_bond_down_right] and not visited[coord_right]:
                x += 1
                direction = right
            elif not self.bond[coord_bond_down_right] and not visited[coord_down]:
                y += 1
                direction = down
            elif self.bond[coord_bond_up_left] and not visited[coord_left]:
                x -= 1
                direction = left
            else:
                loop_closed = True
        elif (x % 2 == 1 and y % 2 == 0) or (x % 2 == 0 and y % 2 == 1):
            if not self.bond[coord_bond_up_right] and not visited[coord_up]:
                y -= 1
                direction = up
            elif self.bond[coord_bond_up_right] and not visited[coord_right]:
                x += 1
                direction = right
            elif not self.bond[coord_bond_down_left] and not visited[coord_down]:
                y += 1
                direction = down
            elif self.bond[coord_bond_down_left] and not visited[coord_left]:
                x -= 1
                direction = left
            else:
                loop_closed = True
        x = x % self.n  # boundary conditions
        y = y % self.t
        return x, y, loop_closed, direction

    def tests(self, seed):
        charge = np.zeros(self.n_clusters + 1)
        for j in range(self.t):
            for c in range(self.n_clusters + 1):
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

        if np.amax(charge) > 1:
            if np.count_nonzero(charge == np.amax(charge)) > 1:
                print('multiple 2 windings')
            for c in charge:
                if abs(c) != np.amax(charge) and c != 0:
                    raise ('clusters of different charges mixed')
        if np.amax(charge) > 1:
            print(np.amax(charge), seed)

        for j in range(self.t):
            for i in range(self.n - 2):
                if self.cluster_id[i, j] != self.cluster_id[i + 1, j] and abs(charge[self.cluster_id[i, j]]) == abs(
                        charge[self.cluster_id[i + 1, j]]) == 1:
                    assert (charge[self.cluster_id[i, j]] != charge[self.cluster_id[i + 1, j]])

    def _identify_charged_clusters(self):
        # determine cluster's charges
        self.cluster_charge = np.zeros(self.n_clusters)
        for i in range(self.n):
            if i % 2:
                self.cluster_charge[self.cluster_id[i, 0]] += 1
            else:
                self.cluster_charge[self.cluster_id[i, 0]] -= 1

        # determine order of charged clusters
        for i in range(self.n):
            if self.cluster_charge[self.cluster_id[i, 0]] != 0 and not self.cluster_id[
                                                                           i, 0] in self.charged_cluster_order:
                self.charged_cluster_order.append(self.cluster_id[i, 0])

    def _identify_horizontal_charges(self):
        # determine cluster's charges
        self.horizontal_winding = np.zeros(self.n_clusters)
        for i in range(self.t):
            if i % 2:
                self.horizontal_winding[self.cluster_id[0, i]] += 1
            else:
                self.horizontal_winding[self.cluster_id[0, i]] -= 1

        # determine order of charged clusters
        for i in range(self.t):
            if self.horizontal_winding[self.cluster_id[0, i]] != 0 and not self.cluster_id[
                                                                               0, i] in self.charged_cluster_order:
                self.charged_cluster_order.append(self.cluster_id[0, i])
                self.cluster_positions[self.cluster_id[0, i]] = (0, i)

    def correct_left_position_of_charged_clusters(self):
        # correct the charged one that is closest from the left to [0,0]
        if len(self.charged_cluster_order) > 0:
            # correct the top leftmost position of leftmost charged cluster
            x = self.cluster_positions[self.charged_cluster_order[-1]][0]
            while self.cluster_id[x, 0] != self.charged_cluster_order[0]:
                x = (x + 1) % self.n
            self.cluster_positions[self.charged_cluster_order[0]] = (x, 0)

    def correct_top_position_of_horizontally_winding_clusters(self):
        # correct the topmost position of topmost horizontally winding cluster
        y = self.cluster_positions[self.charged_cluster_order[-1]][1]
        while self.cluster_id[0, y] != self.charged_cluster_order[0]:
            y = (y + 1) % self.t
        self.cluster_positions[self.charged_cluster_order[1]] = (0, y)

    def _correct_left_positions_of_boundary_clusters(self):
        corrected = []
        visited = np.zeros((self.n, self.t))
        for j in range(self.t):
            if not (self.cluster_id[0, j] in corrected) and self.cluster_charge[self.cluster_id[0, j]] == 0 and \
                    self.horizontal_winding[self.cluster_id[0, j]] == 0:
                x = 0
                y = j
                corrected.append(self.cluster_id[0, j])
                closed_loop = False
                while not closed_loop:
                    x, y, closed_loop, direction = self._cluster_loop_step(x, y, visited)
                    visited[x, y] = True
                    if direction == 3 and x == (self.cluster_positions[self.cluster_id[x, y]][0] - 1) % self.n:
                        self.cluster_positions[self.cluster_id[x, y]] = [x, y]

    def _left_neighbor(self, cluster):
        x = self.cluster_positions[cluster][0]
        y = self.cluster_positions[cluster][1]
        closed_loop = False
        visited = np.zeros((self.n, self.t))

        while not closed_loop:
            visited[x, y] = True
            x, y, closed_loop, direction = self._cluster_loop_step(x, y, visited)
            cluster_up = self.cluster_id[x, (y - 1) % self.t]
            cluster_right = self.cluster_id[(x + 1) % self.n, y]
            cluster_down = self.cluster_id[x, (y + 1) % self.t]
            cluster_left = self.cluster_id[(x - 1) % self.n, y]

            # check left (right) neighbors of cluster and mark them as being in the same (own_id) group
            # and recursively check their neighbors too
            if direction == 0:
                self._find_left_neighboring_cluster(cluster_left)
            elif direction == 1:
                self._find_left_neighboring_cluster(cluster_up)
            elif direction == 2:
                self._find_left_neighboring_cluster(cluster_right)
            elif direction == 3:
                self._find_left_neighboring_cluster(cluster_down)

    def _find_left_neighboring_cluster(self, neighbor_id):
        if neighbor_id not in self.left_neighbors:
            self.left_neighbors.append(neighbor_id)
            self._left_neighbor(neighbor_id)

    def _assign_groups(self):
        self.cluster_group = np.full(self.n_clusters, -1)
        # recursive identification of nearest left charge or surrounding cluster for all neutral clusters
        for i in range(len(self.charged_cluster_order)):
            self._group_neighboring_clusters(self.charged_cluster_order[i - 1], self.charged_cluster_order[i],
                                             self.cluster_positions[self.charged_cluster_order[i]][0],
                                             self.cluster_positions[self.charged_cluster_order[i]][1])

        # find cluster order recursively
        for charge in self.charged_cluster_order:
            self.nesting_brackets[charge] = []
            self.cluster_order[charge] = []
            self._find_cluster_order(charge, charge)

    def _group_neighboring_clusters(self, left_group, right_group, x_start, y_start):
        x = x_start
        y = y_start
        closed_loop = False
        visited = np.zeros((self.n, self.t))

        while not closed_loop:
            visited[x, y] = True
            x, y, closed_loop, direction = self._cluster_loop_step(x, y, visited)
            cluster_up = self.cluster_id[x, (y - 1) % self.t]
            cluster_right = self.cluster_id[(x + 1) % self.n, y]
            cluster_down = self.cluster_id[x, (y + 1) % self.t]
            cluster_left = self.cluster_id[(x - 1) % self.n, y]

            # check left (right) neighbors of cluster and mark them as being in the same (own_id) group
            # and recursively check their neighbors too
            if direction == 0:
                self._mark_neighboring_clusters(cluster_left, left_group)
                self._mark_neighboring_clusters(cluster_right, right_group)
            elif direction == 1:
                self._mark_neighboring_clusters(cluster_up, left_group)
                self._mark_neighboring_clusters(cluster_down, right_group)
            elif direction == 2:
                self._mark_neighboring_clusters(cluster_right, left_group)
                self._mark_neighboring_clusters(cluster_left, right_group)
            elif direction == 3:
                self._mark_neighboring_clusters(cluster_down, left_group)
                self._mark_neighboring_clusters(cluster_up, right_group)

    def _mark_neighboring_clusters(self, neighbor_id, neighbor_group):
        if self.cluster_charge[neighbor_id] == 0 and self.cluster_group[neighbor_id] == -1:
            self.cluster_group[neighbor_id] = neighbor_group
            self._group_neighboring_clusters(neighbor_group, neighbor_id,
                                             self.cluster_positions[neighbor_id][0],
                                             self.cluster_positions[neighbor_id][1])

    # O(cluster_nr^2) TODO: could probably be done faster
    def _find_cluster_order(self, cluster_group, charge_group):
        self.nesting_brackets[charge_group].append(-1)
        for i in range(self.n_clusters):
            if self.cluster_group[i] == cluster_group and i != cluster_group:
                self.cluster_order[charge_group].append(i)
                self._find_cluster_order(i, charge_group)
        self.nesting_brackets[charge_group].append(-2)

    def _evaluate(self, start_index, final_index):
        if start_index >= final_index:
            return np.zeros(2)
        if self.cluster_combinations[start_index + 1, 0] == -3:
            j = start_index + 2
            while self.cluster_combinations[j, 0] != -4:
                j += 1
            combis = self._evaluate(start_index + 2, j - 1)
            combis = np.array([combis[0] + combis[1] + 1, combis[1]])
            self.cluster_combinations[start_index] = combis
            return (combis + 1) * (self._evaluate(j + 1, final_index) + 1) - 1
        if self.cluster_combinations[start_index + 1, 0] == -5:
            j = start_index + 2
            while self.cluster_combinations[j, 0] != -6:
                j += 1
            combis = self._evaluate(start_index + 2, j - 1)
            combis = np.array([combis[0], combis[0] + combis[1] + 1])
            self.cluster_combinations[start_index] = combis
            return (combis + 1) * (self._evaluate(j + 1, final_index) + 1) - 1
        if self.cluster_combinations[start_index + 1, 0] == -1:
            j = start_index + 2
            while self.cluster_combinations[j, 0] != -2:
                j += 1
            self.cluster_combinations[start_index] = self._evaluate(start_index + 2, j - 1)
            self._evaluate(j + 1, final_index)

    def _generate_flips(self):
        boundary_condition = deque()
        flip = []
        i = 0
        while i < self.cluster_combinations.shape[0]:
            if self.cluster_combinations[i + 1, 0] == -1:
                boundary_condition.append(self.cluster_charge[self.cluster_order[i]])
                i += 2
            elif self.cluster_combinations[i, 0] == -2:
                break
            elif np.array_equal(self.cluster_combinations[i], np.array([1, 0])) and boundary_condition[-1] > 0:
                flip.append(0)
                i += 3
            elif np.array_equal(self.cluster_combinations[i], np.array([1, 0])) and boundary_condition[-1] < 0:
                flip.append(0 if random.random() < 0.5 else 1)
                i += 3
            elif np.array_equal(self.cluster_combinations[i], np.array([0, 1])) and boundary_condition[-1] > 0:
                flip.append(0 if random.random() < 0.5 else 1)
                i += 3
            elif np.array_equal(self.cluster_combinations[i], np.array([0, 1])) and boundary_condition[-1] < 0:
                flip.append(0)
                i += 3
            elif self.cluster_combinations[i + 1, 0] == -3 and boundary_condition[-1] < 0:
                if random.random() < (self.cluster_combinations[i, 1] + 1) / (self.cluster_combinations[i, 0] + 1):
                    flip.append(1)
                    i += 2
                    boundary_condition.append(boundary_condition[-1] * -1)
                else:
                    flip.append(0)
                    i += 2
                    boundary_condition.append(boundary_condition[-1])
            elif self.cluster_combinations[i + 1, 0] == -3 and boundary_condition[-1] > 0:
                flip.append(0)
                i += 2
                boundary_condition.append(boundary_condition[-1])
            elif self.cluster_combinations[i + 1, 0] == -5 and boundary_condition[-1] > 0:
                if random.random() < (self.cluster_combinations[i, 0] + 1) / (self.cluster_combinations[i, 1] + 1):
                    flip.append(1)
                    i += 2
                    boundary_condition.append(boundary_condition[-1] * -1)
                else:
                    flip.append(0)
                    i += 2
                    boundary_condition.append(boundary_condition[-1])
            elif self.cluster_combinations[i + 1, 0] == -5 and boundary_condition[-1] < 0:
                flip.append(0)
                i += 2
                boundary_condition.append(boundary_condition[-1])
            # but only if the cluster was actually flipped!
            elif self.cluster_combinations[i, 0] == -4 or self.cluster_combinations[i, 0] == -6:
                boundary_condition.pop()
                i += 1
        return flip

    def mc_step(self):
        seed = 6
        # for seed in range(0, 10000):
        random.seed(seed)

        # reset to reference config
        self._reset()

        # place new bonds
        self._bond_assignment()

        # find clusters
        self._find_clusters()

        # draw config for debug
        self.draw_bonds()

        # optional: run tests to verify hypothesis of cluster structure (very slow)
        # self.tests(seed)

        charged = False
        self._identify_charged_clusters()
        self._identify_horizontal_charges()

        if len(self.charged_cluster_order) > 0:
            charged = True
            self.correct_left_position_of_charged_clusters()
        elif len(self.horizontal_winding_order) > 0:
            # TODO: look at ordering, what constraints does a horizontal charge give?
            self.correct_top_position_of_horizontally_winding_clusters()

        self._correct_left_positions_of_boundary_clusters()

        # if there are no charged clusters and no horizontal windings find an outermost cluster
        if len(self.charged_cluster_order) == 0:
            self.left_neighbors.append(0)
            self._left_neighbor(0)
            # TODO: Don't know if this actually works ordering not quite obvious
            start_cluster = self.left_neighbors[-1]
            self.cluster_group = np.full(self.n_clusters, -1)
            self.cluster_group[start_cluster] = -2
            self._group_neighboring_clusters(-2, start_cluster, self.cluster_positions[start_cluster][0],
                                             self.cluster_positions[start_cluster][1])
            self._find_cluster_order(-2)

        else:
            self._assign_groups()

        # create picture for debugging
        self.draw_bonds()

        # adjust brackets corresponding to sign
        # 0 just skip
        # -1 opens -2 closes charges
        # -3 opens -4 closes +- loops
        # -5 opens -6 closes -+ loops
        current_last_charge = 0
        cluster_order_index = 0
        for i in range(len(self.nesting_brackets)):
            if len(self.charged_cluster_order) == 0 and i == 0 or i == len(self.nesting_brackets) - 1:
                self.nesting_brackets[i] = 0
            if self.nesting_brackets[i] == -1:
                if self.cluster_charge[self.cluster_order[cluster_order_index]] > 0:
                    current_last_charge = +1
                elif self.cluster_charge[cluster_order_index] < 0:
                    current_last_charge = -1
                else:
                    self.nesting_brackets[i] = -4 - current_last_charge

                    # find closing bracket
                    j = i
                    loop_openings_counter = 0
                    while True:
                        if self.nesting_brackets[j] == -2:
                            if loop_openings_counter == 0:
                                break
                            else:
                                loop_openings_counter -= 1
                        if self.nesting_brackets[j] == -1:
                            loop_openings_counter += 1
                        j += 1
                    self.cluster_order[j] = -5 - current_last_charge
                    current_last_charge *= -1
                cluster_order_index += 1
            elif self.cluster_order[i] < 0 and self.cluster_order[i] % 2 == 0:
                current_last_charge *= -1

        # calculate the cluster combinations
        self.cluster_combinations = np.zeros((len(self.cluster_order), 2))
        for i in range(len(self.cluster_order)):
            if self.cluster_order[i] < 0:
                self.cluster_combinations[i] = self.cluster_order[i] * np.array([1, 1])
        self._evaluate(0, len(self.cluster_order) - 1)

        # generate a flip combination with hoomogeneous probability
        histogram = np.zeros(2 ** self.n_clusters)
        for i in range(10):
            flip = self._generate_flips()
            if not flip == []:
                histogram[int("".join(str(k) for k in flip), 2)] += 1
        plt.plot(histogram[histogram != 0], ".")
        plt.ylim(bottom=0)
        plt.show()

        print('test')


def main():
    n = 16  # number of lattice points
    t = 16  # number of half timesteps (#even + #odd)
    beta = 1  # beta
    mc_steps = 1  # number of mc steps
    initial_mc_steps = 5000
    w_a = 3 / 4  # np.exp(b/t)  # weight of a plaquettes U = t = 1
    w_b = 1 / 4  # np.sinh(b/t)  # weight of b plaquettes

    Algorithm = MeronAlgorithm(n, t, w_a, w_b, beta, mc_steps)

    for mc in range(mc_steps):
        Algorithm.mc_step()


if __name__ == "__main__":
    main()
