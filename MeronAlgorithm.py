import random
from abc import abstractmethod

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from itertools import product


class MeronAlgorithm:

    @abstractmethod
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
        self.bond = np.zeros((self.n, self.t))
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
        # left neighbor going counterclockwise of every neutral cluster
        self.cluster_group = np.array([0])
        # order of nested neutral clusters indexed by their surrounding cluster/left charged neighbor
        self.cluster_order = {}
        # saves the nr of flip possibilites for +- and -+ starting from the corresponding cluster
        self.cluster_combinations = np.array([])
        self.flip = []
        self.charge_combinations = np.array([])

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
        # cluster_id of the cluster in a given position
        self.cluster_id = np.full((self.n, self.t), -1)
        # top left most position of each cluster indexed by cluster_id
        self.cluster_positions = {}
        self.charged_clusters_exist = False
        # order of charged clusters only or if only neutrals exist, the horizontally winding clusters
        self.charged_cluster_order = []
        # order of nested neutral clusters indexed by their surrounding cluster/left charged neighbor
        self.cluster_order = {}
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
                previous_coordinates = (x, y)
                self.cluster_positions[cluster_nr] = (x, y)
                # Go around a cluster loop
                loop_closed = False
                while not loop_closed:
                    visited[new_coordinates] = True
                    self.cluster_id[new_coordinates] = cluster_nr  # give cluster its ID
                    # update x and y to next position in cluster loop
                    new_coordinates, loop_closed, direction, previous_coordinates = self._cluster_loop_step(
                        new_coordinates, previous_coordinates, (x, y))

                # look where to find next cluster
                cluster_nr += 1
        self.n_clusters = cluster_nr

        # charge of each cluster in order cluster_id
        self.cluster_charge = np.zeros(self.n_clusters)
        # left neighbor going counterclockwise of every neutral cluster
        self.cluster_group = np.full(self.n_clusters, -1)
        # saves the nr of flip possibilites for +- and -+ starting from the corresponding cluster
        self.cluster_combinations = np.zeros((self.n_clusters, 2))
        for cluster in range(self.n_clusters):
            self.cluster_order[cluster] = []
        self.flip = np.zeros(self.n_clusters)

    def _cluster_loop_step(self, current_position, last_visited, start_of_loop):
        x = current_position[0]
        y = current_position[1]

        up = 0
        right = 1
        down = 2
        left = 3

        direction = -1

        coord_bond_up_left = (x - 1) % self.n, (y - 1) % self.t
        coord_bond_down_left = (x - 1) % self.n, y % self.t
        coord_bond_down_right = x % self.n, y % self.t
        coord_bond_up_right = x % self.n, (y - 1) % self.t

        visited_coord_up = (last_visited[0] == x and last_visited[1] == (y - 1) % self.t)
        visited_coord_right = ((x + 1) % self.n == last_visited[0] and y == last_visited[1])
        visited_coord_down = (last_visited[0] == x and last_visited[1] == (y + 1) % self.t)
        visited_coord_left = (last_visited[0] == (x - 1) % self.n and last_visited[1] == y)

        if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
            if not self.bond[coord_bond_up_left] and not visited_coord_up:
                y -= 1
                direction = up
            elif self.bond[coord_bond_down_right] and not visited_coord_right:
                x += 1
                direction = right
            elif not self.bond[coord_bond_down_right] and not visited_coord_down:
                y += 1
                direction = down
            elif self.bond[coord_bond_up_left] and not visited_coord_left:
                x -= 1
                direction = left
        elif (x % 2 == 1 and y % 2 == 0) or (x % 2 == 0 and y % 2 == 1):
            if not self.bond[coord_bond_up_right] and not visited_coord_up:
                y -= 1
                direction = up
            elif self.bond[coord_bond_up_right] and not visited_coord_right:
                x += 1
                direction = right
            elif not self.bond[coord_bond_down_left] and not visited_coord_down:
                y += 1
                direction = down
            elif self.bond[coord_bond_down_left] and not visited_coord_left:
                x -= 1
                direction = left
        x = x % self.n  # boundary conditions
        y = y % self.t
        if (x, y) == start_of_loop:
            loop_closed = True
        else:
            loop_closed = False
        return (x, y), loop_closed, direction, current_position

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
                self.charged_clusters_exist = True
        self.charge_combinations = np.zeros((self.n_clusters, 2))

    def _correct_positions(self):
        # correct the charged one that is closest from the left to [0,0]
        if len(self.charged_cluster_order) > 0:
            # correct the top leftmost position of leftmost charged cluster
            x = self.cluster_positions[self.charged_cluster_order[-1]][0]
            while self.cluster_id[x, 0] != self.charged_cluster_order[0]:
                x = (x + 1) % self.n
            self.cluster_positions[self.charged_cluster_order[0]] = (x, 0)
        corrected = []
        for j in range(self.t):
            if not (self.cluster_id[0, j] in corrected) and self.cluster_charge[self.cluster_id[0, j]] == 0:
                position = (0, j)
                previous_position = (0, j)
                corrected.append(self.cluster_id[0, j])
                closed_loop = False

                while not closed_loop:
                    position, closed_loop, direction, previous_position = self._cluster_loop_step(position,
                                                                                                  previous_position,
                                                                                                  (0, j))
                    if direction == 3 and position[0] == (
                            self.cluster_positions[self.cluster_id[position]][0] - 1) % self.n:
                        self.cluster_positions[self.cluster_id[position]] = position

    def _assign_groups(self):
        # recursive identification of nearest left charge or surrounding cluster for all neutral clusters
        for i in range(len(self.charged_cluster_order)):
            self._order_neighboring_clusters(self.charged_cluster_order[i], self.charged_cluster_order[i - 1],
                                             self.charged_cluster_order[i])

    def _order_neighboring_clusters(self, start_cluster_id, left_group, right_group):
        closed_loop = False
        start_position = self.cluster_positions[start_cluster_id]
        position = start_position
        previous_position = start_position

        while not closed_loop:
            position, closed_loop, direction, previous_position = self._cluster_loop_step(position, previous_position,
                                                                                          start_position)
            x = position[0]
            y = position[1]
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
            self.cluster_order[neighbor_group].append(neighbor_id)
            self._order_neighboring_clusters(neighbor_id, neighbor_group, neighbor_id)

    def _calculate_neutral_combinations(self, start_cluster, plus_minus):
        result = np.array([0, 0])
        # multiply out product for clusters in the same level
        for i in range(len(self.cluster_order[start_cluster])):
            result = (result + 1) * (
                    self._calculate_neutral_combinations(self.cluster_order[start_cluster][i], not plus_minus) + 1) - 1
        # calculate the effect of the loop
        if self.cluster_charge[start_cluster] == 0:
            if plus_minus:
                result = np.array([result[0] + result[1] + 1, result[1]])
            else:
                result = np.array([result[0], result[0] + result[1] + 1])
        # save result
        if start_cluster >= 0:
            self.cluster_combinations[start_cluster] = result
        return result

    def _generate_neutral_flips(self, charged_cluster, boundary_charge, plus_minus):
        for cluster in self.cluster_order[charged_cluster]:
            if np.array_equal(self.cluster_combinations[cluster], np.array([1, 0])):
                if boundary_charge < 0:
                    self.flip[cluster] = random.random() < 0.5
            elif np.array_equal(self.cluster_combinations[cluster], np.array([0, 1])):
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

    @abstractmethod
    def _charge_automaton(self, row, charge_index, case_character):
        pass

    def _calculate_charge_combinations(self):
        self.charge_combinations = np.full((5, len(self.charged_cluster_order) + 1, 4), 0)

        # define legal final states
        self.charge_combinations[0, -1, 0] = 1
        self.charge_combinations[2, -1, 0] = 1
        self.charge_combinations[3, -1, 0] = 1

        for charge_index in range(len(self.charged_cluster_order) - 1, -1, -1):
            for row in range(5):
                if row == 2:
                    for case_character in range(4):
                        next_row, weight = self._charge_automaton(row, charge_index, case_character)
                        if not next_row == -1:
                            self.charge_combinations[row, charge_index, case_character] += weight * np.sum(
                                self.charge_combinations[next_row, charge_index + 1])
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
                        self.flip[charge] = 0  # dont flip charge
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
            elif row == 0 or row == 3:
                match case_character:
                    case 0:
                        self.flip[charge] = 0
                        self._generate_neutral_flips(charge, - self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                    case 1:
                        self.flip[charge] = 1
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
            else:
                match case_character:
                    case 0:
                        self.flip[charge] = 0
                        self._generate_neutral_flips(charge, self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)
                    case 1:
                        self.flip[charge] = 1
                        self._generate_neutral_flips(charge, - self.cluster_charge[charge],
                                                     self.cluster_charge[charge] < 0)

            row, weight = self._charge_automaton(row, charge_idx, case_character)

    # TODO do actual flipping
    def _flip(self):
        for i in range(self.n_clusters):
            if self.flip[i]:
                start_position = self.cluster_positions[i]
                new_coordinates = start_position
                previous_coordinates = start_position
                loop_closed = False
                while not loop_closed:
                    new_coordinates, loop_closed, direction, previous_coordinates = self._cluster_loop_step(
                        new_coordinates, previous_coordinates, start_position)
                    self.fermion[new_coordinates] = not self.fermion[new_coordinates]

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
        # if the charged cluster order begins with a negative charge, move the first charge to the back
        if self.cluster_charge[self.charged_cluster_order[0]] < 0:
            charged_cluster_0 = self.charged_cluster_order[0]
            for i in range(len(self.charged_cluster_order) - 1):
                self.charged_cluster_order[i] = self.charged_cluster_order[i + 1]
            self.charged_cluster_order[-1] = charged_cluster_0
        self._calculate_charge_combinations()

        n_flip_configs = 100000
        histogram = np.zeros(2 ** self.n_clusters)
        for i in range(n_flip_configs):
            self.flip = np.zeros(self.n_clusters)
            self._generate_flips()
            histogram[int("".join(str(int(k)) for k in self.flip), 2)] += 1

        self._flip()

        plt.plot(histogram, ".")
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid()
        plt.show()
        pass
