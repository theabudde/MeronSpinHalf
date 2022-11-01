import random
from abc import abstractmethod

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from itertools import product


class MeronAlgorithm:
    @abstractmethod
    def __init__(self, n, t, w_a, w_b, w_c, beta, mc_steps):
        # constants
        self.n = n
        self.t = t
        self.w_a = w_a
        self.w_b = w_b
        self.w_c = w_c
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
        self.gauge_field = np.zeros((self.n, self.t))

        # fermion lattice initialized to reference configuration
        self.fermion = np.full((self.n, self.t), False)
        for i in range(self.n // 2):
            for j in range(self.t):
                self.fermion[2 * i, j] = True

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
        if len(self.cluster_group) != 1:
            for x, y in product(range(self.n), range(self.t)):
                color = self._get_random_color(self.cluster_group[self.cluster_id[x, y]])
                draw.rectangle(((x - 0.5) * scale, (y - 0.5) * scale, (x + 0.5) * scale, (y + 0.5) * scale),
                               fill=color)

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
                self.cluster_positions[cluster_nr] = new_coordinates
                while True:
                    visited[new_coordinates] = True
                    self.cluster_id[new_coordinates] = cluster_nr  # give cluster its ID
                    previous_coordinates = new_coordinates
                    new_coordinates = self._cluster_loop_step(new_coordinates)
                    # correct position to be leftmost and at the top of a leftmost edge of the cluster
                    if (new_coordinates[0] == (self.cluster_positions[cluster_nr][0] - 1) % self.n
                        and new_coordinates[0] == (previous_coordinates[0] - 1) % self.n) \
                            or (new_coordinates[1] == (previous_coordinates[1] - 1) % self.t
                                and new_coordinates[0] == self.cluster_positions[cluster_nr][0]
                                and new_coordinates[1] == (self.cluster_positions[cluster_nr][1] - 1) % self.t):
                        self.cluster_positions[cluster_nr] = new_coordinates
                    if new_coordinates == (x, y):
                        break
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

        # TODO: Correct postions of clusters that wind both vertically and horizontally

    # returns 1 for clockwise, 0 for anticlockwise
    def _direction_cluster_is_traversed(self, top_left_position_of_cluster):
        if self.cluster_charge[self.cluster_id[top_left_position_of_cluster]] > 0:
            return 0
        if self.cluster_charge[self.cluster_id[top_left_position_of_cluster]] < 0:
            return 1
        x = top_left_position_of_cluster[0]
        y = top_left_position_of_cluster[1]
        if x % 2 == 0 and y % 2 == 1:
            return 1
        elif x % 2 == 1 and y % 2 == 0:
            return 0
        elif self.bond[(x + 1) % self.n, (y + 1) % self.t]:
            return 1
        else:
            return 0

    # returns True if it is a right neighbor and False if it is a left neighbor
    # not legal for n == 2 or t == 2 because direction is not well defined
    def _is_right_neighbor(self, own_id, own_position_where_they_are_direct_neighbors,
                           neighbor_position_where_they_are_direct_neighors, previous_position_traversed):
        own_x = own_position_where_they_are_direct_neighbors[0]
        own_y = own_position_where_they_are_direct_neighbors[1]
        neighbor_x = neighbor_position_where_they_are_direct_neighors[0]
        neighbor_y = neighbor_position_where_they_are_direct_neighors[1]
        direction_headed_0 = (
            (own_x - previous_position_traversed[0]) % self.n, (own_y - previous_position_traversed[1]) % self.t)
        direction_neighbor = ((neighbor_x - own_x) % self.n, (neighbor_y - own_y) % self.n)
        next_own_pos = self._cluster_loop_step(own_position_where_they_are_direct_neighbors)
        direction_headed_1 = ((next_own_pos[0] - own_x) % self.n, (next_own_pos[1] - own_y) % self.t)

        result = False
        if direction_headed_0 == (0, (-1) % self.t):
            if direction_headed_1 == (0, (-1) % self.t):
                if direction_neighbor == (1, 0):
                    result = True
            elif direction_headed_1 == ((-1) % self.n, 0):
                result = True
        elif direction_headed_0 == (1, 0):
            if direction_headed_1 == (1, 0):
                if direction_neighbor == (0, 1):
                    result = True
            elif direction_headed_1 == (0, (-1) % self.t):
                result = True
        elif direction_headed_0 == (0, 1):
            if direction_headed_1 == (0, 1):
                if direction_neighbor == ((-1) % self.n, 0):
                    result = True
            elif direction_headed_1 == (1, 0):
                result = True
        elif direction_headed_0 == ((-1) % self.n, 0):
            if direction_headed_1 == ((-1) % self.n, 0):
                if direction_neighbor == (0, (-1) % self.t):
                    result = True
            elif direction_headed_1 == (0, 1):
                result = True

        if not self._direction_cluster_is_traversed(self.cluster_positions[own_id]):
            return not result
        return result

    def _neighbors(self, start_cluster_id):
        left_neighbors = []
        right_neighbors = []
        start_position = self.cluster_positions[start_cluster_id]
        position = start_position
        while True:
            previous_position = position
            position = self._cluster_loop_step(position)

            x = position[0]
            y = position[1]

            up = x, (y - 1) % self.t
            right = (x + 1) % self.n, y
            down = x, (y + 1) % self.t
            left = (x - 1) % self.n, y

            for neighbor_pos in [up, right, down, left]:
                neighbor = self.cluster_id[neighbor_pos]
                if neighbor != start_cluster_id and neighbor not in left_neighbors and neighbor not in right_neighbors and \
                        self.cluster_charge[neighbor] == 0:
                    if not self._is_right_neighbor(start_cluster_id, position, neighbor_pos, previous_position):
                        left_neighbors.append(neighbor)
                    else:
                        right_neighbors.append(neighbor)
            if position == start_position:
                break
        return left_neighbors, right_neighbors

    def _assign_groups_with_charges(self):
        # recursive identification of nearest left charge or surrounding cluster for all neutral clusters
        for i in range(len(self.charged_cluster_order)):
            self._order_neighboring_clusters(self.charged_cluster_order[i], self.charged_cluster_order[i - 1],
                                             self.charged_cluster_order[i])

    # will mark all inner neighbors and the direct outer neighbors of cluster start_cluster_id
    def _order_neighboring_clusters(self, start_cluster_id, outer_group, inner_group):
        outer_neighbors, inner_neighbors = self._neighbors(start_cluster_id)
        for outer_neighbor in outer_neighbors:
            self._mark_neighboring_clusters(outer_neighbor, outer_group)
        for inner_neighbor in inner_neighbors:
            self._mark_neighboring_clusters(inner_neighbor, inner_group)

    def _mark_neighboring_clusters(self, neighbor_id, neighbor_group):
        if self.cluster_charge[neighbor_id] == 0 and self.cluster_group[neighbor_id] == -1:
            self.cluster_group[neighbor_id] = neighbor_group
            self.cluster_order[neighbor_group].append(neighbor_id)
            self._order_neighboring_clusters(neighbor_id, neighbor_group, neighbor_id)

    def _group_neighboring_clusters_inside_start_cluster(self, start_cluster_id, current_cluster, outer_group):
        outer_neighbors, inner_neighbors = self._neighbors(current_cluster)
        for outer_neighbor in outer_neighbors:
            if not outer_neighbor == start_cluster_id and self.cluster_group[outer_neighbor] == -1:
                self.cluster_group[outer_neighbor] = outer_group
                self.cluster_order[outer_group].append(outer_neighbor)
                self._group_neighboring_clusters_inside_start_cluster(start_cluster_id, outer_neighbor, outer_group)
        for inner_neighbor in inner_neighbors:
            if self.cluster_group[inner_neighbor] == -1:
                self.cluster_group[inner_neighbor] = current_cluster
                self.cluster_order[current_cluster].append(inner_neighbor)
                self._group_neighboring_clusters_inside_start_cluster(start_cluster_id, inner_neighbor, current_cluster)

    # returns current_clusters neighbors so data can be reused
    def _group_neighboring_clusters_inside_cluster(self, cluster):
        outer_neighbors, inner_neighbors = self._neighbors(cluster)
        for inner_neighbor in inner_neighbors:
            if self.cluster_group[inner_neighbor] == -1:
                self.cluster_group[inner_neighbor] = cluster
                self.cluster_order[cluster].append(inner_neighbor)
                self._group_neighboring_clusters_inside_start_cluster(cluster, inner_neighbor, cluster)
        return outer_neighbors, inner_neighbors

    def _assign_groups_only_neutrals(self, start_cluster):
        # dictionary because it is ordered but does not allow duplicates
        cluster_queue = [start_cluster]
        outer_group = -2
        self.cluster_order[-2] = []
        queue_index = 0
        while True:
            current_cluster = cluster_queue[queue_index]
            self.cluster_group[current_cluster] = outer_group
            self.cluster_order[outer_group].append(current_cluster)
            outer_neighbors, inner_neighbors = self._group_neighboring_clusters_inside_cluster(current_cluster)
            for outer_neighbor in outer_neighbors:
                if outer_neighbor not in cluster_queue:
                    outer_neighbors_neighbors, inner_neighbors_neighbor = self._neighbors(outer_neighbor)
                    if current_cluster in inner_neighbors_neighbor:
                        for cluster in self.cluster_order[-2]:
                            self.cluster_group[cluster] = outer_neighbor
                        self.cluster_order[outer_neighbor] = self.cluster_order[-2]
                        self.cluster_order.pop(-2)
                        self._group_neighboring_clusters_inside_cluster(outer_neighbor)
                        self._assign_groups_only_neutrals(outer_neighbor)
                        return
                    cluster_queue.append(outer_neighbor)
            queue_index += 1
            if queue_index >= len(cluster_queue):
                break

    def _test_group_assignment(self):
        if -1 in self.cluster_group:
            raise 'Not all clusters have been assigned a group'
        # TODO: Think of more creative tests

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
                        reweight_factor *= (self.w_a - 2 * self.w_c) / normalizing_factor
                    else:
                        reweight_factor *= (self.w_a + 2 * self.w_c) / normalizing_factor
                else:
                    raise ("fermion got flipped wrong")
        return reweight_factor

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
            self.draw_bonds()
            raise 'gauss law broken'

    def mc_step(self):
        self._assign_bonds()
        self.draw_bonds()
        self._reset()
        self._find_clusters()
        self._identify_charged_clusters()
        self._assign_groups_with_charges()
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
