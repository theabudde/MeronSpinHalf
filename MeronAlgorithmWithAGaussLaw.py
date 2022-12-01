from MeronAlgorithm import MeronAlgorithm
import numpy as np
from abc import abstractmethod
from itertools import product
from PIL import Image, ImageDraw


class MeronAlgorithmWithAGaussLaw(MeronAlgorithm):
    @abstractmethod
    def __init__(self, n, t, w_a, w_b, mc_steps):
        MeronAlgorithm.__init__(self, n, t, w_a, w_b, mc_steps)
        # charge of each cluster in order cluster_id
        self.cluster_charge = np.array([0])
        # order of charged clusters only or if only neutrals exist, the horizontally winding clusters, starts with negative cluster
        self.charged_cluster_order = []
        # left neighbor going counterclockwise of every neutral cluster
        self.cluster_group = np.array([0])
        # order of nested neutral clusters indexed by their surrounding cluster/left charged neighbor
        self.cluster_order = {}
        # saves the nr of flip possibilites for +- and -+ starting from the corresponding cluster
        self.cluster_combinations = np.array([])
        self.charge_combinations = np.array([])
        self.gauge_field = np.zeros((self.n, self.t))

    def _reset(self):
        MeronAlgorithm._reset(self)
        self.charged_clusters_exist = False
        # order of charged clusters only or if only neutrals exist, the horizontally winding clusters
        self.charged_cluster_order = []
        # order of nested neutral clusters indexed by their surrounding cluster/left charged neighbor
        self.cluster_order = {}

    def _set_sizes_of_arrays(self):
        self.cluster_charge = np.zeros(self.n_clusters)
        self.cluster_group = np.full(self.n_clusters, -1)
        self.cluster_combinations = np.zeros((self.n_clusters, 2))
        for cluster in range(self.n_clusters):
            self.cluster_order[cluster] = []

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

    def _identify_charged_clusters(self):
        # determine cluster's charges
        self.cluster_charge = np.zeros(self.n_clusters)
        for i in range(self.n):
            if i % 2:
                self.cluster_charge[self.cluster_id[i, 0]] += 1
            else:
                self.cluster_charge[self.cluster_id[i, 0]] -= 1
        if np.max(self.cluster_charge) > 0:
            self.charged_clusters_exist = True

        if self.charged_clusters_exist:
            # TODO find out whether there are an even or odd nr of consecutive occupancies and only add to order if its odd
            # determine order of charged clusters
            saved_positive_cluster = -1
            saved_positive_cluster_exists = False
            is_first_cluster = True
            for i in range(self.n):
                if self.cluster_charge[self.cluster_id[i, 0]] != 0 and not self.cluster_id[
                                                                               i, 0] in self.charged_cluster_order:
                    n_consecutive_cluster_occupancies = 1
                    for j in range(1, self.n):
                        if self.cluster_id[i, 0] == self.cluster_id[(i + j) % self.n, 0]:
                            n_consecutive_cluster_occupancies += 1
                        elif self.cluster_charge[self.cluster_id[(i + j) % self.n, 0]] == 0:
                            continue
                        else:
                            break
                    for j in range(1, self.n):
                        if self.cluster_id[i, 0] == self.cluster_id[(i - j) % self.n, 0]:
                            n_consecutive_cluster_occupancies += 1
                        elif self.cluster_charge[self.cluster_id[(i - j) % self.n, 0]] == 0:
                            continue
                        else:
                            break
                    if n_consecutive_cluster_occupancies % 2 == 1:
                        if is_first_cluster:
                            is_first_cluster = False
                            if self.cluster_charge[self.cluster_id[i, 0]] > 0:
                                saved_positive_cluster = self.cluster_id[i, 0]
                                saved_positive_cluster_exists = True
                            else:
                                self.charged_cluster_order.append(self.cluster_id[i, 0])
                        elif self.cluster_id[i, 0] != saved_positive_cluster:
                            if n_consecutive_cluster_occupancies % 2 == 1:
                                self.charged_cluster_order.append(self.cluster_id[i, 0])

            if saved_positive_cluster_exists:
                self.charged_cluster_order.append(saved_positive_cluster)
            self.charge_combinations = np.zeros((self.n_clusters, 2))

    def _identify_horizontal_winding(self):
        self.horizontal_winding_order = []
        self.horizontally_winding_clusters_exist = False
        # determine cluster's winding
        self.horizontal_winding = np.zeros(self.n_clusters)
        for i in range(self.t):
            if i % 2:
                self.horizontal_winding[self.cluster_id[0, i]] += 1
            else:
                self.horizontal_winding[self.cluster_id[0, i]] -= 1

        if np.max(self.horizontal_winding) > 0:
            self.horizontally_winding_clusters_exist = True

        if self.horizontally_winding_clusters_exist:
            # determine order of charged clusters
            saved_positive_cluster = -1
            saved_positive_cluster_exists = False
            is_first_cluster = True
            saved_cluster_position = (-1, -1)
            # TODO: make sure negative cluster is first and count from bottom to top so neutrals are on right side
            # determine order of winding clusters
            for i in range(self.t - 1, -1, -1):
                if self.horizontal_winding[self.cluster_id[0, i]] != 0 and not self.cluster_id[
                                                                                   0, i] in self.horizontal_winding_order:
                    n_consecutive_cluster_occupancies = 1
                    for j in range(1, self.t):
                        if self.cluster_id[0, i] == self.cluster_id[0, (i + j) % self.t]:
                            n_consecutive_cluster_occupancies += 1
                        elif self.cluster_charge[self.cluster_id[0, (i + j) % self.t]] == 0:
                            continue
                        else:
                            break
                    for j in range(1, self.t):
                        if self.cluster_id[0, i] == self.cluster_id[0, (i - j) % self.t]:
                            n_consecutive_cluster_occupancies += 1
                        elif self.cluster_charge[self.cluster_id[0, (i - j) % self.t]] == 0:
                            continue
                        else:
                            break
                    if n_consecutive_cluster_occupancies % 2 == 1:
                        if is_first_cluster:
                            is_first_cluster = False
                            if self.horizontal_winding[self.cluster_id[0, i]] > 0:
                                saved_positive_cluster = self.cluster_id[0, i]
                                saved_positive_cluster_exists = True
                                saved_cluster_position = (0, i)
                            else:
                                self.horizontal_winding_order.append(self.cluster_id[0, i])
                                self.cluster_positions[self.cluster_id[0, i]] = (0, i)
                        elif self.cluster_id[0, i] != saved_positive_cluster:
                            self.horizontal_winding_order.append(self.cluster_id[0, i])
                            self.cluster_positions[self.cluster_id[0, i]] = (0, i)
            if saved_positive_cluster_exists:
                self.horizontal_winding_order.append(saved_positive_cluster)
                self.cluster_positions[saved_positive_cluster] = saved_cluster_position

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
        direction_neighbor = ((neighbor_x - own_x) % self.n, (neighbor_y - own_y) % self.t)
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

    @abstractmethod
    def charge_automaton(self, row, charge_index, case_character):
        pass
