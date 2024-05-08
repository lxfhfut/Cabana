import cv2
import time
import numpy as np
from log import Log
import networkx as nx
from numba import jit
import imageio.v3 as iio
import scipy.ndimage as ndi
from utils import add_colorbar
from skimage.morphology import skeletonize, remove_small_holes
from detector import FibreDetector
from scipy.interpolate import splprep, splev


class SkeletonAnalyzer:
    def __init__(self, skel_thresh=20, branch_thresh=10, hole_threshold=8, dark_line=True):
        self.pts_image = None
        self.skel_thresh = skel_thresh
        self.branch_thresh = branch_thresh
        self.hole_thresh = hole_threshold
        self.proj_area = 0.0
        self.num_tips = 0
        self.num_branches = 0
        self.total_length = 0.0
        self.growth_unit = 0.0
        self.frac_dim = 0.0
        self.lacunarity = 0.0
        self.avg_curve_long = 0.0
        self.avg_curve_all = 0.0
        self.avg_curve_spline = 0.0
        self.raw_image = None
        self.skel_image = None
        self.pruned_image = None
        self.key_pts_image = None
        self.long_path_image = None
        self.curve_map_long = None
        self.curve_map_all = None
        self.length_map_long = None
        self.length_map_all = None
        self.subgraphs = []
        self.dark_line = dark_line
        self.FOREGROUND = 255
        self.BACKGROUND = 0

    def reset(self):
        self.proj_area = 0.0
        self.num_tips = 0
        self.num_branches = 0
        self.total_length = 0.0
        self.growth_unit = 0.0
        self.frac_dim = 0.0
        self.lacunarity = 0.0
        self.avg_curve_long = 0.0
        self.avg_curve_all = 0.0
        self.avg_curve_spline = 0.0
        self.raw_image = None
        self.skel_image = None
        self.pruned_image = None
        self.key_pts_image = None
        self.long_path_image = None
        self.curve_map_long = None
        self.curve_map_all = None
        self.length_map_long = None
        self.length_map_all = None
        self.subgraphs = []

    @staticmethod
    @jit(nopython=True)
    def count_neighbors(skel_image, y, x, radius=1, val=1):
        count = 0
        for i in range(y-radius, y+radius+1):
            for j in range(x-radius, x+radius+1):
                if skel_image[i, j] == val and ((y != i) or (x != j)):
                    count += 1
        return count

    @staticmethod
    def traverse_skeletons(skel_image, end_points, brh_points, foreground):
        # Initialize a dictionary to store lengths and paths
        lengths_paths = []
        visited = set()

        if not end_points and not brh_points:
            return lengths_paths
        elif end_points and not brh_points:  # only end points available
            if len(end_points) == 1:
                Log.logger.warning("Isolated point found. Ignored!")
                return lengths_paths
            elif len(end_points) > 2:
                Log.logger.warning("More than 2 points for a branch. Ignored!")
                return lengths_paths
            else:
                stack = [(end_points[0], 0, [end_points[0]])]
                while stack:
                    current, length, path = stack.pop()

                    if current == end_points[1]:
                        lengths_paths.append((path[0], current, length, path, 'end-to-end'))
                        continue  # Skip adding neighbors if destination is reached

                    visited.add(current)

                    for neighbor in SkeletonAnalyzer.get_neighbors(skel_image, current, foreground):
                        if neighbor not in visited:
                            dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                           (neighbor[1] - current[1]) ** 2.0)
                            new_path = path + [neighbor]
                            stack.append((neighbor, length + dist, new_path))

        elif brh_points and not end_points:  # only branch points available
            # Trace from branch points to branch points
            for src in brh_points:
                # The last element in the tuple indicates if another point has been visited
                stack = [(src, 0, [src], False)]

                while stack:
                    current, length, path, visited_other = stack.pop()

                    if current == src and visited_other:
                        # Found a self-loop
                        lengths_paths.append((src, src, length, path, 'brh-to-brh'))
                        continue

                    if current in visited:
                        continue  # Avoid reprocessing nodes except for the self-loop check

                    visited.add(current)
                    neighbors = SkeletonAnalyzer.get_neighbors(skel_image, current, foreground)
                    is_reached = [(n != src) and (n in brh_points) for n in neighbors]
                    if any(is_reached):
                        neighbor = neighbors[np.where(is_reached)[0][0]]
                        dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                       (neighbor[1] - current[1]) ** 2.0)
                        new_path = path + [neighbor]
                        lengths_paths.append((path[0], neighbor, length + dist, new_path, 'brh-to-brh'))
                        continue

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                           (neighbor[1] - current[1]) ** 2.0)
                            new_path = path + [neighbor]
                            stack.append((neighbor, length + dist, new_path, True if neighbor != src else visited_other))

        else:  # neither end points nor branch points are empty
            # Trace from end points to branch points
            stack = [(src, 0, [src]) for src in end_points]
            while stack:
                current, length, path = stack.pop()

                if current in brh_points:
                    lengths_paths.append((path[0], current, length, path, 'end-to-brh'))
                    continue

                visited.add(current)

                # If any neighbors is in dst points
                neighbors = SkeletonAnalyzer.get_neighbors(skel_image, current, foreground)
                neighbors_reached = [n for n in neighbors if n in brh_points]
                if neighbors_reached:
                    neighbor = neighbors_reached[0]
                    dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                   (neighbor[1] - current[1]) ** 2.0)
                    new_path = path + [neighbor]
                    lengths_paths.append((path[0], neighbor, length + dist, new_path, 'end-to-brh'))
                    continue

                for neighbor in neighbors:
                    if neighbor not in visited:
                        dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                       (neighbor[1] - current[1]) ** 2.0)
                        new_path = path + [neighbor]
                        stack.append((neighbor, length + dist, new_path))

            # Trace from branch points to branch points
            for src in brh_points:
                # The last element in the tuple indicates if another point has been visited
                stack = [(src, 0, [src], False)]

                while stack:
                    current, length, path, visited_other = stack.pop()

                    if current == src and visited_other:
                        # Found a self-loop
                        lengths_paths.append((src, current, length, path, 'brh-to-brh'))
                        continue

                    if current in visited:
                        continue

                    visited.add(current)
                    neighbors = SkeletonAnalyzer.get_neighbors(skel_image, current, foreground)
                    is_reached = [(n != src) and (n in brh_points) for n in neighbors]
                    if any(is_reached):
                        neighbor = neighbors[np.where(is_reached)[0][0]]
                        dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                       (neighbor[1] - current[1]) ** 2.0)
                        new_path = path + [neighbor]
                        lengths_paths.append((src, neighbor, length + dist, new_path, 'brh-to-brh'))
                        continue

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            dist = np.sqrt((neighbor[0] - current[0]) ** 2.0 +
                                           (neighbor[1] - current[1]) ** 2.0)
                            new_path = path + [neighbor]
                            stack.append((neighbor, length + dist, new_path, True if neighbor != src else visited_other))

        return lengths_paths

    @staticmethod
    @jit(nopython=True)
    def get_neighbors(skel_image, point, foreground):
        """
        Get neighboring foreground points of a given point.

        Args:
        - image: Numpy array of the binary image.
        - point: Current point (row, col).

        Returns:
        - List of neighboring foreground points.
        """
        row, col = point
        neighbors = []
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i != row or j != col) and 0 <= i < skel_image.shape[0] and 0 <= j < skel_image.shape[1]:
                    if skel_image[i, j] == foreground:
                        neighbors.append((i, j))
        return neighbors

    @staticmethod
    @jit(nopython=True)
    def is_branchpoint(skel_image, row, col, foreground):
        height, width = skel_image.shape[:2]
        # Search north and south
        for y in [row - 1, row + 1]:
            previous = -1
            for x in range(col - 1, col + 2):
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                if skel_image[y, x] == foreground and previous == foreground:
                    return False
                previous = skel_image[y, x]

        # Search east and west
        for x in [col - 1, col + 1]:
            previous = -1
            for y in range(row - 1, row + 2):
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                if skel_image[y, x] == foreground and previous == foreground:
                    return False
                previous = skel_image[y, x]

        return True

    @staticmethod
    def longest_path(graph, wgt='length'):
        # First, get all shortest path lengths using Dijkstra's algorithm
        all_distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight=wgt))

        # Initialize variables to keep track of the longest path found
        max_length = 0
        max_path_nodes = (None, None)

        # Iterate over all pairs and find the one with the greatest distance
        for source, target_dict in all_distances.items():
            for target, distance in target_dict.items():
                if distance > max_length:
                    max_length = distance
                    max_path_nodes = (source, target)

        # Extract the longest path using the nodes found
        if max_path_nodes[0] is not None and max_path_nodes[1] is not None:
            # We use nx.dijkstra_path to get the path itself
            longest_path = nx.dijkstra_path(graph, max_path_nodes[0], max_path_nodes[1], weight=wgt)
            return longest_path, max_length
        else:
            return [], 0

    @staticmethod
    @jit(nopython=True)
    def is_endpoint(skel_image, row, col, background):
        """
        Determine if a pixel is an endpoint based on the longest chain of background pixels.

        Args:
        - skel_image: Numpy array of the binary skeleton image.
        - pixel_row: Row index of the pixel.
        - pixel_col: Column index of the pixel.

        Returns:
        - True if the pixel is an endpoint, False otherwise.
        """
        # Define the 8-connected neighborhood relative positions
        neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        longest_chain = 0
        current_chain = 0

        # Start checking from the pixel's right neighbor and move in a circular manner
        for i in range(len(neighbors) * 2):  # Multiply by 2 to allow wrapping around the neighborhood
            dx, dy = neighbors[i % len(neighbors)]
            ny, nx = row + dy, col + dx

            # Check if the neighbor is within the image bounds
            if 0 <= ny < skel_image.shape[0] and 0 <= nx < skel_image.shape[1]:
                if skel_image[ny, nx] == background:  # Background pixel
                    current_chain += 1
                    longest_chain = max(longest_chain, current_chain)
                else:  # Foreground pixel, reset chain length
                    current_chain = 0
            else:
                # Treat out-of-bounds as background to continue the chain.
                current_chain += 1

        # if the longest chain of background pixels is 5 or more, it's considered an endpoint.
        return longest_chain >= 5

    def construct_graphs(self):
        """
        Find endpoints in a binary image based on the longest chain of background pixels.

        Args:
        - binary_image: Numpy array of the binary image.

        Returns:
        - A binary image where endpoints are marked with 1s.
        """
        # Label each connected components
        eight_con = np.ones((3, 3), dtype=int)
        labels, num = ndi.label(self.skel_image, eight_con)
        if self.skel_thresh > 0.:
            segment_sums = ndi.sum(self.skel_image, labels, range(1, num + 1))
            labels_remove = np.where(segment_sums <= self.skel_thresh * self.FOREGROUND)[0]

            for label in labels_remove:
                self.skel_image[np.where(labels == label + 1)] = self.BACKGROUND

            # Relabel after deleting short skeletons.
            labels, num = ndi.label(self.skel_image, eight_con)

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        num_neighbors = ndi.convolve((self.skel_image == self.FOREGROUND).astype(np.uint8),
                                     kernel, mode="constant")

        height, width = self.skel_image.shape[:2]

        def get_binary_3x3(img, row, col):
            i1 = False if row - 1 < 0 or col - 1 < 0 else img[row - 1, col - 1] == self.FOREGROUND
            i2 = False if row - 1 < 0 else img[row - 1, col] == self.FOREGROUND
            i3 = False if row - 1 < 0 or col + 1 >= width else img[row - 1, col + 1] == self.FOREGROUND
            i4 = False if col - 1 < 0 else img[row, col - 1] == self.FOREGROUND
            i5 = True
            i6 = False if col + 1 >= width else img[row, col + 1] == self.FOREGROUND
            i7 = False if row + 1 >= height or col - 1 < 0 else img[row + 1, col - 1] == self.FOREGROUND
            i8 = False if row + 1 >= height else img[row + 1, col] == self.FOREGROUND
            i9 = False if row + 1 >= height or col + 1 >= width else img[row + 1, col + 1] == self.FOREGROUND
            return np.array([[i1, i2, i3], [i4, i5, i6], [i7, i8, i9]])

        # Localize end and branch points
        selems_2 = list()
        selems_2.append(np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        selems_2.append(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))
        selems_2 = [np.rot90(selems_2[i], k=j) for i in range(2) for j in range(4)]

        selems_3 = list()
        selems_3.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
        selems_3.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
        selems_3.append(np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]]))
        selems_3 = [np.rot90(selems_3[i], k=j) for i in range(5) for j in range(4)]
        selems_4 = selems_3.copy()
        selems_4.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        selems_4.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
        selems_tmp = list()
        selems_tmp.append(np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]]))
        selems_tmp.append(np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]]))
        selems_tmp.append(np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]]))
        selems_4.extend([np.rot90(selems_tmp[i], k=j) for i in range(3) for j in range(4)])

        self.pruned_image = self.skel_image.copy()
        # removed_edges_count = 0
        # removed_end_nodes_count = 0
        # removed_brh_nodes_count = 0
        for label in range(1, num+1):  # For each segment
            # Canvas image to draw a segment at a time
            canvas = np.zeros(self.skel_image.shape, dtype=np.uint8)
            row_idx, col_idx = np.where(labels == label)
            canvas[row_idx, col_idx] = self.FOREGROUND
            row_start, row_end = row_idx.min(), row_idx.max()
            col_start, col_end = col_idx.min(), col_idx.max()

            endpoints, branchpoints = [], []
            for row in range(row_start, row_end+1):
                for col in range(col_start, col_end+1):
                    if canvas[row, col] == self.FOREGROUND:
                        # SkeletonAnalyzer.is_endpoint(canvas, row, col, self.BACKGROUND)
                        if num_neighbors[row, col] == 1:
                            endpoints.append((row, col))
                        elif num_neighbors[row, col] == 2:
                            binary = get_binary_3x3(self.skel_image, row, col)
                            for selem in selems_2:
                                if not np.logical_xor(binary, selem).any():
                                    endpoints.append((row, col))
                                    break
                        elif num_neighbors[row, col] == 3:
                            binary = get_binary_3x3(self.skel_image, row, col)
                            for selem in selems_3:
                                if not np.logical_xor(binary, selem).any():
                                    branchpoints.append((row, col))
                                    break
                        elif num_neighbors[row, col] == 4:
                            binary = get_binary_3x3(self.skel_image, row, col)
                            for selem in selems_4:
                                if not np.logical_xor(binary, selem).any():
                                    branchpoints.append((row, col))
                                    break
                        elif num_neighbors[row, col] > 4:
                            branchpoints.append((row, col))

            # Trace paths between end connectivity points
            # pts_image = np.repeat(canvas[:, :, None], 3, axis=2)
            G = nx.Graph()
            for end_pt in endpoints:
                G.add_node(end_pt, node_type="end-point")
            for brh_pt in branchpoints:
                G.add_node(brh_pt, node_type="branch-point")

            lengths_paths = SkeletonAnalyzer.traverse_skeletons(canvas, endpoints, branchpoints, self.FOREGROUND)
            # Be careful when adding edges, only one edge is added if there are multiple edges between two nodes
            for src, dst, length, path, typ in lengths_paths:
                G.add_edge(src, dst, length=length, path=path, type=typ)
                # print(f"{src}--->{dst}: {typ} {length}")

            # Prune short branches
            edges_to_remove = [edge for edge in G.edges(data=True)
                               if edge[2]['length'] <= self.branch_thresh and
                               edge[2]['type'] == 'end-to-brh']
            G.remove_edges_from([edge for edge in edges_to_remove])

            # Remove isolated nodes
            isolated_nodes = [node[0] for node in G.nodes(data=True)
                              if G.degree(node[0]) == 0]
            G.remove_nodes_from(isolated_nodes)

            self.subgraphs.append(G)

            #  Update pruned image
            for edge in edges_to_remove:
                path_array = np.asarray(edge[2]['path'])[:-1, :]
                self.pruned_image[path_array[:, 0], path_array[:, 1]] = self.BACKGROUND

            for node in isolated_nodes:
                self.pruned_image[node[0], node[1]] = self.BACKGROUND

    def calc_curve_all(self, win_sz=11):
        curve_map = np.zeros_like(self.pruned_image, dtype=float)
        count_map = np.zeros_like(self.pruned_image, dtype=int)
        side = win_sz // 2
        for G in self.subgraphs:
            for u, v, a in G.edges(data=True):
                edge_path = a['path']
                if len(edge_path) > 0:
                    points = np.pad(np.asarray(edge_path), ((side, side), (0, 0)), mode='reflect', reflect_type='odd')
                    for j in range(side, len(points) - side):
                        theta1 = np.arctan2(
                            points[j - side, 0] - points[j, 0], points[j, 1] - points[j - side, 1])
                        theta2 = np.arctan2(
                            points[j, 0] - points[j + side, 0], points[j + side, 1] - points[j, 1])
                        angle_diff = abs(theta1 - theta2)
                        if angle_diff >= np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        curve_map[points[j, 0], points[j, 1]] += np.rad2deg(angle_diff)
                        count_map[points[j, 0], points[j, 1]] += 1
        calc_mask = count_map >= 1
        curve_map[calc_mask] /= count_map[calc_mask]
        self.curve_map_all = curve_map
        self.avg_curve_all = np.mean(curve_map[calc_mask]) if calc_mask.any() else 0.0
        # print(f"Mean curvature (win_sz={win_sz}): {np.mean(curve_map[calc_mask])}")
        # self.curve_map_all = add_colorbar(curve_map, clabel="Curvature (degrees)", cmap="inferno")

    def calc_curve_long(self, win_sz=11):
        # height, width = self.pruned_image.shape[:2]
        self.curve_map_long = np.zeros_like(self.pruned_image, dtype=float)
        side = win_sz // 2
        curvatures = []
        for G in self.subgraphs:
            longest_path = SkeletonAnalyzer.longest_path(G, "length")[0]
            trajectory = []
            if longest_path:  # is not empty
                src_node = longest_path[0]
                for mid_idx, dst_node in enumerate(longest_path[1:]):
                    edge_path = G.edges[src_node, dst_node]['path']
                    if mid_idx == 0:
                        trajectory.extend(edge_path)
                    else:
                        if trajectory[-1] == edge_path[0]:
                            trajectory.extend(edge_path[1:])
                        elif trajectory[-1] == edge_path[-1]:
                            trajectory.extend(edge_path[::-1][1:])
                        else:
                            # there seems a problem for some branches 头尾不衔接的情况
                            pass

                    src_node = dst_node

            if trajectory:
                points = np.pad(np.asarray(trajectory), ((side, side), (0, 0)), mode='reflect', reflect_type='odd')
                for j in range(side, len(points) - side):
                    theta1 = np.arctan2(
                        points[j - side, 0] - points[j, 0], points[j, 1] - points[j - side, 1])
                    theta2 = np.arctan2(
                        points[j, 0] - points[j + side, 0], points[j + side, 1] - points[j, 1])
                    angle_diff = abs(theta1 - theta2)
                    if angle_diff >= np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    self.curve_map_long[points[j, 0], points[j, 1]] = np.rad2deg(angle_diff)
                    curvatures.append(np.rad2deg(angle_diff))
        self.avg_curve_long = np.mean(curvatures) if curvatures else 0.0

    def points_test(self):
        def get_binary_3x3(img, row, col):
            i1 = False if row - 1 < 0 or col - 1 < 0 else img[row - 1, col - 1] == self.FOREGROUND
            i2 = False if row - 1 < 0 else img[row - 1, col] == self.FOREGROUND
            i3 = False if row - 1 < 0 or col + 1 >= width else img[row - 1, col + 1] == self.FOREGROUND
            i4 = False if col - 1 < 0 else img[row, col - 1] == self.FOREGROUND
            i5 = True
            i6 = False if col + 1 >= width else img[row, col + 1] == self.FOREGROUND
            i7 = False if row + 1 >= height or col - 1 < 0 else img[row + 1, col - 1] == self.FOREGROUND
            i8 = False if row + 1 >= height else img[row + 1, col] == self.FOREGROUND
            i9 = False if row + 1 >= height or col + 1 >= width else img[row + 1, col + 1] == self.FOREGROUND
            return np.array([[i1, i2, i3], [i4, i5, i6], [i7, i8, i9]])

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        num_neighbors = ndi.convolve((self.pruned_image == self.FOREGROUND).astype(np.uint8),
                                     kernel, mode="constant")
        self.pts_image = np.repeat(self.pruned_image[:, :, None], 3, axis=2)
        height, width = self.pruned_image.shape[:2]
        brh_pts_cnt = 0
        end_pts_cnt = 0
        selems_2 = list()
        selems_2.append(np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]]))
        selems_2.append(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))
        selems_2 = [np.rot90(selems_2[i], k=j) for i in range(2) for j in range(4)]

        selems_3 = list()
        selems_3.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
        selems_3.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
        selems_3.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
        selems_3.append(np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]]))
        selems_3 = [np.rot90(selems_3[i], k=j) for i in range(5) for j in range(4)]
        selems_4 = selems_3.copy()
        selems_4.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        selems_4.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
        selems_tmp = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
        selems_4.extend([np.rot90(selems_tmp, k=j) for j in range(4)])
        for row in range(height):
            for col in range(width):
                if self.skel_image[row, col] == self.FOREGROUND:
                    # SkeletonAnalyzer.is_endpoint(canvas, row, col, self.BACKGROUND)
                    if num_neighbors[row, col] == 1:
                        cv2.circle(self.pts_image, (col, row), 3, (0, 255, 0), 2)
                        end_pts_cnt += 1
                    elif num_neighbors[row, col] == 2:
                        binary = get_binary_3x3(self.skel_image, row, col)
                        for selem in selems_2:
                            if not np.logical_xor(binary, selem).any():
                                cv2.circle(self.pts_image, (col, row), 3, (0, 255, 0), 2)
                                end_pts_cnt += 1
                                break

                    elif num_neighbors[row, col] == 3:
                        binary = get_binary_3x3(self.skel_image, row, col)
                        for selem in selems_3:
                            if not np.logical_xor(binary, selem).any():
                                cv2.circle(self.pts_image, (col, row), 3, (255, 255, 0), 2)
                                brh_pts_cnt += 1
                                break

                    elif num_neighbors[row, col] == 4:
                        binary = get_binary_3x3(self.skel_image, row, col)
                        for selem in selems_4:
                            if not np.logical_xor(binary, selem).any():
                                cv2.circle(self.pts_image, (col, row), 3, (255, 255, 0), 2)
                                brh_pts_cnt += 1
                                break
                    elif num_neighbors[row, col] > 4:
                        cv2.circle(self.pts_image, (col, row), 3, (255, 255, 0), 2)
                        brh_pts_cnt += 1
        print(f"points_test: {end_pts_cnt} end points, {brh_pts_cnt} branch points.")

    def draw_key_points(self):
        # height, width = self.pruned_image.shape[:2]
        self.key_pts_image = np.repeat(self.pruned_image[:, :, None], 3, axis=2)
        brh_pts_cnt = 0
        end_pts_cnt = 0
        for G in self.subgraphs:
            for node, attrs in G.nodes(data=True):
                if G.degree(node) == 1:  # attrs['node_type'] == "end-point":
                    cv2.circle(self.key_pts_image, (node[1], node[0]), 3, (255, 0, 0), 2)
                    end_pts_cnt += 1
                elif G.degree(node) > 2:  # attrs['node_type'] == "branch-point":
                    cv2.circle(self.key_pts_image, (node[1], node[0]), 3, (255, 255, 0), 2)
                    brh_pts_cnt += 1
                else:
                    pass
        self.num_tips = end_pts_cnt
        self.num_branches = brh_pts_cnt
        # print(f"draw_key_points: {end_pts_cnt} end points, {brh_pts_cnt} branch points.")

    def calc_total_len(self):
        self.total_length = np.sum(self.pruned_image == self.FOREGROUND)

    def calc_growth_unit(self):
        self.growth_unit = 2.0 * self.total_length / (self.num_tips + self.num_branches) \
            if self.num_tips + self.num_branches > 0 else 0.0

    def calc_len_map_all(self):
        self.length_map_all = np.zeros_like(self.pruned_image, dtype=float)
        for G in self.subgraphs:
            for u, v, a in G.edges(data=True):
                edge_path = a['path']
                if len(edge_path) > 0:
                    points = np.asarray(edge_path)
                    traj_len = np.sum(
                        np.sqrt((points[1:, 0] - points[:-1, 0]) ** 2 + (points[1:, 1] - points[:-1, 1]) ** 2))
                    self.length_map_all[points[:, 0], points[:, 1]] = traj_len

    def calc_len_map_long(self):
        self.length_map_long = np.zeros_like(self.pruned_image, dtype=float)
        for G in self.subgraphs:
            longest_path = SkeletonAnalyzer.longest_path(G, "length")[0]
            trajectory = []
            if longest_path:  # is not empty
                src_node = longest_path[0]
                for mid_idx, dst_node in enumerate(longest_path[1:]):
                    edge_path = G.edges[src_node, dst_node]['path']
                    if mid_idx == 0:
                        trajectory.extend(edge_path)
                    else:
                        if trajectory[-1] == edge_path[0]:
                            trajectory.extend(edge_path[1:])
                        elif trajectory[-1] == edge_path[-1]:
                            trajectory.extend(edge_path[::-1][1:])
                        else:
                            pass

                    src_node = dst_node

            if trajectory:
                points = np.asarray(trajectory)
                traj_len = np.sum(np.sqrt((points[1:, 0] - points[:-1, 0]) ** 2 + (points[1:, 1] - points[:-1, 1]) ** 2))
                self.length_map_long[points[:, 0], points[:, 1]] = traj_len

    def draw_longest_path(self):
        self.long_path_image = np.repeat(self.pruned_image[:, :, None], 3, axis=2)

        for G in self.subgraphs:
            longest_path = SkeletonAnalyzer.longest_path(G, "length")[0]
            if longest_path:  # is not empty
                src_node = longest_path[0]
                for dst_node in longest_path[1:]:
                    edge_path = np.asarray(G.edges[src_node, dst_node]['path'])
                    cv2.polylines(self.long_path_image, [np.asarray(edge_path)[:, ::-1]],
                                  False, [255, 255, 0], 2)
                    src_node = dst_node

    def calc_frac_dim(self):
        height, width = self.pruned_image.shape[:2]

        # Minimal dimension of image
        p = min(self.pruned_image.shape)

        # Greatest power of 2 less than or equal to p
        n = 2 ** np.floor(np.log(p) / np.log(2)) - 2

        # Extract the exponent
        n = int(np.log(n) / np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2 ** np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            S = np.add.reduceat(
                np.add.reduceat(self.pruned_image == self.FOREGROUND,
                                np.arange(0, height, size), axis=0),
                np.arange(0, width, size), axis=1)
            counts.append(len(np.where((S > 0) & (S < size ** 2))[0]))

        # Check if all counts are zero
        if np.all(counts == 0):
            Log.logger.warning("All counts are zero. Fractal dimension cannot be computed meaningfully.")
            return None
        else:
            # Replace zero counts with a small positive number
            counts = np.maximum(counts, 1e-10)
        self.frac_dim = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

    def calc_lacunarity(self):
        mask_img = np.zeros_like(self.pruned_image)
        mask_img[self.pruned_image == self.FOREGROUND] = 1
        if np.count_nonzero(mask_img) == 0:
            Log.logger.warning("No foreground pixels. Lacunarity cannot be computed meaningfully.")
            return None
        else:
            self.lacunarity = abs(np.var(mask_img.flatten()) / np.mean(mask_img.flatten()) ** 2 - 1.0)

    def calc_curve_spline(self, s=3):
        curve_map = np.zeros_like(self.pruned_image, dtype=float)
        count_map = np.zeros_like(self.pruned_image, dtype=int)
        for G in self.subgraphs:
            for u, v, a in G.edges(data=True):
                edge_path = a['path']
                if len(edge_path) > 3:  # At least three points are needed for spline fitting
                    contour = np.asarray(edge_path)

                    tck, u = splprep(contour.T, s=s)

                    # Evaluate the spline to get points on the curve
                    points = np.asarray(splev(u, tck)).round().astype(int).T

                    # calculate the curvature of points
                    dx, dy = splev(u, tck, der=1)
                    ddx, ddy = splev(u, tck, der=2)

                    # Calculate curvature at each point
                    curvature = (np.abs(dx * ddy - dy * ddx) /
                                 ((dx ** 2 + dy ** 2 + np.finfo(float).eps) ** 1.5))

                    curve_map[points[:, 0], points[:, 1]] = curvature
                    count_map[points[:, 0], points[:, 1]] += 1
        calc_mask = count_map >= 1
        curve_map[calc_mask] /= count_map[calc_mask]
        self.avg_curve_spline = np.mean(curve_map[calc_mask]) if calc_mask.any() else 0.0

    @staticmethod
    def dilate_color(color_image, mask):
        # Get the height and width of the images
        height, width = mask.shape
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Compute the distance transform
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        # Get the nearest pixel coordinates from the non-dilated areas
        nearest_idx = np.round(np.argwhere(dist_transform == 0))

        # Create an output image initialized with the original color image
        output_image = np.copy(color_image)

        # For each pixel in the dilated mask that is not in the original mask
        for i in range(height):
            for j in range(width):
                if dilated_mask[i, j] != 0 and mask[i, j] == 0:
                    # Find the nearest non-zero pixel from the original mask
                    distances = np.sqrt((nearest_idx[:, 0] - i) ** 2 + (nearest_idx[:, 1] - j) ** 2)
                    nearest_pixel = nearest_idx[np.argmin(distances)]
                    # Copy the color from the nearest non-zero pixel
                    output_image[i, j] = color_image[nearest_pixel[0], nearest_pixel[1]]

        return output_image

    def calc_proj_area(self):
        self.proj_area = np.sum(self.raw_image == self.FOREGROUND)

    def analyze_image(self, image):
        self.raw_image = iio.imread(image) if isinstance(image, str) else image
        if self.raw_image.ndim > 2:
            self.raw_image = self.raw_image[..., 0]

        if len(np.unique(self.raw_image.flatten())) > 2:
            Log.logger.warning("Image to be analyzed has to be binary.")
            return

        if self.dark_line:
            self.raw_image = 255 - self.raw_image
            self.FOREGROUND = self.raw_image.max()
            self.BACKGROUND = self.raw_image.min()

        # Skeletonize
        skeleton = skeletonize(remove_small_holes(self.raw_image == self.FOREGROUND, self.hole_thresh))
        self.skel_image = (skeleton * self.FOREGROUND).astype(np.uint8)
        self.construct_graphs()
        self.draw_key_points()
        self.calc_len_map_long()
        self.calc_len_map_all()
        self.calc_total_len()
        self.calc_proj_area()
        self.calc_growth_unit()
        self.calc_frac_dim()
        self.calc_lacunarity()

        # fig, axes = plt.subplots(3, 2, figsize=(16, 16))
        # axes[0, 0].imshow(self.raw_image, cmap='gray')
        # axes[0, 0].set_title("Ridges")
        # axes[0, 1].imshow(self.skel_image, cmap='gray')
        # axes[0, 1].set_title("Skeletonized")
        # axes[1, 0].imshow(self.pruned_image, cmap='gray')
        # axes[1, 0].set_title("Pruned")
        # axes[1, 1].imshow(self.key_pts_image)
        # axes[1, 1].set_title("Key Points")
        # axes[2, 0].imshow(self.length_map)
        # axes[2, 0].set_title("Longest Paths")
        # axes[2, 0].axis('off')
        # axes[2, 1].imshow(self.curve_map_longest)
        # axes[2, 1].axis('off')
        # plt.show()


if __name__ == "__main__":
    t1 = time.time()
    img_path = "/Users/lxfhfut/Downloads/1681222495317.jpg"
    det = FibreDetector(line_widths=[11],
                        low_contrast=100,
                        high_contrast=200,
                        dark_line=False,
                        extend_line=True,
                        correct_pos=False,
                        min_len=5)
    det.detect_lines(img_path)
    print(time.time() - t1)
    det.save_results(save_dir="/Users/lxfhfut/Desktop/img2/", make_binary=True, draw_junc=True, draw_width=True)
    skel = SkeletonAnalyzer(skel_thresh=10, branch_thresh=5)
    skel.analyze_image("/Users/lxfhfut/Desktop/img2/binary_contours.png")
