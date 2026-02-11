"""KD-tree for exact nearest-neighbor search in 3D."""

import math
from typing import List, Tuple, Optional

Point3D = Tuple[int, float, float, float]

class KDTreeNode3D:
    __slots__ = ("point", "left", "right", "axis")

    def __init__(self, point: Point3D, axis: int):
        self.point = point
        self.left = None
        self.right = None
        self.axis = axis

class KDTree3D:
    def __init__(self, drone_points: List[Point3D]):
        self.root = self.build_tree(drone_points, depth=0)

    def build_tree(self, point_list: List[Point3D], depth: int) -> Optional[KDTreeNode3D]:
        if not point_list:
            return None
        axis = depth % 3
        point_list.sort(key=lambda point: point[axis + 1])
        median_index = len(point_list) // 2
        node = KDTreeNode3D(point_list[median_index], axis)
        node.left = self.build_tree(point_list[:median_index], depth + 1)
        node.right = self.build_tree(point_list[median_index + 1 :], depth + 1)
        return node

    def compute_squared_distance(self, point_a: Point3D, point_b: Point3D) -> float:
        return (
            (point_a[1] - point_b[1]) ** 2
            + (point_a[2] - point_b[2]) ** 2
            + (point_a[3] - point_b[3]) ** 2
        )

    def find_nearest_neighbors(self, query_point: Point3D, k: int = 1) -> List[Tuple[float, Point3D]]:
        best_neighbors = []
        import heapq

        def search(node: KDTreeNode3D):
            if node is None:
                return
            split_point = node.point
            if split_point[0] != query_point[0]:
                distance_sq = self.compute_squared_distance(query_point, split_point)
                if len(best_neighbors) < k:
                    heapq.heappush(best_neighbors, (-distance_sq, split_point))
                else:
                    current_worst = -best_neighbors[0][0]
                    current_pair_ids = tuple(sorted((best_neighbors[0][1][0], query_point[0])))
                    candidate_pair_ids = tuple(sorted((split_point[0], query_point[0])))
                    if distance_sq < current_worst or (
                        distance_sq == current_worst and candidate_pair_ids < current_pair_ids
                    ):
                        heapq.heapreplace(best_neighbors, (-distance_sq, split_point))
            axis = node.axis
            query_axis_value = query_point[axis + 1]
            node_axis_value = split_point[axis + 1]
            visit_left_first = query_axis_value <= node_axis_value
            near_branch = node.left if visit_left_first else node.right
            far_branch = node.right if visit_left_first else node.left
            search(near_branch)

            if len(best_neighbors) < k or (query_axis_value - node_axis_value) ** 2 <= -best_neighbors[0][0]:
                search(far_branch)

        search(self.root)
        return [
            (math.sqrt(-neg_distance), neighbor_point)
            for (neg_distance, neighbor_point) in sorted(best_neighbors, reverse=True)
        ]


def find_closest_pair_3d(drone_points: List[Point3D]):
    if len(drone_points) < 2:
        return None, float("inf")
    kd_tree = KDTree3D(drone_points)
    best_pair = None
    best_distance = float("inf")
    for point in drone_points:
        nearest_neighbors = kd_tree.nearest(point, k=2)
        for distance, neighbor in nearest_neighbors:
            if neighbor[0] == point[0]:
                continue
            if distance < best_distance - 1e-12 or (
                abs(distance - best_distance) <= 1e-12
                and tuple(sorted((point[0], neighbor[0])))
                < tuple(sorted((best_pair[0][0], best_pair[1][0])))
                if best_pair
                else True
            ):
                best_pair = (point, neighbor)
                best_distance = distance
    if best_pair[0][0] <= best_pair[1][0]:
        return best_pair, best_distance
    else:
        return (best_pair[1], best_pair[0]), best_distance


if __name__ == "__main__":
    sample_points = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 0.1, 0.1, 0.1)]
    print(find_closest_pair_3d(sample_points))
