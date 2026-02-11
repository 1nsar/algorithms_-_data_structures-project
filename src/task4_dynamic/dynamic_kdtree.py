"Dynamic structure for non static drones" 

from typing import List, Tuple, Dict, Optional, Set
import random
import heapq
import math

from src.task2_3d.kdtree_3d import KDTree3D
from src.task3_topk.topk_kdtree import find_top_k_pairs

Point3D = Tuple[int, float, float, float]


class DynamicDrones3D: 
    def __init__(self, points: List[Point3D], rebuild_threshold: int = 50):
        self.points: List[Point3D] = list(points)
        self.rebuild_threshold = rebuild_threshold

        self._index: Dict[int, int] = {p[0]: i for i, p in enumerate(self.points)}

        self._dirty = 0
        self._dirty_ids: Set[int] = set()
        self._tree: Optional[KDTree3D] = KDTree3D(self.points)
        self._cache_k: int = 0
        self._cached_topk = []

    def update_drone_point(self, drone_id: int, new_coords: Tuple[float, float, float]): # update the coordinates of a point with a given id
        if drone_id not in self._index:
            return
        idx = self._index[drone_id]
        self.points[idx] = (drone_id, float(new_coords[0]), float(new_coords[1]), float(new_coords[2]))
        self._dirty += 1
        self._dirty_ids.add(drone_id)

        if self._dirty >= self.rebuild_threshold:
            self.rebuild_index()

    def rebuild_index(self):
        self._tree = KDTree3D(self.points)
        self._dirty = 0
        self._dirty_ids.clear()
        self._cache_k = 0
        self._cached_topk = []

    def _ensure_fresh_tree_for_query(self):
        if self._tree is None:
            self.rebuild_index()
        if self._dirty >= self.rebuild_threshold:
            self.rebuild_index()

    def batch_random_walk(self, fraction: float = 0.01, step: float = 1.0, seed=None): # simulation of movement of drones 
        if seed is not None:
            random.seed(seed)

        n = len(self.points)
        if n == 0:
            return
        m = max(1, int(n * fraction))
        selected_ids = random.sample(list(self._index.keys()), m)

        for drone_id in selected_ids:
            idx = self._index[drone_id]
            point = self.points[idx]
            new_coordinates = [coordinate + random.uniform(-step, step) for coordinate in point[1:]]
            self.update_drone_point(drone_id, (new_coordinates[0], new_coordinates[1], new_coordinates[2]))

    def _squared_distance(self, point_a: Point3D, point_b: Point3D) -> float:
        dx = point_a[1] - point_b[1]
        dy = point_a[2] - point_b[2]
        dz = point_a[3] - point_b[3]
        return dx * dx + dy * dy + dz * dz

    def _pair_key(self, point_a: Point3D, point_b: Point3D) -> Tuple[int, int]:
        ia, ib = point_a[0], point_b[0]
        return (ia, ib) if ia <= ib else (ib, ia)

    def _ensure_topk_cache(self, k: int):
        if k <= self._cache_k and self._cached_topk:
            return
        self._ensure_fresh_tree_for_query()
        if not self.points or len(self.points) < 2:
            self._cache_k = k
            self._cached_topk = []
            return
        if self._tree is None:
            self._tree = KDTree3D(self.points)

        neighbor_k = max(k + 1, 32)

        seen_pairs = set()
        candidate_pairs = []

        for point in self.points:
            neighbors = self._tree.find_nearest_neighbors(point, k=neighbor_k + 1)
            for distance, neighbor in neighbors:
                if neighbor[0] == point[0]:
                    continue
                pair_key = self._pair_key(point, neighbor)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                if point[0] <= neighbor[0]:
                    candidate_pairs.append((distance, (point, neighbor)))
                else:
                    candidate_pairs.append((distance, (neighbor, point)))

        self._cached_topk = heapq.nsmallest(
            k,
            candidate_pairs,
            key=lambda x: (x[0], x[1][0][0], x[1][1][0]),
        )
        self._cache_k = k

    def current_topk(self, k: int): # current top k closest pairs of points
        self._ensure_fresh_tree_for_query()
        if k <= 0 or len(self.points) < 2:
            return []

        if self._dirty == 0:
            self._ensure_topk_cache(k)
            return self._cached_topk[:k]

        id_to_point = {p[0]: p for p in self.points}

        self._ensure_topk_cache(max(k, self._cache_k))

        base = []
        for d, (a, b) in self._cached_topk:
            if a[0] in self._dirty_ids or b[0] in self._dirty_ids:
                continue
            aa = id_to_point[a[0]]
            bb = id_to_point[b[0]]
            dist = math.sqrt(self._squared_distance(aa, bb))
            base.append((dist, (aa, bb)))

        neighbor_k = max(k + 1, 32)
        seen = {self._pair_key(pair[0], pair[1]) for _, pair in base}

        dirty_points = [id_to_point[i] for i in self._dirty_ids if i in id_to_point]

        for i in range(len(dirty_points)):
            for j in range(i + 1, len(dirty_points)):
                a = dirty_points[i]
                b = dirty_points[j]
                key = self._pair_key(a, b)
                if key in seen:
                    continue
                seen.add(key)
                dist = math.sqrt(self._squared_distance(a, b))
                if a[0] <= b[0]:
                    base.append((dist, (a, b)))
                else:
                    base.append((dist, (b, a)))

        if self._tree is None:
            self._tree = KDTree3D(self.points)

        for p in dirty_points:
            neighs = self._tree.find_nearest_neighbors(p, k=neighbor_k + 1)
            for _, q_old in neighs:
                if q_old[0] == p[0]:
                    continue
                q = id_to_point.get(q_old[0], q_old)
                key = self._pair_key(p, q)
                if key in seen:
                    continue
                seen.add(key)
                dist = math.sqrt(self._squared_distance(p, q))
                if p[0] <= q[0]:
                    base.append((dist, (p, q)))
                else:
                    base.append((dist, (q, p)))

        return heapq.nsmallest(
            k,
            base,
            key=lambda x: (x[0], x[1][0][0], x[1][1][0]),
        )

    def current_closest(self): # current closest pair of points
        res = self.current_topk(1)
        if not res:
            return None, float("inf")
        d, pair = res[0]
        return pair, d

if __name__ == "__main__":
    pts = [(i, float(i), float(i), float(i)) for i in range(100)]
    dyn = DynamicDrones3D(pts, rebuild_threshold=10)
    dyn.update_drone_point(3, (0.05, 0.05, 0.05))
    print(dyn.current_closest())
    print(dyn.current_topk(5))
