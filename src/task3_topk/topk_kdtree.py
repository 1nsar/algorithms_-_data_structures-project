"Task 3 - finding the top k closest pairs of points using a KD-Tree"

import math
import heapq
from typing import List, Tuple, Set, Optional, Dict

from src.task2_3d.kdtree_3d import KDTree3D

Point3D = Tuple[int, float, float, float]
PairOut = Tuple[float, Tuple[Point3D, Point3D]]

_EPS = 1e-12


def _pair_key(a: Point3D, b: Point3D) -> Tuple[int, int]:
    ia, ib = a[0], b[0]
    return (ia, ib) if ia <= ib else (ib, ia)


def find_topk_pairs_baseline(points: List[Point3D], k: int) -> List[PairOut]: # this is a baseline function to find the top k closest pairs of points
    if k <= 0 or len(points) < 2:
        return []

    candidates: List[PairOut] = []
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            dx = pi[1] - pj[1]
            dy = pi[2] - pj[2]
            dz = pi[3] - pj[3]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)

            if pi[0] <= pj[0]:
                candidates.append((d, (pi, pj)))
            else:
                candidates.append((d, (pj, pi)))

    return heapq.nsmallest(k, candidates, key=lambda x: (x[0], x[1][0][0], x[1][1][0]))


def find_topk_pairs_optimized(points: List[Point3D], k: int, neighbor_k: Optional[int] = None) -> List[PairOut]: # this is an optimized function to find the top k closest pairs of points using a KD-Tree
    if k <= 0 or len(points) < 2:
        return []

    if neighbor_k is None:
        neighbor_k = max(k + 1, 32)

    tree = KDTree3D(points)
    id_to_point: Dict[int, Point3D] = {p[0]: p for p in points}

    nk = max(1, neighbor_k)
    while True:
        candidates: List[PairOut] = []
        seen: Set[Tuple[int, int]] = set()
        radii: Dict[int, float] = {}

        for p in points:
            neighs = tree.find_nearest_neighbors(p, k=nk + 1)
            far = 0.0
            cnt = 0
            for dist, q in neighs:
                if q[0] == p[0]:
                    continue
                q_now = id_to_point.get(q[0], q)
                cnt += 1
                if dist > far:
                    far = dist
                key = _pair_key(p, q_now)
                if key in seen:
                    continue
                seen.add(key)
                if p[0] <= q_now[0]:
                    candidates.append((dist, (p, q_now)))
                else:
                    candidates.append((dist, (q_now, p)))
            radii[p[0]] = far if cnt > 0 else float("inf")

        if not candidates:
            if nk >= len(points) - 1:
                return []
            nk = min(len(points) - 1, max(nk + 1, nk * 2))
            continue

        best = heapq.nsmallest(k, candidates, key=lambda x: (x[0], x[1][0][0], x[1][1][0]))
        dk = best[-1][0] if len(best) >= k else float("inf")

        if len(best) < k:
            if nk >= len(points) - 1:
                return best
            nk = min(len(points) - 1, max(nk + 1, nk * 2))
            continue

        certified = True
        for rid in radii.values():
            if rid <= dk + _EPS:
                certified = False
                break

        if certified or nk >= len(points) - 1:
            return best

        nk = min(len(points) - 1, max(nk + 1, nk * 2))


def find_top_k_pairs( # finding the top k closest pairs
    points: List[Point3D],
    k: int,
    *,
    method: str = "auto",
    exact_threshold: int = 2000,
    neighbor_k: Optional[int] = None,
    validate_on_small: bool = False,
) -> List[PairOut]:

    n = len(points)

    if method not in {"auto", "exact", "optimized"}:
        raise ValueError("method must be one of: auto, exact, optimized")

    if method == "auto":
        use_exact = n <= exact_threshold
    else:
        use_exact = (method == "exact")

    if use_exact:
        exact = find_topk_pairs_baseline(points, k)
        if validate_on_small:
            approx = find_topk_pairs_optimized(points, k, neighbor_k=neighbor_k)
            if exact != approx:
                raise AssertionError("Validation failed: optimized output != baseline output for this dataset.")
        return exact

    return find_topk_pairs_optimized(points, k, neighbor_k=neighbor_k)


if __name__ == "__main__":
    pts = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 0.1, 0.1, 0.1), (3, 10, 10, 10)]
    print("Exact:", find_top_k_pairs(pts, 3, method="exact"))
    print("Opt  :", find_top_k_pairs(pts, 3, method="optimized"))
