"""Closest pair of 2D drone points using divide and conquer."""

import math
from typing import List, Tuple

Point2D = Tuple[int, float, float]

def compute_distance_2d(point_a: Point2D, point_b: Point2D) -> float:
    return math.hypot(point_a[1] - point_b[1], point_a[2] - point_b[2])

def closest_pair_bruteforce_2d(points: List[Point2D]) -> Tuple[Tuple[Point2D, Point2D], float]:
    best_pair = (None, None)
    best_distance = float("inf")
    point_count = len(points)
    for left_index in range(point_count):
        for right_index in range(left_index + 1, point_count):
            distance = compute_distance_2d(points[left_index], points[right_index])
            candidate_ids = tuple(sorted((points[left_index][0], points[right_index][0])))
            current_ids = (
                tuple(sorted((best_pair[0][0], best_pair[1][0])))
                if best_pair[0]
                else (float("inf"), float("inf"))
            )
            if distance < best_distance - 1e-12 or (
                abs(distance - best_distance) <= 1e-12 and candidate_ids < current_ids
            ):
                best_pair = (points[left_index], points[right_index])
                best_distance = distance
    return best_pair, best_distance

def closest_pair_in_strip_2d(strip_points: List[Point2D], current_best_distance: float):
    best_pair = (None, None)
    best_distance = current_best_distance
    strip_count = len(strip_points)
    for first_index in range(strip_count):
        second_index = first_index + 1

        while (
            second_index < strip_count
            and (strip_points[second_index][2] - strip_points[first_index][2]) < best_distance
        ):
            candidate_distance = compute_distance_2d(strip_points[first_index], strip_points[second_index])
            candidate_ids = tuple(
                sorted((strip_points[first_index][0], strip_points[second_index][0]))
            )
            current_ids = (
                tuple(sorted((best_pair[0][0], best_pair[1][0])))
                if best_pair[0]
                else (float("inf"), float("inf"))
            )
            if candidate_distance < best_distance - 1e-12 or (
                abs(candidate_distance - best_distance) <= 1e-12 and candidate_ids < current_ids
            ):
                best_pair = (strip_points[first_index], strip_points[second_index])
                best_distance = candidate_distance
            second_index += 1
    return best_pair, best_distance

def closest_pair_recursive_2d(points_by_x: List[Point2D], points_by_y: List[Point2D]):
    point_count = len(points_by_x)
    if point_count <= 3:
        return closest_pair_bruteforce_2d(points_by_x)
    mid_index = point_count // 2
    left_sorted_x = points_by_x[:mid_index]
    right_sorted_x = points_by_x[mid_index:]
    split_x = points_by_x[mid_index][1]
    left_sorted_y = []
    right_sorted_y = []
    for point in points_by_y:
        if point[1] <= split_x:
            left_sorted_y.append(point)
        else:
            right_sorted_y.append(point)
    (left_a, left_b), left_distance = closest_pair_recursive_2d(left_sorted_x, left_sorted_y)
    (right_a, right_b), right_distance = closest_pair_recursive_2d(right_sorted_x, right_sorted_y)
    if left_distance < right_distance - 1e-12 or (
        abs(left_distance - right_distance) <= 1e-12
        and tuple(sorted((left_a[0], left_b[0]))) < tuple(sorted((right_a[0], right_b[0])))
    ):
        best_pair, best_distance = (left_a, left_b), left_distance
    else:
        best_pair, best_distance = (right_a, right_b), right_distance
    strip_points = [point for point in points_by_y if abs(point[1] - split_x) < best_distance]
    strip_pair, strip_distance = closest_pair_in_strip_2d(strip_points, best_distance)
    if strip_pair[0] is not None and (
        strip_distance < best_distance - 1e-12
        or (
            abs(strip_distance - best_distance) <= 1e-12
            and tuple(sorted((strip_pair[0][0], strip_pair[1][0])))
            < tuple(sorted((best_pair[0][0], best_pair[1][0])))
        )
    ):
        return strip_pair, strip_distance
    return best_pair, best_distance

def find_closest_pair_2d(drone_points: List[Point2D]):
    if len(drone_points) < 2:
        return None, float("inf")
    points_by_x = sorted(drone_points, key=lambda point: (point[1], point[2], point[0]))
    points_by_y = sorted(drone_points, key=lambda point: (point[2], point[1], point[0]))
    best_pair, best_distance = closest_pair_recursive_2d(points_by_x, points_by_y)

    if best_pair[0][0] <= best_pair[1][0]:
        return best_pair, best_distance
    else:
        return (best_pair[1], best_pair[0]), best_distance


if __name__ == "__main__":
    sample_points = [(0, 0.0, 0.0), (1, 1.0, 1.0), (2, 2.0, 2.0), (3, 0.1, 0.1)]
    print(find_closest_pair_2d(sample_points))
