# Generations of drones for both 2D and 3D

import random

def generate_drone_points_2d(point_count, bound=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    return [(index, random.random() * bound, random.random() * bound) for index in range(point_count)]

def generate_drone_points_3d(point_count, bound=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    return [
        (index, random.random() * bound, random.random() * bound, random.random() * bound)
        for index in range(point_count)
    ]

