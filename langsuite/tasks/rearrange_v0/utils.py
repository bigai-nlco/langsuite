from typing import Tuple
from langsuite.shapes import Point2D
from langsuite.worlds.basic2d_v0.world import Basic2DWorld_V0


def compute_square_hack(world: Basic2DWorld_V0, point: Point2D) -> Tuple[int, int]:
    # HACK only work for single room
    it = iter(world.rooms.values())
    next(it) # drop floor
    room = next(it)
    width_grid_size = (room.geometry.x_max - room.geometry.x_min) / 3
    length_grid_size = (room.geometry.y_max - room.geometry.y_min) / 3
    x_pos = int((point.x - room.geometry.x_min) / width_grid_size)
    y_pos = int((point.y - room.geometry.y_min) / length_grid_size)
    return x_pos, y_pos
