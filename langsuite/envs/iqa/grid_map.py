# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

from collections import deque
from math import ceil, floor

import numpy as np

from langsuite.envs.iqa.iqa_world import IqaWorld
from langsuite.shapes import Point2D


class GridMap:
    def __init__(self, world: IqaWorld):
        self.grid_size = world.grid_size
        x = []
        z = []

        for i in world.rooms:
            room = world.rooms[i]
            room_x, room_z = room.geometry.shapely_geo.exterior.xy
            x.extend(room_x)
            z.extend(room_z)
        self.max_x = max(x)
        self.min_x = min(x)
        self.max_z = max(z)
        self.min_z = min(z)
        self.grid_map = []
        self.grid_map_size = (
            ceil((self.max_z - self.min_z) / self.grid_size),
            ceil((self.max_x - self.min_x) / self.grid_size),
        )
        for axis_z in range(self.grid_map_size[0]):
            z_cells = []
            for axis_x in range(self.grid_map_size[0]):
                center_z = self.min_z + (axis_z + 0.5) * self.grid_size
                center_x = self.min_x + (axis_x + 0.5) * self.grid_size
                z_cells.append(Cell(axis_z, axis_x, center_z, center_x, 0))
            self.grid_map.append(z_cells)
        self.update_walls(world)
        self.update_objects(world)

    def update_walls(self, world):
        for j in world.walls:
            wall = world.walls[j]
            wall_x, wall_z = wall.geometry.shapely_geo.exterior.xy
            if wall_x[0] == wall_x[1]:  # 垂直线
                wall_min_z = min(wall_z)
                wall_max_z = max(wall_z)

                x_start = floor((wall_x[0] - self.min_x) / self.grid_size)
                z_start = floor((wall_min_z - self.min_z) / self.grid_size)
                z_end = floor((wall_max_z - self.min_z) / self.grid_size)
                for w_z in range(z_start, z_end + 1):
                    self.grid_map[w_z][x_start].occupied_type = 1

            elif wall_z[0] == wall_z[1]:  # 水平线
                wall_min_x = min(wall_x)
                wall_max_x = max(wall_x)
                x_start = floor((wall_min_x - self.min_x) / self.grid_size)
                x_end = floor((wall_max_x - self.min_x) / self.grid_size)
                z_start = floor((wall_z[0] - self.min_z) / self.grid_size)

                for w_x in range(x_start, x_end + 1):
                    self.grid_map[z_start][w_x].occupied_type = 1

    def update_objects(self, world):
        self.object_cells = {}
        # reset object cells
        for axis_z in range(self.grid_map_size[0]):
            for axis_x in range(self.grid_map_size[0]):
                if self.grid_map[axis_z][axis_x].occupied_type == 2:
                    self.grid_map[axis_z][axis_x].occupied_type = 0
        for _, obj in world.objects.items():
            cells = []
            obj_x, obj_z = obj.geometry.shapely_geo.exterior.xy
            obj_min_x = min(obj_x)
            obj_max_x = max(obj_x)
            obj_min_z = min(obj_z)
            obj_max_z = max(obj_z)
            marked_flag = False
            for o_x in np.arange(obj_min_x, obj_max_x, self.grid_size):
                for o_z in np.arange(obj_min_z, obj_max_z, self.grid_size):
                    # 检查点是否在旋转后的矩形内
                    if obj.geometry.intersects(Point2D(o_x, o_z)):
                        grid_x = floor((o_x - self.min_x) / self.grid_size)
                        grid_z = floor((o_z - self.min_z) / self.grid_size)
                        # and grid_map[grid_z, grid_x] == 0
                        if (
                            0 <= grid_x < self.grid_map_size[1]
                            and 0 <= grid_z < self.grid_map_size[0]
                        ):
                            self.grid_map[grid_z][grid_x].occupied_type = 2
                            cells.append(self.grid_map[grid_z][grid_x])
                            marked_flag = True
            # if the object is too small, add all the corner points
            if not marked_flag:
                for o_x, o_z in zip(obj_x, obj_z):
                    if obj.geometry.intersects(Point2D(o_x, o_z)):
                        grid_x = floor((o_x - self.min_x) / self.grid_size)
                        grid_z = floor((o_z - self.min_z) / self.grid_size)
                        # and grid_map[grid_z, grid_x] == 0
                        if (
                            0 <= grid_x < self.grid_map_size[1]
                            and 0 <= grid_z < self.grid_map_size[0]
                        ):
                            self.grid_map[grid_z][grid_x].occupied_type = 2
                            cells.append(self.grid_map[grid_z][grid_x])
                            marked_flag = True
            self.object_cells[obj.id] = cells

    def get_reachable_positions(self, start_cell):
        # rows, cols = len(grid), len(grid[0])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        visited = set()
        reachable_positions = []
        # start_cell = self.grid_map[start_x, start_z]
        queue = deque([start_cell])
        visited.add(start_cell)

        def is_valid_move(z, x):
            return (
                0 <= x < self.grid_size[1]
                and 0 <= z < self.grid_size[0]
                and self.grid_map[z][x].occupied_type == 0
            )

        while queue:
            current_cell = queue.popleft()
            reachable_positions.append(current_cell)
            for dz, dx in directions:
                new_z, new_x = current_cell.grid_z + dz, current_cell.grid_x + dx
                if (
                    is_valid_move(new_z, new_x)
                    and self.grid_map[new_z][new_x] not in visited
                ):
                    queue.append(self.grid_map[new_z][new_x])
                    visited.add(self.grid_map[new_z][new_x])

        return reachable_positions

    def get_interactable_positions(
        self, obj_id, max_operation_distance, reachable_positions
    ):
        # TODO
        cells = self.object_cells[obj_id]
        interactable_positions = []
        for position in reachable_positions:
            pass

        return interactable_positions

    def get_grid_cell(self, real_x, real_z):
        grid_x = round((real_x - self.min_x) / self.grid_size)
        grid_z = round((real_z - self.min_z) / self.grid_size)

        return self.grid_map[grid_z][grid_x]


class Cell:
    def __init__(self, grid_z, grid_x, center_z, center_x, occupied_type):
        self.grid_z = grid_z
        self.grid_x = grid_x
        self.center_z = center_z
        self.center_x = center_x
        self.occupied_type = occupied_type  # {0: empty, 1: wall, 2:object}
