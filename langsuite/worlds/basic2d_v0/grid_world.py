from __future__ import annotations

from ast import Break
from dataclasses import dataclass
import enum
import heapq
import math
from collections import deque
from optparse import Option
from typing import Optional, Set, Tuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from langsuite.shapes import Point2D, Polygon2D
from langsuite.worlds.basic2d_v0.physical_entity import Object2D


class AstarState():
    def __init__(self, x, y, last: Optional[AstarState], targets: Set[Tuple[int, int]]) -> None:
        self.x = x
        self.y = y
        self.h = min(abs(x_t - x) + abs(y_t - y) for x_t, y_t in targets)
        self.g = last.g + 1 if last is not None else 0
        self.f = self.g + self.h

    def __le__(self, other: Optional['AstarState']):
        return (other is None) or self.f <= other.f

# TODO only support single room now.
class GridWorld:
    D_X = [-1, 0, 1, 0]
    D_Y = [0, -1, 0, 1]
    D_N = ["L", "U", "R", "D"]

    def __init__(self, poly_2d: Polygon2D, grid_step_length, max_view_step):
        self.x_min = poly_2d.x_min
        self.x_max = poly_2d.x_max
        self.y_min = poly_2d.y_min
        self.y_max = poly_2d.y_max
        self.grid_step_length = grid_step_length
        self.x_len = int((self.x_max - self.x_min - 1e-5) / grid_step_length) + 1
        self.y_len = int((self.y_max - self.y_min - 1e-5) / grid_step_length) + 1
        self.step_limit = max_view_step
        self.grids = np.zeros((self.x_len, self.y_len), dtype=bool)

    def add_object(self, obj: Object2D):
        x_min = obj.geometry.x_min
        x_max = obj.geometry.x_max
        y_min = obj.geometry.y_min
        y_max = obj.geometry.y_max

        x_min_idx = int((x_min - self.x_min) / self.grid_step_length)
        x_max_idx = int((x_max - self.x_min - 1e-5) / self.grid_step_length) + 1
        y_min_idx = int((y_min - self.y_min) / self.grid_step_length)
        y_max_idx = int((y_max - self.y_min - 1e-5) / self.grid_step_length) + 1

        self.grids[x_min_idx:x_max_idx+1, y_min_idx:y_max_idx+1] = True

    def is_valid_pos(self, x, y):
        return (0 <= x < self.x_len) and (0 <= y < self.y_len)

    def get_candidate_ends(self, x: int, y: int):
        for i in range(4):
            j = 0
            x_c, y_c = x, y
            while self.is_valid_pos(x_c, y_c)  and j < self.step_limit:
                x_c += self.D_X[i]
                y_c += self.D_Y[i]
                j += 1
                if not self.grids[x_c][y_c]:
                    yield (x_c, y_c)

  
    def get_next_states(self, st: AstarState, targets: Set[Tuple[int, int]]):
        for i in range(4):
            n_x = st.x + self.D_X[i]
            n_y = st.y + self.D_Y[i]
            if self.is_valid_pos(n_x, n_y):
                yield(AstarState(n_x, n_y, st, targets))

    def differentiate_path(self, st: AstarState):
        for i in range(4):
            

    def get_path(self, start_pos: Point2D, target_pos: Point2D, direction, grid):
        x1, y1 = int((start_pos.x - self.x_min) / self.grid_step_length), int(
            (start_pos.y - self.y_min) / self.grid_step_length
        )
        x2, y2 = int((target_pos.x - self.x_min) / self.grid_step_length), int(
            (target_pos.y - self.y_min) / self.grid_step_length
        )
        self.grids[x2][y2] = True
        ends = set(self.get_candidate_ends(x2, y2))
        queue = []
        heapq.heappush(queue, AstarState(x1, y1, None, ends))
        visited = np.zeros((self.x_len, self.y_len), dtype=bool)
        ans = None
        while len(ends) > 0:
            u: AstarState = heapq.heappop(queue)
            if u <= ans:
                break
            if self.grids[u.x][u.y]:
                ans = u
                continue
            visited[u.x][u.y] = True
            if (u.x, u.y) in ends:
                ans = u
            for v in self.get_next_states(u, ends):
                if not visited[v.x][v.y]:
                    heapq.heappush(queue, v)

        path = self.differentiate_path(ans)



def point_traj_2action_traj(points, direction, grid_step_length=1):
    # x1, y1 = points[0]
    y1, x1 = points[0]
    path = []
    for point in points[1:]:
        # x2 ,y2 = point
        y2, x2 = point
        dx = x2 - x1
        dy = y2 - y1
        rotate_flag = True
        while rotate_flag:
            if direction == "up":
                if dy > 0:
                    path.append("MoveAhead")
                    y1 += grid_step_length
                    dy -= grid_step_length
                    rotate_flag = False
                elif dy < 0:
                    path.append("TurnLeft")
                    path.append("TurnLeft")
                    direction = "down"
                elif dx > 0:
                    path.append("TurnRight")
                    direction = "right"
                elif dx < 0:
                    path.append("TurnLeft")
                    direction = "left"
            elif direction == "down":
                if dy < 0:
                    path.append("MoveAhead")
                    y1 -= grid_step_length
                    dy += grid_step_length
                    rotate_flag = False
                elif dy > 0:
                    path.append("TurnLeft")
                    path.append("TurnLeft")
                    direction = "up"
                elif dx > 0:
                    path.append("TurnLeft")
                    direction = "right"
                elif dx < 0:
                    path.append("TurnRight")
                    direction = "left"
            elif direction == "left":
                if dx < 0:
                    path.append("MoveAhead")
                    x1 -= grid_step_length
                    dx += grid_step_length
                    rotate_flag = False
                elif dx > 0:
                    path.append("TurnLeft")
                    path.append("TurnLeft")
                    direction = "right"
                elif dy > 0:
                    path.append("TurnRight")
                    direction = "up"
                elif dy < 0:
                    path.append("TurnLeft")
                    direction = "down"
            elif direction == "right":
                if dx > 0:
                    path.append("MoveAhead")
                    x1 += grid_step_length
                    dx -= grid_step_length
                    rotate_flag = False
                elif dx < 0:
                    path.append("TurnLeft")
                    path.append("TurnLeft")
                    direction = "left"

                elif dy > 0:
                    path.append("TurnLeft")
                    direction = "up"
                elif dy < 0:
                    path.append("TurnRight")
                    direction = "down"
    return path


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal):
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    return math.sqrt(dx**2 + dy**2)


def get_neighbours(node, grid):
    neighbours = []
    # Directions: up, down, right, left
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dir in dirs:
        neighbour_pos = (node.position[0] + dir[0], node.position[1] + dir[1])
        if (
            neighbour_pos[0] < 0
            or neighbour_pos[0] >= len(grid)
            or neighbour_pos[1] < 0
            or neighbour_pos[1] >= len(grid[0])
        ):
            continue
        # obstacles
        if grid[neighbour_pos[0]][neighbour_pos[1]] == 1:
            continue
        neighbour_node = Node(neighbour_pos, node)
        neighbours.append(neighbour_node)
    return neighbours




def cal_wall_min_max(wall_list):
    x_min, x_max, y_min, y_max = (
        float("inf"),
        -float("inf"),
        float("inf"),
        -float("inf"),
    )
    for x, y in wall_list:
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    return x_min, x_max, y_min, y_max


def get_direction(view_vector):
    x, y = round(view_vector.x), round(view_vector.y)
    direction = None
    if x > 0:
        direction = "right"
    elif x < 0:
        direction = "left"
    elif y > 0:
        direction = "up"
    elif y < 0:
        direction = "down"
    return direction


def get_relative_direction(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) > abs(dy):
        if dx < 0:
            return "right"
        else:
            return "left"
    else:
        if dy > 0:
            return "down"
        else:
            return "up"
