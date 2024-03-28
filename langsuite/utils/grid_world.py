from __future__ import annotations

import heapq
import math
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class GridWorld:
    def __init__(self, x_min, x_max, y_min, y_max, grid_step_length):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.grid_step_length = grid_step_length
        self.grid_width = int((x_max - x_min) / grid_step_length) + 1
        self.grid_height = int((y_max - y_min) / grid_step_length) + 1
        self.grid = [[0] * self.grid_width for _ in range(self.grid_height)]
        self.objects = []

    def add_object(self, obj_x, obj_y, corners):
        x_min = min(corners[0])
        x_max = max(corners[0])
        y_min = min(corners[1])
        y_max = max(corners[1])

        x_min_idx = int((x_min - self.x_min) / self.grid_step_length)
        x_max_idx = int((x_max - self.x_min) / self.grid_step_length)
        y_min_idx = int((y_min - self.y_min) / self.grid_step_length)
        y_max_idx = int((y_max - self.y_min) / self.grid_step_length)
        for y in range(y_min_idx, y_max_idx + 1):
            for x in range(x_min_idx, x_max_idx + 1):
                self.grid[y][x] = 1
        self.objects.append((obj_x, obj_y))

    def get_path(self, start_point, target_position, direction, grid):
        target_x, target_y = target_position
        x2, y2 = int((target_x - self.x_min) / self.grid_step_length), int(
            (target_y - self.y_min) / self.grid_step_length
        )
        obj_cordinate = x2, y2

        x1, y1 = start_point.x, start_point.y
        x1, y1 = int((x1 - self.x_min) / self.grid_step_length), int(
            (y1 - self.y_min) / self.grid_step_length
        )
        start = (y1, x1)
        self.grid[y2][x2] = 3
        self.grid[y1][x1] = 4
        candidate_end = bfs(grid, y2, x2)
        grid_trajectory = None
        while candidate_end:
            end = candidate_end.pop()

            # for x, y in (start, end):
            #     self.grid[x][y] = 2
            # self.render()
            grid_trajectory = a_star_search(grid, start, end)
            if grid_trajectory:
                break
        # print(grid_trajectory)
        # for x, y in grid_trajectory:
        #     self.grid[x][y] = 2
        # self.render()

        if grid_trajectory is None:
            return None, None, None
        if len(grid_trajectory) == 1:
            return False, None, None
        cordinate_diff = (
            grid_trajectory[-1][0] - grid_trajectory[-2][0],
            grid_trajectory[-1][1] - grid_trajectory[-2][1],
        )
        last_trajectory_direction = ""
        if cordinate_diff[1] > 0:
            last_trajectory_direction = "right"
        elif cordinate_diff[1] < 0:
            last_trajectory_direction = "left"
        elif cordinate_diff[0] > 0:
            last_trajectory_direction = "up"
        elif cordinate_diff[0] < 0:
            last_trajectory_direction = "down"
        y2, x2 = end
        last_direction = get_relative_direction(
            obj_cordinate[0], obj_cordinate[1], x2, y2
        )
        extra_rotate_list = generate_rotation_actions(
            last_trajectory_direction, last_direction
        )
        opetation_rotate_list = generate_rotation_actions(direction, last_direction)

        action_trajectory = point_traj_2action_traj(grid_trajectory, direction)
        action_trajectory.extend(extra_rotate_list)

        end_coordinate = (
            x2 * self.grid_step_length + self.x_min,
            y2 * self.grid_step_length + self.y_min,
        )
        return end_coordinate, action_trajectory, opetation_rotate_list

    def render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal")
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y][x] == 1:
                    rect = Rectangle(
                        (
                            self.x_min + x * self.grid_step_length,
                            self.y_min + y * self.grid_step_length,
                        ),
                        self.grid_step_length,
                        self.grid_step_length,
                        facecolor="black",
                    )
                    ax.add_patch(rect)
                elif self.grid[y][x] == 0:
                    rect = Rectangle(
                        (
                            self.x_min + x * self.grid_step_length,
                            self.y_min + y * self.grid_step_length,
                        ),
                        self.grid_step_length,
                        self.grid_step_length,
                        facecolor="white",
                        edgecolor="black",
                    )
                    ax.add_patch(rect)
                elif self.grid[y][x] == 3:
                    rect = Rectangle(
                        (
                            self.x_min + x * self.grid_step_length,
                            self.y_min + y * self.grid_step_length,
                        ),
                        self.grid_step_length,
                        self.grid_step_length,
                        facecolor="red",
                        edgecolor="black",
                    )
                    ax.add_patch(rect)
                elif self.grid[y][x] == 4:
                    rect = Rectangle(
                        (
                            self.x_min + x * self.grid_step_length,
                            self.y_min + y * self.grid_step_length,
                        ),
                        self.grid_step_length,
                        self.grid_step_length,
                        facecolor="green",
                        edgecolor="black",
                    )
                    ax.add_patch(rect)
                else:
                    rect = Rectangle(
                        (
                            self.x_min + x * self.grid_step_length,
                            self.y_min + y * self.grid_step_length,
                        ),
                        self.grid_step_length,
                        self.grid_step_length,
                        facecolor="yellow",
                        edgecolor="black",
                    )
                    ax.add_patch(rect)
        plt.show()


def bfs(grid, x, y):
    n = len(grid)
    m = len(grid[0])

    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    res_lis = []

    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    while queue:
        curr_x, curr_y = queue.popleft()
        for di in directions:
            new_x = curr_x + di[0]
            new_y = curr_y + di[1]
            try:
                if (
                    grid[new_x][new_y] == 0
                    and (new_x != x and new_y != y)
                    and len(res_lis) < 10
                ):
                    res_lis.append((new_x, new_y))

                if 0 <= new_x < n and 0 <= new_y < m and (new_x, new_y) not in visited:
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))

                if len(res_lis) > 10:
                    break
            except:
                continue
    return res_lis


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


def a_star_search(grid, start, end):
    open_list = []
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(end)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position in closed_set:
            continue
        closed_set.add(current_node.position)
        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path

        neighbours = get_neighbours(current_node, grid)
        for neighbour in neighbours:
            if neighbour.position in closed_set:
                continue
            neighbour.g = current_node.g + 1
            neighbour.h = heuristic(neighbour, goal_node)
            neighbour.f = neighbour.g + neighbour.h

            heapq.heappush(open_list, neighbour)

    return None


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


def generate_rotation_actions(initial_direction, target_direction):
    directions = ["up", "right", "down", "left"]
    actions = []

    initial_index = directions.index(initial_direction)
    target_index = directions.index(target_direction)

    rotations = target_index - initial_index
    if rotations == 1 or rotations == -3:
        actions.append("TurnRight")
    elif rotations == 3 or rotations == -1:
        actions.append("TurnLeft")
    elif rotations == 2 or rotations == -2:
        actions.extend(["TurnRight", "TurnRight"])

    return actions
