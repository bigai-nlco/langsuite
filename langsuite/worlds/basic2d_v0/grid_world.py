from __future__ import annotations

import heapq
from typing import Dict, Optional, Tuple
import numpy as np


from langsuite.shapes import Point2D, Polygon2D, Vector2D
from langsuite.suit.exceptions import UnexecutableWithSptialError
from langsuite.worlds.basic2d_v0.physical_entity import Object2D


class AstarState:
    def __init__(
        self, x, y, d, last: Optional[AstarState], targets: Dict[Tuple[int, int], int]
    ) -> None:
        self.x = x
        self.y = y
        self.d = d
        self.last = last
        self.h = min(abs(x_t - x) + abs(y_t - y) for (x_t, y_t) in targets)
        self.g = last.g + 1 if last is not None else 0
        self.f = self.g + self.h

    def __lt__(self, other: Optional["AstarState"]):
        return (other is None) or self.f < other.f


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

        self.grids[x_min_idx : x_max_idx + 1, y_min_idx : y_max_idx + 1] = True

    def _is_valid_pos(self, x, y):
        return (0 <= x < self.x_len) and (0 <= y < self.y_len)

    def _get_candidate_ends(self, x: int, y: int):
        for i in range(4):
            j = 0
            x_c, y_c = x, y
            while self._is_valid_pos(x_c, y_c) and j < self.step_limit:
                if not self.grids[x_c][y_c]:
                    yield ((x_c, y_c), (i + 2) % 4)
                x_c += self.D_X[i]
                y_c += self.D_Y[i]
                j += 1

    def _get_next_states(self, st: AstarState, targets: Dict[Tuple[int, int], int]):
        for i in range(4):
            n_x = st.x + self.D_X[i]
            n_y = st.y + self.D_Y[i]
            if self._is_valid_pos(n_x, n_y):
                yield (AstarState(n_x, n_y, i, st, targets))

    def _differentiate_path(self, st: AstarState, direction: int) -> list[str]:
        def add_turn(d) -> list[str]:
            if d == 1:
                return ["turn_left"]
            elif d == 2:
                return ["turn_left", "turn_left"]
            elif d == 3:
                return ["turn_right"]
            else:
                raise NotImplementedError("no such direction!")

        paths = []
        if st.d != direction:
            paths.extend(add_turn(abs(direction - st.d)))
        while st.last is not None:
            for i in range(4):
                l_x = st.x - self.D_X[i]
                l_y = st.y - self.D_Y[i]
                if (l_x, l_y) == (st.last.x, st.last.y):
                    if i != st.d:
                        paths.extend(add_turn(abs(st.d - i)))
                    paths.append("move_ahead")
            st = st.last
        paths = reversed(paths)
        return list(paths)

    def _get_direction_id(self, vec_dir: Vector2D):
        for i in range(4):
            if (
                abs(vec_dir.x - self.D_X[i]) < 0.1
                and abs(vec_dir.y - self.D_Y[i]) < 0.1
            ):
                return i
        raise NotImplementedError("Only support UDLR directions.")

    def get_path(self, start_pos: Point2D, start_dir: Vector2D, target_pos: Point2D):
        x1, y1 = int((start_pos.x - self.x_min) / self.grid_step_length), int(
            (start_pos.y - self.y_min) / self.grid_step_length
        )
        x2, y2 = int((target_pos.x - self.x_min) / self.grid_step_length), int(
            (target_pos.y - self.y_min) / self.grid_step_length
        )
        self.grids[x2][y2] = True
        ends = {p: d for p, d in self._get_candidate_ends(x2, y2)}
        queue = []
        heapq.heappush(
            queue, AstarState(x1, y1, self._get_direction_id(start_dir), None, ends)
        )
        visited = np.zeros((self.x_len, self.y_len), dtype=bool)
        visited[x2][y2] = True
        ans = None
        while len(ends) > 0 and len(queue) > 0:
            u: AstarState = heapq.heappop(queue)
            if not u < ans:
                break
            visited[u.x][u.y] = True
            if (u.x, u.y) in ends:
                ans = u
                continue
            for v in self._get_next_states(u, ends):
                if not visited[v.x][v.y]:
                    heapq.heappush(queue, v)
        if ans is None:
            raise UnexecutableWithSptialError({"path": "no avail path"})
        
        end_d = ends[(ans.x, ans.y)]
        path = self._differentiate_path(ans, end_d)

        position = Point2D(
            start_pos.x + (ans.x - x1) * self.grid_step_length,
            start_pos.y + (ans.y - y1) * self.grid_step_length,
        )
        direction = Vector2D(self.D_X[end_d], self.D_Y[end_d])
        
        return position, direction, path
