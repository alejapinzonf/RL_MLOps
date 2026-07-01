from dataclasses import dataclass
import random

import numpy as np


@dataclass
class PaperStepResult:
    next_state: tuple[int, int, int, int]
    reward: float
    done: bool
    info: dict


class PaperGridWorldEnv:
    N_ACTIONS = 5
    STAY_ACTION = 4

    def __init__(
        self,
        grid_size: int = 20,
        scenario: str = "wall",
        obstacle_length: int = 4,
        max_steps: int = 300,
        min_goal_start_distance: int = 2,
        seed: int | None = None,
        reward_params: dict | None = None,
    ):
        if scenario not in ("wall", "l_shape", "u_shape"):
            raise ValueError(
                f"Escenario no soportado: {scenario}. Usa: wall, l_shape o u_shape."
            )

        self.grid_size = grid_size
        self.scenario = scenario
        self.obstacle_length = obstacle_length
        self.max_steps = max_steps
        self.min_goal_start_distance = min_goal_start_distance

        self.random = random.Random(seed)

        # Reward params: defaults tomados directamente de Code.py / ql_wall.py
        defaults = {
            "arrival_reward": 150.0,
            "goal_stay_reward": 50.0,
            "goal_stay_out_penalty": 100.0,
            "step_penalty": 1.0,
            "stay_outside_penalty": 60.0,
            "obstacle_hit_penalty": 100.0,
            "goal_position_scale": 2.0,
            "obstacle_position_scale": 2.0,
            "distance_scale_constant": 29.0,  # > distancia euclidiana máxima en 20x20
            "arrival_bonus_multiplier": 100.0,  # multiplicador usado en Code.py: (100*arrival)
        }
        self.reward_params = {**defaults, **(reward_params or {})}

        self.start_pos = None
        self.goal_pos = None
        self.obstacle_cells: tuple[tuple[int, int], ...] = ()

        self.agent_pos = None
        self.current_step = 0
        self._reached_goal_once = False

    @property
    def n_actions(self) -> int:
        return self.N_ACTIONS

    @staticmethod
    def _euclidean(dx: float, dy: float) -> float:
        return float(np.sqrt(dx**2 + dy**2))

    def _nearest_obstacle_cell(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        return min(
            self.obstacle_cells,
            key=lambda cell: (cell[0] - x) ** 2 + (cell[1] - y) ** 2,
        )

    def _build_obstacle_candidates(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[tuple[int, int], ...]]:

        sx, sy = start
        gx, gy = goal
        abx = gx - sx
        aby = gy - sy
        ab_len_sq = abx**2 + aby**2

        if ab_len_sq == 0:
            return []

        n = self.grid_size
        length = self.obstacle_length
        distance_thresholds = [1.5, 2.5, 3.5, 5.0]

        for threshold in distance_thresholds:
            candidates: list[tuple[tuple[int, int], ...]] = []

            for cx in range(n):
                for cy in range(n):
                    for cells in self._candidate_shapes_at(cx, cy, length):
                        if any(
                            x < 0 or x >= n or y < 0 or y >= n for x, y in cells
                        ):
                            continue

                        if start in cells or goal in cells:
                            continue

                        center_x = float(np.mean([c[0] for c in cells]))
                        center_y = float(np.mean([c[1] for c in cells]))

                        apx = center_x - sx
                        apy = center_y - sy

                        t = (apx * abx + apy * aby) / ab_len_sq
                        t = max(0.0, min(1.0, t))

                        closest_x = sx + t * abx
                        closest_y = sy + t * aby

                        dist_segment = self._euclidean(
                            center_x - closest_x, center_y - closest_y
                        )

                        if dist_segment <= threshold and 0.2 <= t <= 0.8:
                            candidates.append(cells)

            if candidates:
                return candidates

        return []

    def _candidate_shapes_at(
        self, cx: int, cy: int, length: int
    ) -> list[tuple[tuple[int, int], ...]]:

        if self.scenario == "wall":
            horizontal = tuple((cx, cy + i) for i in range(length))
            vertical = tuple((cx + i, cy) for i in range(length))
            return [horizontal, vertical]

        if self.scenario == "l_shape":
            shapes = [
                tuple([(cx + i, cy) for i in range(length)] + [(cx, cy + j) for j in range(1, length)]),
                tuple([(cx + i, cy) for i in range(length)] + [(cx, cy - j) for j in range(1, length)]),
                tuple([(cx - i, cy) for i in range(length)] + [(cx, cy + j) for j in range(1, length)]),
                tuple([(cx - i, cy) for i in range(length)] + [(cx, cy - j) for j in range(1, length)]),
            ]
            return [tuple(sorted(set(s))) for s in shapes]

        # u_shape
        shapes = [
            tuple(
                [(cx + i, cy) for i in range(length)]
                + [(cx + i, cy + length - 1) for i in range(length)]
                + [(cx + length - 1, cy + j) for j in range(1, length - 1)]
            ),
            tuple(
                [(cx - i, cy) for i in range(length)]
                + [(cx - i, cy + length - 1) for i in range(length)]
                + [(cx - length + 1, cy + j) for j in range(1, length - 1)]
            ),
            tuple(
                [(cx, cy + j) for j in range(length)]
                + [(cx + length - 1, cy + j) for j in range(length)]
                + [(cx + i, cy + length - 1) for i in range(1, length - 1)]
            ),
            tuple(
                [(cx, cy - j) for j in range(length)]
                + [(cx + length - 1, cy - j) for j in range(length)]
                + [(cx + i, cy - length + 1) for i in range(1, length - 1)]
            ),
        ]
        return [tuple(sorted(set(s))) for s in shapes]

    def _is_inside_obstacle_region(self, pos: tuple[int, int]) -> bool:

        if self.scenario != "u_shape" or not self.obstacle_cells:
            return False

        xs = [c[0] for c in self.obstacle_cells]
        ys = [c[1] for c in self.obstacle_cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        x, y = pos
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        return pos not in self.obstacle_cells

    def _random_start_goal_obstacle(self):
        n = self.grid_size
        length = self.obstacle_length

        while True:
            start = (self.random.randrange(n), self.random.randrange(n))
            goal = (self.random.randrange(n), self.random.randrange(n))

            if start == goal:
                continue

            dist = self._euclidean(start[0] - goal[0], start[1] - goal[1])
            if dist < max(self.min_goal_start_distance, length + 1):
                continue

            candidates = self._build_obstacle_candidates(start, goal)
            if not candidates:
                continue

            obstacle_cells = self.random.choice(candidates)
            return start, goal, obstacle_cells

    def reset(self) -> tuple[int, int, int, int]:
        self.start_pos, self.goal_pos, self.obstacle_cells = (
            self._random_start_goal_obstacle()
        )
        self.agent_pos = self.start_pos
        self.current_step = 0
        self._reached_goal_once = False

        return self._relative_state(self.agent_pos)

    def _relative_state(self, pos: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = pos
        gx, gy = self.goal_pos
        ox, oy = self._nearest_obstacle_cell(pos)

        return (gx - x, gy - y, ox - x, oy - y)

    def _move(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        x, y = pos
        n = self.grid_size

        if action == 0:
            candidate = (x, min(y + 1, n - 1))
        elif action == 1:
            candidate = (x, max(y - 1, 0))
        elif action == 2:
            candidate = (max(x - 1, 0), y)
        elif action == 3:
            candidate = (min(x + 1, n - 1), y)
        elif action == self.STAY_ACTION:
            candidate = pos
        else:
            raise ValueError(f"Acción inválida: {action}")

        hit = candidate in self.obstacle_cells

        if hit:
            return pos, True

        return candidate, False

    def step(self, action: int) -> PaperStepResult:
        if action not in range(self.N_ACTIONS):
            raise ValueError(f"Acción inválida: {action}")

        params = self.reward_params
        self.current_step += 1

        pos = self.agent_pos
        gx, gy = self.goal_pos

        dx, dy = gx - pos[0], gy - pos[1]
        ox, oy = self._nearest_obstacle_cell(pos)
        dxo, dyo = ox - pos[0], oy - pos[1]

        next_pos, hit = self._move(pos, action)

        dx2, dy2 = gx - next_pos[0], gy - next_pos[1]
        ox2, oy2 = self._nearest_obstacle_cell(next_pos)
        dxo2, dyo2 = ox2 - next_pos[0], oy2 - next_pos[1]

        d_t = self._euclidean(dx, dy)
        d_t1 = self._euclidean(dx2, dy2)
        d_obs_t = self._euclidean(dxo, dyo)

        in_goal_now = (dx == 0 and dy == 0)
        in_goal_next = (dx2 == 0 and dy2 == 0)
        stay_action = action == self.STAY_ACTION
        move_action = not stay_action

        newly_arrived = in_goal_next and not in_goal_now
        if newly_arrived:
            self._reached_goal_once = True

        goal_position_reward = params["goal_position_scale"] * (
            params["distance_scale_constant"] - d_t
        )
        obstacle_position_penalty = params["obstacle_position_scale"] * (
            params["distance_scale_constant"] - d_obs_t
        )
        arrival_bonus = (
            params["arrival_bonus_multiplier"] * float(newly_arrived)
        )
        stay_in_goal_reward = params["goal_stay_reward"] * float(
            in_goal_now and stay_action
        )
        stay_outside_penalty = params["stay_outside_penalty"] * float(
            (not in_goal_now) and stay_action
        )
        obstacle_hit_penalty = params["obstacle_hit_penalty"] * float(hit)
        move_after_goal_penalty = params["goal_stay_out_penalty"] * float(
            in_goal_now and move_action
        )

        reward = (
            goal_position_reward
            - obstacle_position_penalty
            - params["step_penalty"]
            + arrival_bonus
            + stay_in_goal_reward
            - stay_outside_penalty
            - obstacle_hit_penalty
            - move_after_goal_penalty
        )

        self.agent_pos = next_pos
        done = self.current_step >= self.max_steps

        info = {
            "collision": hit,
            "reached_goal": in_goal_next,
            "newly_arrived": newly_arrived,
            "step": self.current_step,
            "position": self.agent_pos,
            "distance_to_goal": d_t1,
            "agent_inside_obstacle_region": self._is_inside_obstacle_region(self.agent_pos),
        }

        return PaperStepResult(
            next_state=(dx2, dy2, dxo2, dyo2),
            reward=float(reward),
            done=done,
            info=info,
        )
