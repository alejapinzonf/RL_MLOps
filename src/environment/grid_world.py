from dataclasses import dataclass
import random


@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool
    info: dict


class GridWorldEnv:

    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(
        self,
        grid_size: int = 8,
        scenario: str = "wall",
        reward_version: str = "reward_v1_base",
        max_steps: int = 100,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.scenario = scenario
        self.reward_version = reward_version
        self.max_steps = max_steps

        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        self.random = random.Random(seed)

        self.obstacles = self._build_obstacles(scenario)
        self.agent_pos = self.start_pos
        self.current_step = 0

    @property
    def n_states(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def _build_obstacles(self, scenario: str) -> set[tuple[int, int]]:
        obstacles = set()
        n = self.grid_size

        if scenario == "wall":
            wall_col = n // 2
            gap_row = n // 2

            for row in range(1, n - 1):
                if row != gap_row:
                    obstacles.add((row, wall_col))

        elif scenario == "l_shape":
            col = n // 2
            row = n - 3

            for r in range(2, n - 1):
                obstacles.add((r, col))

            for c in range(col, n - 1):
                obstacles.add((row, c))

        elif scenario == "u_shape":
            left_col = 2
            right_col = n - 3
            bottom_row = n - 3

            for r in range(2, bottom_row + 1):
                obstacles.add((r, left_col))
                obstacles.add((r, right_col))

            for c in range(left_col, right_col + 1):
                obstacles.add((bottom_row, c))

        else:
            raise ValueError(
                f"Escenario no soportado: {scenario}. "
                "Usa: wall, l_shape o u_shape."
            )

        obstacles.discard(self.start_pos)
        obstacles.discard(self.goal_pos)

        return obstacles

    def _state_from_pos(self, pos: tuple[int, int]) -> int:
        row, col = pos
        return row * self.grid_size + col

    def _is_outside_grid(self, pos: tuple[int, int]) -> bool:
        row, col = pos
        return (
            row < 0
            or row >= self.grid_size
            or col < 0
            or col >= self.grid_size
        )

    def _get_rewards(self) -> dict:
        if self.reward_version == "reward_v1_base":
            return {
                "step": -1.0,
                "collision": -10.0,
                "goal": 100.0,
            }

        if self.reward_version == "reward_v2_step_penalty":
            return {
                "step": -2.0,
                "collision": -10.0,
                "goal": 100.0,
            }

        if self.reward_version == "reward_v3_collision_penalty":
            return {
                "step": -1.0,
                "collision": -20.0,
                "goal": 100.0,
            }

        raise ValueError(
            f"reward_version no soportada: {self.reward_version}"
        )

    def reset(self) -> int:
        self.agent_pos = self.start_pos
        self.current_step = 0
        return self._state_from_pos(self.agent_pos)

    def step(self, action: int) -> StepResult:
        if action not in self.ACTIONS:
            raise ValueError(f"Acción inválida: {action}")

        rewards = self._get_rewards()
        self.current_step += 1

        row, col = self.agent_pos
        d_row, d_col = self.ACTIONS[action]
        candidate_pos = (row + d_row, col + d_col)

        collision = False

        if self._is_outside_grid(candidate_pos) or candidate_pos in self.obstacles:
            collision = True
            next_pos = self.agent_pos
            reward = rewards["collision"]
        else:
            next_pos = candidate_pos
            reward = rewards["step"]

        reached_goal = next_pos == self.goal_pos

        if reached_goal:
            reward = rewards["goal"]

        self.agent_pos = next_pos

        done = reached_goal or self.current_step >= self.max_steps

        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "step": self.current_step,
            "position": self.agent_pos,
        }

        return StepResult(
            next_state=self._state_from_pos(self.agent_pos),
            reward=reward,
            done=done,
            info=info,
        )

    def render_text(self) -> str:
        rows = []

        for r in range(self.grid_size):
            row_items = []

            for c in range(self.grid_size):
                pos = (r, c)

                if pos == self.agent_pos:
                    row_items.append("A")
                elif pos == self.start_pos:
                    row_items.append("S")
                elif pos == self.goal_pos:
                    row_items.append("G")
                elif pos in self.obstacles:
                    row_items.append("#")
                else:
                    row_items.append(".")

            rows.append(" ".join(row_items))

        return "\n".join(rows)
