from pathlib import Path
from datetime import datetime
import argparse
import random
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment.grid_world import GridWorldEnv
from training.checkpoint_utils import (
    save_q_table_checkpoint,
    save_q_table_model,
)


ALGORITHM = "q_learning"

NEW_RUNS_DIR = PROJECT_ROOT / "data" / "new_runs"
NEW_RUN_PATH = NEW_RUNS_DIR / "new_run_demo.csv"

CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CANDIDATE_MODELS_DIR = PROJECT_ROOT / "models" / "candidate"
TRAINING_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"


VALID_SCENARIOS = ["wall", "l_shape", "u_shape"]
VALID_REWARD_VERSIONS = [
    "reward_v1_base",
    "reward_v2_step_penalty",
    "reward_v3_collision_penalty",
]
VALIDATION_MODES = ["strict", "normal", "flexible"]


def generate_run_id(scenario: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"real_q_learning_{scenario}_{timestamp}"


def choose_action(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    n_actions: int,
) -> int:
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)

    return int(np.argmax(q_table[state]))


def train_one_episode(
    env: GridWorldEnv,
    q_table: np.ndarray,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> dict:
    state = env.reset()

    total_reward = 0.0
    collisions = 0
    reached_goal = False
    first_reach_step = None
    td_errors = []

    for step in range(1, env.max_steps + 1):
        action = choose_action(
            q_table=q_table,
            state=state,
            epsilon=epsilon,
            n_actions=env.n_actions,
        )

        result = env.step(action)

        old_value = q_table[state, action]
        best_next_value = np.max(q_table[result.next_state])

        td_target = result.reward + gamma * best_next_value * (not result.done)
        td_error = td_target - old_value

        q_table[state, action] = old_value + alpha * td_error

        total_reward += result.reward
        td_errors.append(abs(float(td_error)))

        if result.info["collision"]:
            collisions += 1

        if result.info["reached_goal"]:
            reached_goal = True
            first_reach_step = step

        state = result.next_state

        if result.done:
            break

    episode_metrics = {
        "episode_reward": total_reward,
        "success": int(reached_goal),
        "first_reach_step": first_reach_step if first_reach_step is not None else env.max_steps,
        "collisions": collisions,
        "steps": step,
        "td_error": float(np.mean(td_errors)) if td_errors else 0.0,
    }

    return episode_metrics


def summarize_final_metrics(
    episode_metrics: list[dict],
    training_time_sec: float,
    window_size: int,
) -> dict:
    window = episode_metrics[-window_size:]

    success_rate = np.mean([item["success"] for item in window])
    avg_reward = np.mean([item["episode_reward"] for item in window])
    first_reach_step = np.mean([item["first_reach_step"] for item in window])
    collisions = np.mean([item["collisions"] for item in window])
    avg_steps = np.mean([item["steps"] for item in window])

    return {
        "success_rate": round(float(success_rate), 4),
        "avg_reward": round(float(avg_reward), 4),
        "first_reach_step": round(float(first_reach_step), 4),
        "collisions": round(float(collisions), 4),
        "avg_steps": round(float(avg_steps), 4),
        "training_time_sec": round(float(training_time_sec), 4),
    }


def save_episode_metrics(run_id: str, episode_metrics: list[dict]) -> Path:
    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TRAINING_REPORTS_DIR / f"{run_id}_episode_metrics.csv"

    df = pd.DataFrame(episode_metrics)
    df.to_csv(output_path, index=False)

    return output_path


def save_new_run_csv(
    run_id: str,
    args: argparse.Namespace,
    final_metrics: dict,
):
    NEW_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    new_run = {
        "run_id": run_id,
        "algorithm": ALGORITHM,
        "scenario": args.scenario,
        "reward_version": args.reward_version,
        "validation_mode": args.validation_mode,
        "success_rate": final_metrics["success_rate"],
        "avg_reward": final_metrics["avg_reward"],
        "first_reach_step": final_metrics["first_reach_step"],
        "collisions": final_metrics["collisions"],
        "avg_steps": final_metrics["avg_steps"],
        "training_time_sec": final_metrics["training_time_sec"],
        "episodes": args.episodes,
        "seed": args.seed,
        "notes": "real q-learning training run",
    }

    pd.DataFrame([new_run]).to_csv(NEW_RUN_PATH, index=False)

    return NEW_RUN_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento real Q-learning para GridWorld."
    )

    parser.add_argument(
        "--scenario",
        choices=VALID_SCENARIOS,
        default="wall",
        help="Escenario de navegación.",
    )

    parser.add_argument(
        "--reward-version",
        choices=VALID_REWARD_VERSIONS,
        default="reward_v1_base",
        help="Versión de recompensa.",
    )

    parser.add_argument(
        "--validation-mode",
        choices=VALIDATION_MODES,
        default="normal",
        help="Modo de validación histórica.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Número de episodios de entrenamiento.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Intervalo para guardar checkpoints.",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Tamaño del grid.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Máximo de pasos por episodio.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Factor de descuento.",
    )

    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Epsilon inicial.",
    )

    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Epsilon mínimo.",
    )

    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Decaimiento multiplicativo de epsilon.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval debe ser mayor que 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = generate_run_id(args.scenario)

    env = GridWorldEnv(
        grid_size=args.grid_size,
        scenario=args.scenario,
        reward_version=args.reward_version,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)

    epsilon = args.epsilon_start
    episode_metrics = []
    checkpoint_paths = []

    start_time = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        metrics = train_one_episode(
            env=env,
            q_table=q_table,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=epsilon,
        )

        metrics["episode"] = episode
        metrics["epsilon"] = epsilon
        episode_metrics.append(metrics)

        if episode % args.checkpoint_interval == 0 or episode == args.episodes:
            checkpoint_path = save_q_table_checkpoint(
                checkpoint_dir=CHECKPOINTS_DIR,
                run_id=run_id,
                episode=episode,
                algorithm=ALGORITHM,
                scenario=args.scenario,
                reward_version=args.reward_version,
                q_table=q_table.copy(),
                metrics=metrics,
                epsilon=epsilon,
            )
            checkpoint_paths.append(checkpoint_path)

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if episode % max(1, args.episodes // 10) == 0:
            recent = episode_metrics[-20:]
            recent_success = np.mean([item["success"] for item in recent])
            recent_reward = np.mean([item["episode_reward"] for item in recent])

            print(
                f"Episode {episode}/{args.episodes} "
                f"| success_rate_20={recent_success:.2f} "
                f"| avg_reward_20={recent_reward:.2f} "
                f"| epsilon={epsilon:.3f}"
            )

    training_time_sec = time.perf_counter() - start_time

    final_metrics = summarize_final_metrics(
        episode_metrics=episode_metrics,
        training_time_sec=training_time_sec,
        window_size=min(50, args.episodes),
    )

    training_config = {
        "algorithm": ALGORITHM,
        "scenario": args.scenario,
        "reward_version": args.reward_version,
        "episodes": args.episodes,
        "grid_size": args.grid_size,
        "max_steps": args.max_steps,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "seed": args.seed,
    }

    episode_metrics_path = save_episode_metrics(
        run_id=run_id,
        episode_metrics=episode_metrics,
    )

    model_path, metadata_path = save_q_table_model(
        model_dir=CANDIDATE_MODELS_DIR,
        run_id=run_id,
        algorithm=ALGORITHM,
        scenario=args.scenario,
        reward_version=args.reward_version,
        q_table=q_table,
        final_metrics=final_metrics,
        training_config=training_config,
    )

    new_run_path = save_new_run_csv(
        run_id=run_id,
        args=args,
        final_metrics=final_metrics,
    )

    print("\nEntrenamiento real Q-learning completado.")
    print(f"run_id: {run_id}")
    print(f"Escenario: {args.scenario}")
    print(f"Reward version: {args.reward_version}")
    print(f"Archivo de nueva corrida: {new_run_path}")
    print(f"Métricas por episodio: {episode_metrics_path}")
    print(f"Modelo Q-table: {model_path}")
    print(f"Metadata del modelo: {metadata_path}")

    print("\nCheckpoints guardados:")
    for path in checkpoint_paths:
        print(f"- {path}")

    print("\nMétricas finales:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
