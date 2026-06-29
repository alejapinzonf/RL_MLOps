from pathlib import Path
from datetime import datetime
import argparse
import random
import pandas as pd
import sys

from checkpoint_utils import save_checkpoint, save_final_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.checkpoint_utils import save_checkpoint, save_final_model
NEW_RUNS_DIR = PROJECT_ROOT / "data" / "new_runs"
NEW_RUN_PATH = NEW_RUNS_DIR / "new_run_demo.csv"

CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CANDIDATE_MODELS_DIR = PROJECT_ROOT / "models" / "candidate"


VALID_ALGORITHMS = ["q_learning", "dqn", "sac"]
VALID_SCENARIOS = ["wall", "l_shape", "u_shape"]
VALID_REWARD_VERSIONS = [
    "reward_v1_base",
    "reward_v2_step_penalty",
    "reward_v3_collision_penalty",
]
VALIDATION_MODES = ["strict", "normal", "flexible"]


BASE_METRICS = {
    "q_learning": {
        "success_rate": 0.86,
        "avg_reward": 160,
        "first_reach_step": 65,
        "collisions": 4,
        "avg_steps": 85,
        "training_time_sec": 120,
    },
    "dqn": {
        "success_rate": 0.88,
        "avg_reward": 175,
        "first_reach_step": 58,
        "collisions": 3,
        "avg_steps": 78,
        "training_time_sec": 320,
    },
    "sac": {
        "success_rate": 0.80,
        "avg_reward": 145,
        "first_reach_step": 75,
        "collisions": 5,
        "avg_steps": 95,
        "training_time_sec": 500,
    },
}


SCENARIO_EFFECTS = {
    "wall": {
        "success_rate": 0.05,
        "avg_reward": 20,
        "first_reach_step": -15,
        "collisions": -1,
        "avg_steps": -20,
    },
    "l_shape": {
        "success_rate": -0.04,
        "avg_reward": -15,
        "first_reach_step": 15,
        "collisions": 2,
        "avg_steps": 20,
    },
    "u_shape": {
        "success_rate": -0.08,
        "avg_reward": -25,
        "first_reach_step": 25,
        "collisions": 3,
        "avg_steps": 35,
    },
}


REWARD_EFFECTS = {
    "reward_v1_base": {
        "success_rate": 0.00,
        "avg_reward": 0,
        "first_reach_step": 0,
        "collisions": 0,
        "avg_steps": 0,
    },
    "reward_v2_step_penalty": {
        "success_rate": 0.01,
        "avg_reward": -5,
        "first_reach_step": -8,
        "collisions": 0,
        "avg_steps": -10,
    },
    "reward_v3_collision_penalty": {
        "success_rate": 0.00,
        "avg_reward": -10,
        "first_reach_step": 0,
        "collisions": -2,
        "avg_steps": -5,
    },
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def generate_run_id(algorithm: str, scenario: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"new_{algorithm}_{scenario}_{timestamp}"


def get_target_metrics(
    algorithm: str,
    scenario: str,
    reward_version: str,
) -> dict:
    """
    Calcula métricas objetivo de la corrida simulada.
    Estas representan el desempeño final esperado.
    """

    base = BASE_METRICS[algorithm]
    scenario_effect = SCENARIO_EFFECTS[scenario]
    reward_effect = REWARD_EFFECTS[reward_version]

    target = {}

    for metric in base:
        target[metric] = (
            base[metric]
            + scenario_effect.get(metric, 0)
            + reward_effect.get(metric, 0)
        )

    return target


def simulate_training_curve(
    target_metrics: dict,
    episodes: int,
    seed: int | None = None,
) -> list[dict]:
    if seed is not None:
        random.seed(seed)

    episode_metrics = []

    for episode in range(1, episodes + 1):
        progress = episode / episodes

        success_rate = target_metrics["success_rate"] * progress
        avg_reward = target_metrics["avg_reward"] * progress

        first_reach_step = (
            target_metrics["first_reach_step"]
            + (1 - progress) * 80
        )

        collisions = (
            target_metrics["collisions"]
            + (1 - progress) * 8
        )

        avg_steps = (
            target_metrics["avg_steps"]
            + (1 - progress) * 100
        )

        metrics = {
            "episode": episode,
            "episode_reward": round(avg_reward + random.gauss(0, 8), 4),
            "success_rate": round(
                clamp(success_rate + random.gauss(0, 0.03), 0.0, 1.0),
                4,
            ),
            "first_reach_step": round(max(first_reach_step + random.gauss(0, 5), 1), 4),
            "collisions": round(max(collisions + random.gauss(0, 1), 0), 4),
            "avg_steps": round(max(avg_steps + random.gauss(0, 8), 1), 4),
        }

        episode_metrics.append(metrics)

    return episode_metrics


def summarize_final_metrics(
    episode_metrics: list[dict],
    target_metrics: dict,
    training_time_sec: float,
    window_size: int = 20,
) -> dict:
    """
    Calcula métricas finales usando los últimos episodios.
    """

    last_window = episode_metrics[-window_size:]

    final_success_rate = sum(item["success_rate"] for item in last_window) / len(last_window)
    final_avg_reward = sum(item["episode_reward"] for item in last_window) / len(last_window)
    final_first_reach_step = sum(item["first_reach_step"] for item in last_window) / len(last_window)
    final_collisions = sum(item["collisions"] for item in last_window) / len(last_window)
    final_avg_steps = sum(item["avg_steps"] for item in last_window) / len(last_window)

    final_metrics = {
        "success_rate": round(clamp(final_success_rate, 0.0, 1.0), 4),
        "avg_reward": round(final_avg_reward, 4),
        "first_reach_step": round(max(final_first_reach_step, 1), 4),
        "collisions": round(max(final_collisions, 0), 4),
        "avg_steps": round(max(final_avg_steps, 1), 4),
        "training_time_sec": round(max(training_time_sec, 1), 4),
    }

    return final_metrics


def save_episode_metrics(
    run_id: str,
    episode_metrics: list[dict],
):
    """
    Guarda las métricas por episodio en CSV.
    """

    output_dir = PROJECT_ROOT / "reports" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{run_id}_episode_metrics.csv"

    df = pd.DataFrame(episode_metrics)
    df.to_csv(output_path, index=False)

    return output_path


def build_new_run(
    args: argparse.Namespace,
    run_id: str,
    final_metrics: dict,
) -> dict:
    run = {
        "run_id": run_id,
        "algorithm": args.algorithm,
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
        "notes": "simulated training run with checkpoints",
    }

    return run


def save_new_run(run: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([run])
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento demo para pipeline RL MLOps con checkpoints."
    )

    parser.add_argument(
        "--algorithm",
        choices=VALID_ALGORITHMS,
        default="q_learning",
        help="Algoritmo RL usado en la corrida demo.",
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
        help="Versión de la función de recompensa.",
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
        default=100,
        help="Número de episodios simulados.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Cada cuántos episodios se guarda un checkpoint.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria para reproducibilidad.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval debe ser mayor que 0.")

    run_id = generate_run_id(args.algorithm, args.scenario)

    target_metrics = get_target_metrics(
        algorithm=args.algorithm,
        scenario=args.scenario,
        reward_version=args.reward_version,
    )

    episode_metrics = simulate_training_curve(
        target_metrics=target_metrics,
        episodes=args.episodes,
        seed=args.seed,
    )

    checkpoint_paths = []

    for item in episode_metrics:
        episode = item["episode"]

        if episode % args.checkpoint_interval == 0 or episode == args.episodes:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=CHECKPOINTS_DIR,
                run_id=run_id,
                episode=episode,
                algorithm=args.algorithm,
                scenario=args.scenario,
                reward_version=args.reward_version,
                metrics=item,
            )
            checkpoint_paths.append(checkpoint_path)

    final_metrics = summarize_final_metrics(
        episode_metrics=episode_metrics,
        target_metrics=target_metrics,
        training_time_sec=target_metrics["training_time_sec"],
    )

    model_path, metadata_path = save_final_model(
        model_dir=CANDIDATE_MODELS_DIR,
        run_id=run_id,
        algorithm=args.algorithm,
        scenario=args.scenario,
        reward_version=args.reward_version,
        final_metrics=final_metrics,
    )

    episode_metrics_path = save_episode_metrics(
        run_id=run_id,
        episode_metrics=episode_metrics,
    )

    new_run = build_new_run(
        args=args,
        run_id=run_id,
        final_metrics=final_metrics,
    )

    save_new_run(new_run, NEW_RUN_PATH)

    print("Entrenamiento demo completado.")
    print(f"run_id: {run_id}")
    print(f"Archivo de nueva corrida: {NEW_RUN_PATH}")
    print(f"Métricas por episodio: {episode_metrics_path}")
    print(f"Modelo candidato: {model_path}")
    print(f"Metadata del modelo: {metadata_path}")

    print("\nCheckpoints guardados:")
    for path in checkpoint_paths:
        print(f"- {path}")

    print("\nMétricas finales:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
