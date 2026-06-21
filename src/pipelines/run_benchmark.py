from pathlib import Path
from datetime import datetime
import argparse
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_EXPERIMENT_PLAN = [
    {
        "algorithm": "q_learning",
        "scenario": "wall",
        "reward_version": "reward_v1_base",
        "validation_mode": "normal",
        "seed": 42,
    },
    {
        "algorithm": "q_learning",
        "scenario": "l_shape",
        "reward_version": "reward_v2_step_penalty",
        "validation_mode": "normal",
        "seed": 43,
    },
    {
        "algorithm": "dqn",
        "scenario": "wall",
        "reward_version": "reward_v1_base",
        "validation_mode": "normal",
        "seed": 44,
    },
    {
        "algorithm": "dqn",
        "scenario": "u_shape",
        "reward_version": "reward_v2_step_penalty",
        "validation_mode": "normal",
        "seed": 45,
    },
    {
        "algorithm": "sac",
        "scenario": "wall",
        "reward_version": "reward_v3_collision_penalty",
        "validation_mode": "flexible",
        "seed": 46,
    },
    {
        "algorithm": "sac",
        "scenario": "l_shape",
        "reward_version": "reward_v3_collision_penalty",
        "validation_mode": "flexible",
        "seed": 47,
    },
]


def build_benchmark_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"benchmark_{timestamp}"


def run_experiment(
    experiment: dict,
    episodes: int,
    checkpoint_interval: int,
    benchmark_id: str,
    index: int,
    total: int,
):
    python_executable = sys.executable

    command = [
        python_executable,
        "src/pipelines/run_local_pipeline.py",
        "--algorithm",
        experiment["algorithm"],
        "--scenario",
        experiment["scenario"],
        "--reward-version",
        experiment["reward_version"],
        "--validation-mode",
        experiment["validation_mode"],
        "--episodes",
        str(episodes),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--seed",
        str(experiment["seed"]),
    ]

    env = os.environ.copy()
    env["BENCHMARK_ID"] = benchmark_id
    env["RUN_GROUP"] = "benchmark"

    print("\n" + "=" * 80)
    print(f"Ejecutando experimento {index}/{total}")
    print("=" * 80)
    print(f"algorithm: {experiment['algorithm']}")
    print(f"scenario: {experiment['scenario']}")
    print(f"reward_version: {experiment['reward_version']}")
    print(f"validation_mode: {experiment['validation_mode']}")
    print(f"seed: {experiment['seed']}")
    print("\nComando:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Falló el experimento {index}/{total}: {experiment}"
        )

    print(f"Experimento {index}/{total} completado.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta un benchmark local de agentes RL y registra resultados en MLflow."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Número de episodios por corrida.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Intervalo para guardar checkpoints.",
    )

    parser.add_argument(
        "--benchmark-id",
        type=str,
        default=None,
        help="Identificador del benchmark. Si no se pasa, se genera automáticamente.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    benchmark_id = args.benchmark_id or build_benchmark_id()
    total = len(DEFAULT_EXPERIMENT_PLAN)

    print("\n" + "#" * 80)
    print("BENCHMARK RL MLOPS")
    print("#" * 80)
    print(f"benchmark_id: {benchmark_id}")
    print(f"n_experiments: {total}")
    print(f"episodes: {args.episodes}")
    print(f"checkpoint_interval: {args.checkpoint_interval}")

    for index, experiment in enumerate(DEFAULT_EXPERIMENT_PLAN, start=1):
        run_experiment(
            experiment=experiment,
            episodes=args.episodes,
            checkpoint_interval=args.checkpoint_interval,
            benchmark_id=benchmark_id,
            index=index,
            total=total,
        )

    print("\n" + "#" * 80)
    print("Benchmark completado correctamente.")
    print("#" * 80)
    print(f"benchmark_id: {benchmark_id}")
    print("\nAbre MLflow con:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("\nLuego filtra por tag:")
    print(f"benchmark_id = {benchmark_id}")


if __name__ == "__main__":
    main()

