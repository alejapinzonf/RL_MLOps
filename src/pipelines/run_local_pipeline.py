from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


VALID_ALGORITHMS = ["q_learning", "dqn", "sac"]
VALID_SCENARIOS = ["wall", "l_shape", "u_shape"]
VALID_REWARD_VERSIONS = [
    "reward_v1_base",
    "reward_v2_step_penalty",
    "reward_v3_collision_penalty",
]
VALIDATION_MODES = ["strict", "normal", "flexible"]


def run_step(step_name: str, command: list[str]):
    """
    Ejecuta un paso del pipeline y detiene todo si algo falla.
    """

    print("\n" + "=" * 60)
    print(f"Ejecutando paso: {step_name}")
    print("=" * 60)

    print("Comando:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Falló el paso: {step_name}")

    print(f"Paso completado: {step_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline local para entrenamiento demo y validación histórica RL."
    )

    parser.add_argument(
        "--algorithm",
        choices=VALID_ALGORITHMS,
        default="q_learning",
        help="Algoritmo RL.",
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
        help="Modo de validación.",
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
        help="Intervalo para guardar checkpoints.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    python_executable = sys.executable

    run_step(
        step_name="Cargar datos históricos",
        command=[
            python_executable,
            "src/data/load_data.py",
        ],
    )

    run_step(
        step_name="Limpiar datos históricos",
        command=[
            python_executable,
            "src/data/clean_data.py",
        ],
    )

    training_command = [
        python_executable,
        "src/training/demo_train.py",
        "--algorithm",
        args.algorithm,
        "--scenario",
        args.scenario,
        "--reward-version",
        args.reward_version,
        "--validation-mode",
        args.validation_mode,
        "--episodes",
        str(args.episodes),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]

    if args.seed is not None:
        training_command.extend(["--seed", str(args.seed)])

    run_step(
        step_name="Entrenamiento demo con checkpoints",
        command=training_command,
    )

    run_step(
        step_name="Validación histórica",
        command=[
            python_executable,
            "src/validation/validate_run.py",
        ],
    )

    print("\n" + "=" * 60)
    print("Pipeline local completado correctamente.")
    print("=" * 60)

    print("\nArchivos principales generados:")
    print("- data/interim/historical_loaded.csv")
    print("- data/processed/historical_results_clean.csv")
    print("- data/new_runs/new_run_demo.csv")
    print("- models/checkpoints/")
    print("- models/candidate/")
    print("- reports/training/")
    print("- reports/validation/validation_result.json")


if __name__ == "__main__":
    main()
