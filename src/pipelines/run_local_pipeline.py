from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advisor.historical_advisor import advise as historical_advise
from advisor.historical_advisor import format_advice as format_historical_advice
from advisor.reward_shaping_advisor import analyze_reward_params, format_warnings


VALID_ALGORITHMS = ["q_learning", "dqn", "sac"]
VALID_SCENARIOS = ["wall", "l_shape", "u_shape"]
VALID_REWARD_VERSIONS = [
    "reward_v1_base",
    "reward_v2_step_penalty",
    "reward_v3_collision_penalty",
]
VALIDATION_MODES = ["strict", "normal", "flexible"]

TRAINER_REGISTRY = {
    "demo": {
        "script": "src/training/demo_train.py",
        "fixed_algorithm": None,  # usa --algorithm del usuario
        "reward_version": None,  # usa --reward-version del usuario
        "uses_paper_reward": False,
        "description": "Entrenamiento simulado (no entrena un agente real). Soporta q_learning/dqn/sac.",
    },
    "real_q_learning": {
        "script": "src/training/train_q_learning.py",
        "fixed_algorithm": "q_learning",
        "reward_version": None,  # usa --reward-version del usuario (reward simple)
        "uses_paper_reward": False,
        "description": "Q-learning real sobre GridWorldEnv simplificado (8x8, reward simple).",
    },
    "real_q_learning_paper": {
        "script": "src/training/train_q_learning_paper.py",
        "fixed_algorithm": "q_learning",
        "reward_version": "paper_reward_v1",
        "uses_paper_reward": True,
        "description": "Q-learning real sobre PaperGridWorldEnv (20x20, reward fiel al paper).",
    },
    "real_dqn_paper": {
        "script": "src/training/train_dqn_paper.py",
        "fixed_algorithm": "dqn",
        "reward_version": "paper_reward_v1",
        "uses_paper_reward": True,
        "description": "DQN real (PyTorch) sobre PaperGridWorldEnv.",
    },
    "real_sac_paper": {
        "script": "src/training/train_sac_paper.py",
        "fixed_algorithm": "sac",
        "reward_version": "paper_reward_v1",
        "uses_paper_reward": True,
        "description": "Discrete SAC real (PyTorch) sobre PaperGridWorldEnv.",
    },
}

DEFAULT_PAPER_REWARD_PARAMS = {
    "arrival_reward": 150.0,
    "goal_stay_reward": 50.0,
    "goal_stay_out_penalty": 100.0,
    "step_penalty": 1.0,
    "stay_outside_penalty": 60.0,
    "obstacle_hit_penalty": 100.0,
    "goal_position_scale": 2.0,
    "obstacle_position_scale": 2.0,
    "arrival_bonus_multiplier": 100.0,
}


def run_step(step_name: str, command: list[str]):
    """
    Ejecuta un paso del pipeline y detiene todo si algo falla.
    """

    print("\n" + "=" * 70)
    print(f"Ejecutando paso: {step_name}")
    print("=" * 70)

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


def resolve_algorithm(trainer_config: dict, requested_algorithm: str) -> str:
    """
    Determina el algorithm efectivo: si el trainer tiene uno fijo
    (p. ej. 'real_dqn_paper' siempre es dqn), valida que no choque con
    lo que el usuario pidió explícitamente; si no hay fijo, usa el
    solicitado.
    """
    fixed = trainer_config["fixed_algorithm"]

    if fixed is None:
        return requested_algorithm

    return fixed


def run_advisors(
    args: argparse.Namespace,
    trainer_config: dict,
    effective_algorithm: str,
) -> bool:

    print("\n" + "=" * 70)
    print("Asesor de hiperparámetros (antes de entrenar)")
    print("=" * 70)

    effective_reward_version = trainer_config["reward_version"] or args.reward_version

    historical_advice = historical_advise(
        algorithm=effective_algorithm,
        scenario=args.scenario,
        planned_episodes=args.episodes,
        reward_version=effective_reward_version,
    )
    print("\n[Histórico]")
    print(format_historical_advice(historical_advice))

    unfavorable = historical_advice["verdict"] == "likely_insufficient"

    if trainer_config["uses_paper_reward"]:
        warnings = analyze_reward_params(DEFAULT_PAPER_REWARD_PARAMS)
        print("\n[Reward shaping]")
        print(format_warnings(warnings))

        if any(w.severity == "high" for w in warnings):
            unfavorable = True

    if args.skip_advisor:
        return True

    if unfavorable:
        print("\nEl asesor detectó posibles problemas con esta configuración.")
        answer = input("¿Quieres continuar igualmente? [s/N]: ").strip().lower()
        return answer in ("s", "si", "sí", "y", "yes")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline local RL MLOps con entrenamiento, validación y MLflow."
    )

    parser.add_argument(
        "--trainer",
        choices=list(TRAINER_REGISTRY.keys()),
        default="demo",
        help=(
            "Entrenador a usar: "
            + "; ".join(f"{name} ({cfg['description']})" for name, cfg in TRAINER_REGISTRY.items())
        ),
    )

    parser.add_argument("--algorithm", choices=VALID_ALGORITHMS, default="q_learning")
    parser.add_argument("--scenario", choices=VALID_SCENARIOS, default="wall")
    parser.add_argument("--reward-version", choices=VALID_REWARD_VERSIONS, default="reward_v1_base")
    parser.add_argument("--validation-mode", choices=VALIDATION_MODES, default="normal")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument(
        "--skip-advisor",
        action="store_true",
        help="Omite los avisos del asesor de hiperparámetros y no pregunta confirmación.",
    )

    return parser.parse_args()


def build_training_command(
    python_executable: str,
    trainer_config: dict,
    effective_algorithm: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [python_executable, trainer_config["script"]]

    # 'demo' es el único trainer que acepta --algorithm explícito,
    # porque es el único que soporta dqn/sac simulados.
    if trainer_config["fixed_algorithm"] is None:
        command.extend(["--algorithm", effective_algorithm])

    command.extend(["--scenario", args.scenario])

    if trainer_config["reward_version"] is None:
        command.extend(["--reward-version", args.reward_version])

    command.extend(["--validation-mode", args.validation_mode])
    command.extend(["--episodes", str(args.episodes)])
    command.extend(["--checkpoint-interval", str(args.checkpoint_interval)])

    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])

    return command


def main():
    args = parse_args()
    python_executable = sys.executable

    trainer_config = TRAINER_REGISTRY[args.trainer]
    effective_algorithm = resolve_algorithm(trainer_config, args.algorithm)

    should_continue = run_advisors(args, trainer_config, effective_algorithm)

    if not should_continue:
        print("\nEntrenamiento cancelado por el usuario tras el aviso del asesor.")
        return

    run_step(
        step_name="Cargar datos históricos",
        command=[python_executable, "src/data/load_data.py"],
    )

    run_step(
        step_name="Limpiar datos históricos",
        command=[python_executable, "src/data/clean_data.py"],
    )

    training_command = build_training_command(
        python_executable, trainer_config, effective_algorithm, args
    )

    run_step(
        step_name=f"Entrenamiento ({args.trainer})",
        command=training_command,
    )

    run_step(
        step_name="Validación histórica",
        command=[python_executable, "src/validation/validate_run.py"],
    )

    run_step(
        step_name="Análisis de convergencia",
        command=[python_executable, "src/evaluation/evaluate_run.py"],
    )

    if not args.skip_mlflow:
        run_step(
            step_name="Registro en MLflow",
            command=[python_executable, "src/tracking/log_to_mlflow.py"],
        )

    print("\n" + "=" * 70)
    print("Pipeline local completado correctamente.")
    print("=" * 70)

    print(f"\nTrainer usado: {args.trainer}")
    print("\nArchivos principales generados:")
    print("- data/interim/historical_loaded.csv")
    print("- data/processed/historical_results_clean.csv")
    print("- data/new_runs/new_run_demo.csv")
    print("- models/checkpoints/")
    print("- models/candidate/")
    print("- reports/training/")
    print("- reports/validation/validation_result.json")
    print("- reports/evaluation/evaluation_result.json")

    if not args.skip_mlflow:
        print("- mlflow.db")
        print("- MLflow experiment: rl_historical_validation")


if __name__ == "__main__":
    main()
