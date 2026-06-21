from pathlib import Path
import json
import pandas as pd
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEW_RUN_PATH = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "validation_result.json"
REPORTS_TRAINING_DIR = PROJECT_ROOT / "reports" / "training"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CANDIDATE_MODELS_DIR = PROJECT_ROOT / "models" / "candidate"

EXPERIMENT_NAME = "rl_historical_validation"


def load_new_run(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de nueva corrida: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("El archivo de nueva corrida está vacío.")

    return df.iloc[0].to_dict()


def load_validation_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe el reporte de validación: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_episode_metrics_file(run_id: str) -> Path | None:
    matches = list(REPORTS_TRAINING_DIR.glob(f"{run_id}_episode_metrics.csv"))
    return matches[0] if matches else None


def find_checkpoint_dir(run_id: str) -> Path | None:
    path = CHECKPOINTS_DIR / run_id
    return path if path.exists() else None


def find_candidate_model_dir(run_id: str) -> Path | None:
    path = CANDIDATE_MODELS_DIR / run_id
    return path if path.exists() else None


def log_params(new_run: dict):
    param_names = [
        "run_id",
        "algorithm",
        "scenario",
        "reward_version",
        "validation_mode",
        "episodes",
        "seed",
        "notes",
    ]

    for name in param_names:
        if name in new_run:
            mlflow.log_param(name, new_run[name])


def log_metrics(new_run: dict, validation_report: dict):
    metric_names = [
        "success_rate",
        "avg_reward",
        "first_reach_step",
        "collisions",
        "avg_steps",
        "training_time_sec",
    ]

    for name in metric_names:
        if name in new_run and pd.notna(new_run[name]):
            mlflow.log_metric(name, float(new_run[name]))

    mlflow.log_metric("history_rows_used", float(validation_report["history_rows_used"]))
    mlflow.log_metric("preliminary_validation", float(validation_report["preliminary_validation"]))


def log_tags(validation_report: dict):
    mlflow.set_tag("project", "rl_mlops_historical_validation")
    mlflow.set_tag("final_status", validation_report["final_status"])
    mlflow.set_tag("reward_version", validation_report["reward_version"])
    mlflow.set_tag("algorithm", validation_report["algorithm"])
    mlflow.set_tag("scenario", validation_report["scenario"])


def log_artifacts(run_id: str):
    mlflow.log_artifact(str(NEW_RUN_PATH), artifact_path="new_run")
    mlflow.log_artifact(str(VALIDATION_REPORT_PATH), artifact_path="validation")

    episode_metrics_file = find_episode_metrics_file(run_id)
    if episode_metrics_file is not None:
        mlflow.log_artifact(str(episode_metrics_file), artifact_path="training_metrics")

    checkpoint_dir = find_checkpoint_dir(run_id)
    if checkpoint_dir is not None:
        mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")

    candidate_model_dir = find_candidate_model_dir(run_id)
    if candidate_model_dir is not None:
        mlflow.log_artifacts(str(candidate_model_dir), artifact_path="candidate_model")


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    new_run = load_new_run(NEW_RUN_PATH)
    validation_report = load_validation_report(VALIDATION_REPORT_PATH)

    run_id = new_run["run_id"]

    with mlflow.start_run(run_name=run_id):
        log_params(new_run)
        log_metrics(new_run, validation_report)
        log_tags(validation_report)
        log_artifacts(run_id)

    print("Registro en MLflow completado.")
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"run_id: {run_id}")
    print(f"final_status: {validation_report['final_status']}")


if __name__ == "__main__":
    main()

