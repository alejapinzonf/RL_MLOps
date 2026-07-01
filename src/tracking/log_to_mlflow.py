from pathlib import Path
import json
import pandas as pd
import mlflow
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEW_RUN_PATH = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "validation_result.json"

REPORTS_TRAINING_DIR = PROJECT_ROOT / "reports" / "training"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CANDIDATE_MODELS_DIR = PROJECT_ROOT / "models" / "candidate"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "rl_historical_validation"


def configure_mlflow():
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_new_run(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de nueva corrida: {path}. "
            "Primero ejecuta el pipeline local."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("El archivo de nueva corrida está vacío.")

    return df.iloc[0].to_dict()


def load_validation_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el reporte de validación: {path}. "
            "Primero ejecuta validate_run.py."
        )

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


def log_run_params(new_run: dict):
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
        value = new_run.get(name)

        if pd.notna(value):
            mlflow.log_param(name, value)


def log_run_metrics(new_run: dict, validation_report: dict):
    metric_names = [
        "success_rate",
        "avg_reward",
        "first_reach_step",
        "collisions",
        "avg_steps",
        "training_time_sec",
    ]

    for name in metric_names:
        value = new_run.get(name)

        if pd.notna(value):
            mlflow.log_metric(name, float(value))

    mlflow.log_metric(
        "history_rows_used",
        float(validation_report.get("history_rows_used", 0)),
    )

    mlflow.log_metric(
        "preliminary_validation",
        float(bool(validation_report.get("preliminary_validation", False))),
    )


def log_metric_level_validation(validation_report: dict):
    """
    Registra como métricas auxiliares el estado de cada métrica validada.

    Codificación:
    approved = 2
    warning = 1
    rejected = 0
    """
    status_score = {
        "approved": 2,
        "warning": 1,
        "rejected": 0,
    }

    metric_results = validation_report.get("metric_results", [])

    for result in metric_results:
        metric_name = result.get("metric")
        status = result.get("status")

        if metric_name is None or status is None:
            continue

        score = status_score.get(status, -1)
        mlflow.log_metric(f"validation_{metric_name}_status_score", score)


def log_run_tags(validation_report: dict):
    benchmark_id = os.environ.get("BENCHMARK_ID", "single_run")
    run_group = os.environ.get("RUN_GROUP", "local_pipeline")

    mlflow.set_tag("project", "rl_mlops_historical_validation")
    mlflow.set_tag("pipeline_stage", "local_training_validation")
    mlflow.set_tag("benchmark_id", benchmark_id)
    mlflow.set_tag("run_group", run_group)

    mlflow.set_tag("final_status", validation_report.get("final_status"))
    mlflow.set_tag("algorithm", validation_report.get("algorithm"))
    mlflow.set_tag("scenario", validation_report.get("scenario"))
    mlflow.set_tag("reward_version", validation_report.get("reward_version"))
    mlflow.set_tag("validation_mode", validation_report.get("validation_mode"))


def log_run_artifacts(run_id: str):
    if NEW_RUN_PATH.exists():
        mlflow.log_artifact(str(NEW_RUN_PATH), artifact_path="new_run")

    if VALIDATION_REPORT_PATH.exists():
        mlflow.log_artifact(str(VALIDATION_REPORT_PATH), artifact_path="validation")

    episode_metrics_file = find_episode_metrics_file(run_id)
    if episode_metrics_file is not None:
        mlflow.log_artifact(
            str(episode_metrics_file),
            artifact_path="training_metrics",
        )

    checkpoint_dir = find_checkpoint_dir(run_id)
    if checkpoint_dir is not None:
        mlflow.log_artifacts(
            str(checkpoint_dir),
            artifact_path="checkpoints",
        )

    candidate_model_dir = find_candidate_model_dir(run_id)
    if candidate_model_dir is not None:
        mlflow.log_artifacts(
            str(candidate_model_dir),
            artifact_path="candidate_model",
        )


def log_current_run_to_mlflow() -> str:
    configure_mlflow()

    new_run = load_new_run(NEW_RUN_PATH)
    validation_report = load_validation_report(VALIDATION_REPORT_PATH)

    run_id = str(new_run["run_id"])
    final_status = validation_report.get("final_status", "unknown")

    mlflow_run_name = f"{run_id}_{final_status}"

    with mlflow.start_run(run_name=mlflow_run_name):
        log_run_params(new_run)
        log_run_metrics(new_run, validation_report)
        log_metric_level_validation(validation_report)
        log_run_tags(validation_report)
        log_run_artifacts(run_id)

        mlflow.set_tag("mlflow_run_name", mlflow_run_name)

    return run_id


def main():
    run_id = log_current_run_to_mlflow()

    validation_report = load_validation_report(VALIDATION_REPORT_PATH)

    print("Registro en MLflow completado.")
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"run_id: {run_id}")
    print(f"final_status: {validation_report.get('final_status')}")


if __name__ == "__main__":
    main()
