from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "historical_results_clean.csv"
NEW_RUN_PATH = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
VALIDATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "validation.yaml"
VALIDATION_REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
VALIDATION_REPORT_PATH = VALIDATION_REPORT_DIR / "validation_result.json"


GROUP_COLUMNS = [
    "algorithm",
    "scenario",
    "reward_version",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_historical_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el histórico limpio: {path}. "
            "Primero ejecuta: python src/load_data.py y python src/clean_data.py"
        )

    return pd.read_csv(path)


def load_new_run(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe la corrida nueva: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("El archivo de nueva corrida está vacío.")

    if len(df) > 1:
        print("Advertencia: el archivo tiene más de una fila. Se usará solo la primera.")

    return df.iloc[0].to_dict()


def filter_comparable_history(historical_df: pd.DataFrame, new_run: dict) -> pd.DataFrame:
    """
    Filtra el historial usando algorithm, scenario y reward_version.
    """
    filtered_df = historical_df.copy()

    for col in GROUP_COLUMNS:
        filtered_df = filtered_df[filtered_df[col] == new_run[col]]

    return filtered_df


def compute_metric_status(
    metric_name: str,
    new_value: float,
    historical_values: pd.Series,
    higher_is_better: bool,
    std_factor: float,
) -> dict:
    """
    Compara una métrica nueva contra el historial usando promedio y desviación estándar.
    """
    historical_values = pd.to_numeric(historical_values, errors="coerce").dropna()

    hist_mean = historical_values.mean()
    hist_std = historical_values.std(ddof=0)

    if pd.isna(new_value):
        return {
            "metric": metric_name,
            "status": "warning",
            "reason": "new_value_missing",
            "new_value": None,
            "historical_mean": hist_mean,
            "historical_std": hist_std,
            "threshold_approved": None,
            "threshold_warning": None,
        }

    if hist_std == 0 or pd.isna(hist_std):
        hist_std = 1e-9

    if higher_is_better:
        approved_threshold = hist_mean - std_factor * hist_std
        warning_threshold = hist_mean - 2 * std_factor * hist_std

        if new_value >= approved_threshold:
            status = "approved"
            reason = "within_expected_range"
        elif new_value >= warning_threshold:
            status = "warning"
            reason = "slightly_below_historical_range"
        else:
            status = "rejected"
            reason = "too_low_against_history"

    else:
        approved_threshold = hist_mean + std_factor * hist_std
        warning_threshold = hist_mean + 2 * std_factor * hist_std

        if new_value <= approved_threshold:
            status = "approved"
            reason = "within_expected_range"
        elif new_value <= warning_threshold:
            status = "warning"
            reason = "slightly_above_historical_range"
        else:
            status = "rejected"
            reason = "too_high_against_history"

    return {
        "metric": metric_name,
        "status": status,
        "reason": reason,
        "new_value": float(new_value),
        "historical_mean": float(hist_mean),
        "historical_std": float(hist_std),
        "threshold_approved": float(approved_threshold),
        "threshold_warning": float(warning_threshold),
    }


def combine_status(metric_results: list[dict]) -> str:
    """
    Combina resultados individuales en una decisión final.
    Regla simple:
    - Si alguna métrica está rejected -> rejected
    - Si alguna métrica está warning -> warning
    - Si todas están approved -> approved
    """
    statuses = [result["status"] for result in metric_results]

    if "rejected" in statuses:
        return "rejected"

    if "warning" in statuses:
        return "warning"

    return "approved"


def validate_new_run(
    historical_df: pd.DataFrame,
    new_run: dict,
    config: dict,
) -> dict:
    validation_mode = new_run.get("validation_mode", config["default_mode"])

    if pd.isna(validation_mode):
        validation_mode = config["default_mode"]

    if validation_mode not in config["validation_modes"]:
        raise ValueError(
            f"Modo de validación no soportado: {validation_mode}. "
            f"Opciones válidas: {list(config['validation_modes'].keys())}"
        )

    std_factor = config["validation_modes"][validation_mode]["std_factor"]
    min_history_runs = config["min_history_runs"]

    comparable_history = filter_comparable_history(historical_df, new_run)

    validation_notes = []

    if len(comparable_history) < min_history_runs:
        validation_notes.append(
            "No hay suficiente historial con el mismo algorithm, scenario y reward_version."
        )

        final_status = "warning"

        return {
            "run_id": new_run.get("run_id"),
            "algorithm": new_run.get("algorithm"),
            "scenario": new_run.get("scenario"),
            "reward_version": new_run.get("reward_version"),
            "validation_mode": validation_mode,
            "final_status": final_status,
            "history_rows_used": int(len(comparable_history)),
            "min_history_runs": int(min_history_runs),
            "preliminary_validation": True,
            "metric_results": [],
            "validation_notes": validation_notes,
            "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    metric_results = []

    for metric_name, metric_config in config["metrics"].items():
        if metric_name not in comparable_history.columns:
            if metric_config.get("required", False):
                raise ValueError(f"Métrica obligatoria no encontrada: {metric_name}")
            continue

        if metric_name not in new_run:
            if metric_config.get("required", False):
                raise ValueError(f"La nueva corrida no tiene la métrica: {metric_name}")
            continue

        historical_values = comparable_history[metric_name]
        new_value = pd.to_numeric(new_run[metric_name], errors="coerce")

        result = compute_metric_status(
            metric_name=metric_name,
            new_value=new_value,
            historical_values=historical_values,
            higher_is_better=metric_config["higher_is_better"],
            std_factor=std_factor,
        )

        metric_results.append(result)

    final_status = combine_status(metric_results)

    return {
        "run_id": new_run.get("run_id"),
        "algorithm": new_run.get("algorithm"),
        "scenario": new_run.get("scenario"),
        "reward_version": new_run.get("reward_version"),
        "validation_mode": validation_mode,
        "final_status": final_status,
        "history_rows_used": int(len(comparable_history)),
        "min_history_runs": int(min_history_runs),
        "preliminary_validation": False,
        "metric_results": metric_results,
        "validation_notes": validation_notes,
        "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_validation_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def print_validation_summary(report: dict):
    print("\nResultado de validación")
    print("-" * 40)
    print(f"run_id: {report['run_id']}")
    print(f"algorithm: {report['algorithm']}")
    print(f"scenario: {report['scenario']}")
    print(f"reward_version: {report['reward_version']}")
    print(f"validation_mode: {report['validation_mode']}")
    print(f"históricos usados: {report['history_rows_used']}")
    print(f"estado final: {report['final_status'].upper()}")

    if report["preliminary_validation"]:
        print("\nValidación preliminar:")
        for note in report["validation_notes"]:
            print(f"- {note}")

    if report["metric_results"]:
        print("\nDetalle por métrica:")

        for result in report["metric_results"]:
            print(
                f"- {result['metric']}: {result['status']} "
                f"| nuevo={result['new_value']:.4f} "
                f"| media_hist={result['historical_mean']:.4f} "
                f"| std_hist={result['historical_std']:.4f}"
            )


def main():
    config = load_yaml(VALIDATION_CONFIG_PATH)
    historical_df = load_historical_data(HISTORICAL_DATA_PATH)
    new_run = load_new_run(NEW_RUN_PATH)

    report = validate_new_run(
        historical_df=historical_df,
        new_run=new_run,
        config=config,
    )

    save_validation_report(report, VALIDATION_REPORT_PATH)
    print_validation_summary(report)

    print(f"\nReporte guardado en: {VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
