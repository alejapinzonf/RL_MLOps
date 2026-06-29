from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"
NEW_RUN_PATH = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "validation_result.json"
EVALUATION_REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"
EVALUATION_REPORT_PATH = EVALUATION_REPORT_DIR / "evaluation_result.json"

# Mínimo de episodios en cada tramo (inicial/final) para que la
# comparación tenga algún sentido estadístico mínimo.
MIN_WINDOW_SIZE = 5

# Proporción de episodios usada como tramo inicial y como tramo final.
WINDOW_FRACTION = 0.2

# Mejora mínima (en proporción, no absoluta) para considerar que una
# métrica mejoró "de verdad" y no solo por ruido.
MIN_RELATIVE_IMPROVEMENT = 0.05

# Combinación final: training_status x historical_validation_status -> final_recommendation
RECOMMENDATION_TABLE = {
    ("converged", "approved"): "approved",
    ("converged", "warning"): "approved",
    ("converged", "rejected"): "warning",
    ("partially_converged", "approved"): "approved",
    ("partially_converged", "warning"): "warning",
    ("partially_converged", "rejected"): "warning",
    ("not_converged", "approved"): "warning",
    ("not_converged", "warning"): "rejected",
    ("not_converged", "rejected"): "rejected",
}


def load_episode_metrics(run_id: str) -> pd.DataFrame:
    """
    Carga el CSV de métricas por episodio de una corrida.

    Soporta tanto el esquema de demo_train.py (columna 'success_rate'
    ya promediada por episodio) como el de train_q_learning.py
    (columna 'success' binaria por episodio).
    """
    path = TRAINING_REPORTS_DIR / f"{run_id}_episode_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de métricas por episodio: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"El archivo de métricas está vacío: {path}")

    if "success_rate" not in df.columns and "success" in df.columns:
        # train_q_learning.py guarda éxito binario por episodio (0/1).
        # Se deja tal cual; el promedio por tramo ya produce una tasa.
        df = df.rename(columns={"success": "success_rate"})

    required_columns = ["episode_reward", "success_rate"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias en {path.name} para evaluar "
            f"convergencia: {missing}"
        )

    return df.sort_values("episode").reset_index(drop=True)


def split_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide la corrida en un tramo inicial y un tramo final.

    El tamaño de cada tramo es WINDOW_FRACTION del total de episodios,
    con un mínimo de MIN_WINDOW_SIZE episodios. Si la corrida es muy
    corta, los tramos pueden solaparse; esto se reporta como nota.
    """
    n_episodes = len(df)
    window_size = max(MIN_WINDOW_SIZE, int(round(n_episodes * WINDOW_FRACTION)))
    window_size = min(window_size, n_episodes)

    initial_window = df.iloc[:window_size]
    final_window = df.iloc[-window_size:]

    return initial_window, final_window


def relative_improvement(initial_value: float, final_value: float, higher_is_better: bool) -> float:
    """
    Calcula la mejora relativa entre el tramo inicial y el final.

    Devuelve un valor positivo si hubo mejora (en el sentido indicado
    por higher_is_better) y negativo si empeoró. Se normaliza por el
    valor absoluto del tramo inicial para poder compararlo contra
    MIN_RELATIVE_IMPROVEMENT independientemente de la escala de la métrica.
    """
    denominator = abs(initial_value) if abs(initial_value) > 1e-9 else 1e-9

    raw_delta = final_value - initial_value

    if not higher_is_better:
        raw_delta = -raw_delta

    return raw_delta / denominator


def evaluate_convergence(df: pd.DataFrame) -> dict:
    """
    Evalúa si el agente convergió durante el entrenamiento,
    comparando el tramo inicial contra el tramo final de episodios.

    Esta evaluación es independiente del histórico: solo mira si la
    corrida mejoró internamente, sin compararse contra otras corridas.
    """
    initial_window, final_window = split_windows(df)

    initial_avg_reward = float(initial_window["episode_reward"].mean())
    final_avg_reward = float(final_window["episode_reward"].mean())

    initial_success_rate = float(initial_window["success_rate"].mean())
    final_success_rate = float(final_window["success_rate"].mean())

    reward_improvement = relative_improvement(
        initial_avg_reward, final_avg_reward, higher_is_better=True
    )
    success_improvement = relative_improvement(
        initial_success_rate, final_success_rate, higher_is_better=True
    )

    reward_improved = reward_improvement >= MIN_RELATIVE_IMPROVEMENT
    success_improved = success_improvement >= MIN_RELATIVE_IMPROVEMENT

    if reward_improved and success_improved:
        training_status = "converged"
    elif reward_improved or success_improved:
        training_status = "partially_converged"
    else:
        training_status = "not_converged"

    notes = []
    window_size = len(initial_window)
    n_episodes = len(df)

    if window_size * 2 > n_episodes:
        notes.append(
            f"La corrida tiene pocos episodios ({n_episodes}); los tramos "
            f"inicial y final se solapan parcialmente (tamaño de tramo={window_size})."
        )

    return {
        "training_status": training_status,
        "n_episodes": int(n_episodes),
        "window_size": int(window_size),
        "initial_avg_reward": round(initial_avg_reward, 4),
        "final_avg_reward": round(final_avg_reward, 4),
        "reward_improvement_pct": round(reward_improvement * 100, 2),
        "initial_success_rate": round(initial_success_rate, 4),
        "final_success_rate": round(final_success_rate, 4),
        "success_improvement_pct": round(success_improvement * 100, 2),
        "evaluation_notes": notes,
    }


def combine_recommendation(
    training_status: str,
    historical_validation_status: str,
) -> str:
    """
    Combina el estado de convergencia con el estado de validación
    histórica en una recomendación final.

    Esto resuelve el problema conceptual del proyecto: una corrida
    puede converger durante el entrenamiento pero ser rechazada por el
    histórico (o viceversa). La recomendación final distingue estos
    casos en lugar de colapsarlos en un solo "rejected".
    """
    key = (training_status, historical_validation_status)

    if key not in RECOMMENDATION_TABLE:
        raise ValueError(
            f"Combinación no soportada: training_status={training_status}, "
            f"historical_validation_status={historical_validation_status}"
        )

    return RECOMMENDATION_TABLE[key]


def load_validation_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el reporte de validación histórica: {path}. "
            "Primero ejecuta validate_run.py."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_new_run(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe la corrida nueva: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("El archivo de nueva corrida está vacío.")

    return df.iloc[0].to_dict()


def evaluate_run(run_id: str, historical_validation_status: str) -> dict:
    episode_metrics = load_episode_metrics(run_id)
    convergence_result = evaluate_convergence(episode_metrics)

    final_recommendation = combine_recommendation(
        training_status=convergence_result["training_status"],
        historical_validation_status=historical_validation_status,
    )

    return {
        "run_id": run_id,
        "training_status": convergence_result["training_status"],
        "historical_validation_status": historical_validation_status,
        "final_recommendation": final_recommendation,
        "convergence_details": convergence_result,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_evaluation_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def print_evaluation_summary(report: dict):
    details = report["convergence_details"]

    print("\nResultado de evaluación de convergencia")
    print("-" * 40)
    print(f"run_id: {report['run_id']}")
    print(f"episodios analizados: {details['n_episodes']} (tramo={details['window_size']})")
    print(
        f"reward: {details['initial_avg_reward']:.4f} -> "
        f"{details['final_avg_reward']:.4f} "
        f"({details['reward_improvement_pct']:+.2f}%)"
    )
    print(
        f"success_rate: {details['initial_success_rate']:.4f} -> "
        f"{details['final_success_rate']:.4f} "
        f"({details['success_improvement_pct']:+.2f}%)"
    )

    print(f"\ntraining_status: {report['training_status'].upper()}")
    print(f"historical_validation_status: {report['historical_validation_status'].upper()}")
    print(f"final_recommendation: {report['final_recommendation'].upper()}")

    if details["evaluation_notes"]:
        print("\nNotas:")
        for note in details["evaluation_notes"]:
            print(f"- {note}")


def main():
    new_run = load_new_run(NEW_RUN_PATH)
    validation_report = load_validation_report(VALIDATION_REPORT_PATH)

    run_id = str(new_run["run_id"])
    historical_validation_status = validation_report.get("final_status", "warning")

    report = evaluate_run(
        run_id=run_id,
        historical_validation_status=historical_validation_status,
    )

    save_evaluation_report(report, EVALUATION_REPORT_PATH)
    print_evaluation_summary(report)

    print(f"\nReporte guardado en: {EVALUATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
