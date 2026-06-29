from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import openpyxl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpeta donde se colocan los .xlsx originales del paper, ya renombrados
# con el patrón: experimento<N>_<algoritmo>_<obstaculo>.xlsx
PAPER_EXCELS_DIR = PROJECT_ROOT / "data" / "raw" / "paper_excels"

# Salida 1: histórico resumen, en el mismo formato que espera clean_data.py
PAPER_HISTORICAL_OUTPUT = PROJECT_ROOT / "data" / "raw" / "historical_results_paper.csv"

# Salida 2: episodios completos por corrida, listos para evaluate_run.py /
# generate_run_report.py (mismo formato que generan demo_train.py /
# train_q_learning.py).
TRAINING_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"

# Proporción final de episodios usada para calcular el resumen "ya entrenado"
# de cada corrida. Se evita promediar desde el episodio 1 (epsilon=1.0,
# agente sin entrenar todavía), que distorsionaría el histórico hacia abajo.
SUMMARY_WINDOW_FRACTION = 0.10
MIN_SUMMARY_WINDOW = 50

ALGORITHM_ALIASES = {
    "ql": "q_learning",
    "qlearning": "q_learning",
    "q_learning": "q_learning",
    "qtable": "q_learning",
    "dqn": "dqn",
    "sac": "sac",
    "discretesac": "sac",
    "discrete_sac": "sac",
}

SCENARIO_ALIASES = {
    "wall": "wall",
    "muro": "wall",
    "l": "l_shape",
    "lshape": "l_shape",
    "l_shape": "l_shape",
    "u": "u_shape",
    "ushape": "u_shape",
    "u_shape": "u_shape",
}

# Columnas tal como aparecen en los excels reales (episodes sheet) ->
# nombres estándar usados en el resto del pipeline (data_schema.yaml).
EPISODE_COLUMN_ALIASES = {
    "reward": "episode_reward",
    "success": "success",  # se mantiene; evaluate_run.py ya soporta este nombre
    "wall_hits": "collisions",
    "steps": "steps",
}

EXPERIMENT_NAME_PATTERN = re.compile(r"^experim?ento?\d*$")


def parse_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Extrae (algorithm, scenario) del nombre de archivo.

    Espera el patrón: experimento<N>_<algoritmo>_<obstaculo>.xlsx
    (insensible a mayúsculas, tolera 'experiment'/'experimento'/'exp').
    Devuelve (None, None) en las partes que no pueda identificar.
    """
    name = Path(filename).stem
    tokens = re.split(r"[_\s]+", name.lower())
    tokens = [t for t in tokens if not EXPERIMENT_NAME_PATTERN.match(t)]

    algorithm = None
    scenario = None

    for token in tokens:
        if algorithm is None and token in ALGORITHM_ALIASES:
            algorithm = ALGORITHM_ALIASES[token]
        elif scenario is None and token in SCENARIO_ALIASES:
            scenario = SCENARIO_ALIASES[token]

    return algorithm, scenario


def find_excel_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de excels del paper: {directory}. "
            "Crea la carpeta y coloca ahí los .xlsx renombrados como "
            "experimento<N>_<algoritmo>_<obstaculo>.xlsx"
        )

    return sorted(directory.glob("*.xlsx"))


def load_episodes_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="episodes", engine="openpyxl")

    if df.empty:
        raise ValueError(f"La hoja 'episodes' está vacía en: {path.name}")

    df = df.rename(columns=EPISODE_COLUMN_ALIASES)
    return df


def compute_summary_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona el tramo final de episodios (último SUMMARY_WINDOW_FRACTION)
    para resumir el desempeño del agente ya entrenado, evitando que el
    arranque con epsilon=1.0 distorsione el promedio hacia abajo.
    """
    n_episodes = len(df)
    window_size = max(MIN_SUMMARY_WINDOW, int(round(n_episodes * SUMMARY_WINDOW_FRACTION)))
    window_size = min(window_size, n_episodes)

    return df.iloc[-window_size:]


def build_historical_row(
    run_id: str,
    algorithm: str,
    scenario: str,
    df: pd.DataFrame,
    source_file: str,
) -> dict:
    window = compute_summary_window(df)

    success_col = "success" if "success" in window.columns else "success_rate"

    row = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": "paper_reward_v1",
        "success_rate": round(float(window[success_col].mean()), 4),
        "avg_reward": round(float(window["episode_reward"].mean()), 4),
        "first_reach_step": round(float(window["first_reach_step"].mean()), 4)
        if "first_reach_step" in window.columns
        else None,
        "collisions": round(float(window["collisions"].mean()), 4)
        if "collisions" in window.columns
        else None,
        "avg_steps": round(float(window["steps"].mean()), 4)
        if "steps" in window.columns
        else None,
        "training_time_sec": round(float(window["episode_time_seconds"].sum()), 4)
        if "episode_time_seconds" in window.columns
        else None,
        "episodes": int(len(df)),
        "seed": None,
        "source_file": source_file,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            f"real paper training run; summary computed over last "
            f"{len(window)}/{len(df)} episodes (post-exploration window)"
        ),
    }

    return row


def save_episode_metrics(run_id: str, df: pd.DataFrame) -> Path:
    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TRAINING_REPORTS_DIR / f"{run_id}_episode_metrics.csv"
    df.to_csv(output_path, index=False)

    return output_path


def import_excel_file(path: Path, index: int, total: int) -> dict | None:
    algorithm, scenario = parse_filename(path.name)

    if algorithm is None or scenario is None:
        print(
            f"[{index}/{total}] OMITIDO: no se pudo identificar algorithm/scenario "
            f"en el nombre '{path.name}'. Esperado: Experiment<N>_<algoritmo>_<obstaculo>.xlsx"
        )
        return None

    print(f"[{index}/{total}] Procesando {path.name} -> algorithm={algorithm}, scenario={scenario}")

    df = load_episodes_sheet(path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"paper_{algorithm}_{scenario}_{path.stem}_{timestamp}"

    episode_metrics_path = save_episode_metrics(run_id, df)
    print(f"    episodios guardados: {episode_metrics_path}")

    historical_row = build_historical_row(
        run_id=run_id,
        algorithm=algorithm,
        scenario=scenario,
        df=df,
        source_file=path.name,
    )

    return historical_row


def main():
    excel_files = find_excel_files(PAPER_EXCELS_DIR)

    if not excel_files:
        print(f"No se encontraron archivos .xlsx en: {PAPER_EXCELS_DIR}")
        return

    historical_rows = []
    total = len(excel_files)

    for index, path in enumerate(excel_files, start=1):
        try:
            row = import_excel_file(path, index, total)
            if row is not None:
                historical_rows.append(row)
        except Exception as error:
            print(f"[{index}/{total}] ERROR procesando {path.name}: {error}")

    if not historical_rows:
        print("\nNo se generó ninguna fila histórica (revisa los nombres de archivo).")
        return

    historical_df = pd.DataFrame(historical_rows)

    PAPER_HISTORICAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    historical_df.to_csv(PAPER_HISTORICAL_OUTPUT, index=False)

    print(f"\nImportación completada: {len(historical_rows)}/{total} archivos procesados.")
    print(f"Histórico guardado en: {PAPER_HISTORICAL_OUTPUT}")
    print("\nResumen por corrida:")
    print(historical_df[["run_id", "algorithm", "scenario", "success_rate", "avg_reward", "episodes"]])


if __name__ == "__main__":
    main()