from pathlib import Path
from datetime import datetime
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "historical_loaded.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "historical_results_clean.csv"


REQUIRED_COLUMNS = [
    "run_id",
    "algorithm",
    "scenario",
    "reward_version",
    "success_rate",
    "avg_reward",
]


STANDARD_COLUMNS = [
    "run_id",
    "algorithm",
    "scenario",
    "reward_version",
    "success_rate",
    "avg_reward",
    "first_reach_step",
    "collisions",
    "avg_steps",
    "training_time_sec",
    "episodes",
    "seed",
    "source_file",
    "created_at",
    "notes",
]


COLUMN_ALIASES = {
    # Identificadores
    "id": "run_id",
    "run": "run_id",
    "experiment_id": "run_id",

    # Algoritmo
    "algo": "algorithm",
    "algoritmo": "algorithm",
    "agent": "algorithm",

    # Escenario
    "escenario": "scenario",
    "obstacle": "scenario",
    "obstacle_type": "scenario",

    # Reward
    "reward": "avg_reward",
    "mean_reward": "avg_reward",
    "average_reward": "avg_reward",

    # Success rate
    "success": "success_rate",
    "success_percent": "success_rate",
    "success_percentage": "success_rate",
    "successrate": "success_rate",

    # Primer paso de llegada
    "first_goal_step": "first_reach_step",
    "first_reach": "first_reach_step",
    "goal_step": "first_reach_step",

    # Colisiones
    "wall_hits": "collisions",
    "obstacle_hits": "collisions",
    "collision": "collisions",

    # Pasos
    "steps": "avg_steps",
    "mean_steps": "avg_steps",
    "average_steps": "avg_steps",

    # Tiempo
    "training_time": "training_time_sec",
    "time": "training_time_sec",
    "train_time": "training_time_sec",

    # Episodios
    "num_episodes": "episodes",
    "n_episodes": "episodes",
}


ALGORITHM_MAP = {
    "q-learning": "q_learning",
    "q learning": "q_learning",
    "qlearning": "q_learning",
    "ql": "q_learning",
    "q_learning": "q_learning",

    "dqn": "dqn",
    "deep q network": "dqn",
    "deep q-network": "dqn",

    "sac": "sac",
    "discrete sac": "sac",
    "discret sac": "sac",
}


SCENARIO_MAP = {
    "wall": "wall",
    "pared": "wall",
    "muro": "wall",

    "l": "l_shape",
    "l-shape": "l_shape",
    "l_shape": "l_shape",
    "l shape": "l_shape",

    "u": "u_shape",
    "u-shape": "u_shape",
    "u_shape": "u_shape",
    "u shape": "u_shape",
}


REWARD_VERSION_MAP = {
    "base": "reward_v1_base",
    "reward_v1": "reward_v1_base",
    "reward_v1_base": "reward_v1_base",

    "step_penalty": "reward_v2_step_penalty",
    "reward_v2": "reward_v2_step_penalty",
    "reward_v2_step_penalty": "reward_v2_step_penalty",

    "collision_penalty": "reward_v3_collision_penalty",
    "reward_v3": "reward_v3_collision_penalty",
    "reward_v3_collision_penalty": "reward_v3_collision_penalty",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas:
    - minúsculas
    - espacios por _
    - aplica alias conocidos
    """
    df = df.copy()

    normalized_columns = {}

    for col in df.columns:
        clean_col = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        clean_col = COLUMN_ALIASES.get(clean_col, clean_col)
        normalized_columns[col] = clean_col

    df = df.rename(columns=normalized_columns)
    return df


def normalize_text_value(value):
    """
    Limpia textos para poder mapear valores.
    """
    if pd.isna(value):
        return value

    return str(value).strip().lower().replace("_", " ")


def normalize_algorithm(value):
    value_clean = normalize_text_value(value)

    if pd.isna(value_clean):
        return value_clean

    return ALGORITHM_MAP.get(value_clean, str(value_clean).replace(" ", "_"))


def normalize_scenario(value):
    value_clean = normalize_text_value(value)

    if pd.isna(value_clean):
        return value_clean

    return SCENARIO_MAP.get(value_clean, str(value_clean).replace(" ", "_"))


def normalize_reward_version(value):
    value_clean = normalize_text_value(value)

    if pd.isna(value_clean):
        return "reward_v1_base"

    return REWARD_VERSION_MAP.get(value_clean, str(value).strip())


def normalize_success_rate(series: pd.Series) -> pd.Series:
    """
    Convierte success_rate a escala 0-1.
    Si detecta valores mayores que 1, asume que están en porcentaje.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.max() > 1:
        numeric = numeric / 100.0

    return numeric


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas estándar faltantes como valores vacíos.
    """
    df = df.copy()

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def generate_missing_run_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si falta run_id, genera uno automáticamente.
    """
    df = df.copy()

    if "run_id" not in df.columns:
        df["run_id"] = pd.NA

    for idx in df.index:
        if pd.isna(df.loc[idx, "run_id"]) or str(df.loc[idx, "run_id"]).strip() == "":
            df.loc[idx, "run_id"] = f"hist_{idx + 1:04d}"

    return df


def validate_required_columns(df: pd.DataFrame):
    """
    Verifica que existan las columnas mínimas requeridas.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias después de limpiar nombres: {missing}"
        )


def clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza los datos históricos.
    """
    df = normalize_column_names(df)
    df = generate_missing_run_ids(df)

    validate_required_columns(df)

    df = add_missing_columns(df)

    df["algorithm"] = df["algorithm"].apply(normalize_algorithm)
    df["scenario"] = df["scenario"].apply(normalize_scenario)
    df["reward_version"] = df["reward_version"].apply(normalize_reward_version)

    df["success_rate"] = normalize_success_rate(df["success_rate"])

    numeric_columns = [
        "avg_reward",
        "first_reach_step",
        "collisions",
        "avg_steps",
        "training_time_sec",
        "episodes",
        "seed",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["created_at"] = df["created_at"].fillna(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    df = df[STANDARD_COLUMNS]

    return df


def main():
    if not INTERIM_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe el archivo intermedio: {INTERIM_DATA_PATH}. "
            "Primero ejecuta: python src/load_data.py"
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INTERIM_DATA_PATH)
    clean_df = clean_historical_data(df)

    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Limpieza completada.")
    print(f"Filas procesadas: {clean_df.shape[0]}")
    print(f"Columnas finales: {clean_df.shape[1]}")
    print(f"Archivo generado: {PROCESSED_DATA_PATH}")

    print("\nColumnas finales:")
    print(list(clean_df.columns))

    print("\nPrimeras filas limpias:")
    print(clean_df.head())


if __name__ == "__main__":
    main()
