"""
test_data.py

Quality gate de datos: valida que el histórico procesado
(data/processed/historical_results_clean.csv) cumple el esquema
declarado en configs/data_schema.yaml, y que clean_data.py normaliza
correctamente entradas "sucias" (alias de columnas, texto con
mayúsculas/espacios, success_rate en escala de porcentaje, etc.).

Estos tests no entrenan nada ni dependen de archivos generados por una
corrida real: construyen DataFrames pequeños en memoria, así corren
rápido y de forma determinista en cualquier máquina.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.clean_data import (
    clean_historical_data,
    normalize_algorithm,
    normalize_scenario,
    normalize_reward_version,
    normalize_success_rate,
    STANDARD_COLUMNS,
)

DATA_SCHEMA_PATH = PROJECT_ROOT / "configs" / "data_schema.yaml"


@pytest.fixture(scope="module")
def data_schema() -> dict:
    with open(DATA_SCHEMA_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["standard_columns"]


@pytest.fixture
def raw_messy_df() -> pd.DataFrame:
    """
    Simula un histórico 'sucio' tal como podría llegar de un Excel
    externo: nombres de columna con alias, texto con mayúsculas y
    guiones, success_rate en escala 0-100 en vez de 0-1.
    """
    return pd.DataFrame(
        {
            "ID": ["run_a", "run_b"],
            "Algo": ["Q-Learning", "DQN"],
            "Obstacle": ["Wall", "L-Shape"],
            "reward_version": ["reward_v1_base", "reward_v2_step_penalty"],
            "reward": [180.5, 165.0],
            "Success_Percent": [88.0, 91.5],
            "first_goal_step": [45, 50],
            "wall_hits": [2, 3],
            "steps": [60, 65],
            "training_time": [110, 300],
            "num_episodes": [500, 500],
        }
    )


class TestSchemaCompliance:
    """El histórico limpio debe cumplir exactamente configs/data_schema.yaml."""

    def test_clean_output_has_all_standard_columns(self, raw_messy_df, data_schema):
        clean_df = clean_historical_data(raw_messy_df)

        for column in data_schema:
            assert column in clean_df.columns, f"Falta la columna estándar: {column}"

    def test_required_columns_have_no_nulls(self, raw_messy_df, data_schema):
        clean_df = clean_historical_data(raw_messy_df)

        for column, spec in data_schema.items():
            if spec.get("required", False):
                assert clean_df[column].notna().all(), (
                    f"La columna requerida '{column}' tiene valores nulos "
                    "después de la limpieza."
                )

    def test_allowed_values_are_respected(self, raw_messy_df, data_schema):
        clean_df = clean_historical_data(raw_messy_df)

        for column, spec in data_schema.items():
            allowed = spec.get("allowed_values")
            if allowed is None or column not in clean_df.columns:
                continue

            actual_values = set(clean_df[column].dropna().unique())
            assert actual_values.issubset(set(allowed)), (
                f"Valores fuera de lo permitido en '{column}': "
                f"{actual_values - set(allowed)}"
            )

    def test_success_rate_within_declared_range(self, raw_messy_df, data_schema):
        clean_df = clean_historical_data(raw_messy_df)
        value_range = data_schema["success_rate"]["range"]

        assert clean_df["success_rate"].between(
            value_range["min"], value_range["max"]
        ).all(), "success_rate fuera del rango declarado en data_schema.yaml"


class TestColumnNormalization:
    """Aliases y formatos sucios deben mapearse a los nombres/valores estándar."""

    def test_column_aliases_are_renamed(self, raw_messy_df):
        clean_df = clean_historical_data(raw_messy_df)

        assert "run_id" in clean_df.columns
        assert "algorithm" in clean_df.columns
        assert "scenario" in clean_df.columns
        assert list(clean_df.columns) == STANDARD_COLUMNS

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("Q-Learning", "q_learning"),
            ("q learning", "q_learning"),
            ("QL", "q_learning"),
            ("DQN", "dqn"),
            ("Discrete SAC", "sac"),
        ],
    )
    def test_algorithm_aliases_normalize_correctly(self, raw_value, expected):
        assert normalize_algorithm(raw_value) == expected

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("Wall", "wall"),
            ("L-Shape", "l_shape"),
            ("u shape", "u_shape"),
        ],
    )
    def test_scenario_aliases_normalize_correctly(self, raw_value, expected):
        assert normalize_scenario(raw_value) == expected

    def test_missing_reward_version_defaults_to_base(self):
        assert normalize_reward_version(None) == "reward_v1_base"

    def test_success_rate_percentage_is_rescaled_to_0_1(self):
        percentages = pd.Series([88.0, 91.5, 100.0])
        rescaled = normalize_success_rate(percentages)

        assert rescaled.between(0.0, 1.0).all()
        assert rescaled.iloc[0] == pytest.approx(0.88)

    def test_success_rate_already_in_0_1_is_not_rescaled(self):
        fractions = pd.Series([0.88, 0.915, 1.0])
        rescaled = normalize_success_rate(fractions)

        assert rescaled.iloc[0] == pytest.approx(0.88)


class TestMissingDataHandling:
    """clean_data.py no debe romperse con columnas faltantes, debe avisar con claridad."""

    def test_missing_required_column_raises_clear_error(self):
        incomplete_df = pd.DataFrame(
            {
                "run_id": ["run_a"],
                "algorithm": ["q_learning"],
                "scenario": ["wall"],
                # falta success_rate y avg_reward, ambas requeridas
            }
        )

        with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
            clean_historical_data(incomplete_df)

    def test_missing_run_id_is_auto_generated(self):
        df_without_id = pd.DataFrame(
            {
                "algorithm": ["q_learning"],
                "scenario": ["wall"],
                "reward_version": ["reward_v1_base"],
                "success_rate": [0.9],
                "avg_reward": [150.0],
            }
        )

        clean_df = clean_historical_data(df_without_id)

        assert clean_df["run_id"].iloc[0].startswith("hist_")
