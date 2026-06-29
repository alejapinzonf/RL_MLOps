"""
test_business_logic.py

Quality gate de la lógica de decisión central del pipeline: las
funciones puras de validate_run.py (comparación estadística contra
histórico) y evaluate_run.py (análisis de convergencia y combinación
final). Estas son las funciones que determinan si una corrida se
aprueba, se rechaza, o queda en warning — son el corazón del proyecto,
así que se testean con casos conocidos y construidos a mano.

No requieren archivos en disco: trabajan con pd.Series/DataFrame
construidos directamente en el test.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation.validate_run import compute_metric_status, combine_status
from evaluation.evaluate_run import (
    evaluate_convergence,
    combine_recommendation,
    relative_improvement,
    split_windows,
)


class TestComputeMetricStatus:
    """compute_metric_status compara un valor nuevo contra mean/std histórico."""

    def test_value_within_range_is_approved(self):
        historical = pd.Series([0.85, 0.87, 0.86, 0.88, 0.86])

        result = compute_metric_status(
            metric_name="success_rate",
            new_value=0.87,
            historical_values=historical,
            higher_is_better=True,
            std_factor=1.0,
        )

        assert result["status"] == "approved"

    def test_value_far_below_range_is_rejected_when_higher_is_better(self):
        historical = pd.Series([0.85, 0.87, 0.86, 0.88, 0.86])

        result = compute_metric_status(
            metric_name="success_rate",
            new_value=0.10,
            historical_values=historical,
            higher_is_better=True,
            std_factor=1.0,
        )

        assert result["status"] == "rejected"

    def test_value_far_above_range_is_rejected_when_lower_is_better(self):
        # collisions: menos es mejor, un valor muy alto debe rechazarse.
        historical = pd.Series([2.0, 2.5, 2.2, 2.4, 2.1])

        result = compute_metric_status(
            metric_name="collisions",
            new_value=50.0,
            historical_values=historical,
            higher_is_better=False,
            std_factor=1.0,
        )

        assert result["status"] == "rejected"

    def test_value_far_below_range_is_approved_when_lower_is_better(self):
        # collisions: un valor MUY BAJO es algo bueno, debe aprobarse,
        # no rechazarse — confirma que la dirección del check es correcta.
        historical = pd.Series([2.0, 2.5, 2.2, 2.4, 2.1])

        result = compute_metric_status(
            metric_name="collisions",
            new_value=0.1,
            historical_values=historical,
            higher_is_better=False,
            std_factor=1.0,
        )

        assert result["status"] == "approved"

    def test_zero_std_does_not_crash(self):
        # Histórico sin variación (todas las corridas dieron igual):
        # std=0 debe manejarse sin ZeroDivisionError.
        historical = pd.Series([0.9, 0.9, 0.9])

        result = compute_metric_status(
            metric_name="success_rate",
            new_value=0.9,
            historical_values=historical,
            higher_is_better=True,
            std_factor=1.0,
        )

        assert result["status"] == "approved"

    def test_missing_new_value_is_warning_not_crash(self):
        historical = pd.Series([0.85, 0.87, 0.86])

        result = compute_metric_status(
            metric_name="success_rate",
            new_value=float("nan"),
            historical_values=historical,
            higher_is_better=True,
            std_factor=1.0,
        )

        assert result["status"] == "warning"
        assert result["reason"] == "new_value_missing"

    def test_stricter_std_factor_rejects_more_easily(self):
        historical = pd.Series([0.80, 0.82, 0.81, 0.83, 0.80])
        borderline_value = 0.76

        strict_result = compute_metric_status(
            metric_name="success_rate", new_value=borderline_value,
            historical_values=historical, higher_is_better=True, std_factor=0.5,
        )
        flexible_result = compute_metric_status(
            metric_name="success_rate", new_value=borderline_value,
            historical_values=historical, higher_is_better=True, std_factor=1.5,
        )

        # El modo estricto no puede ser más permisivo que el flexible.
        status_order = {"rejected": 0, "warning": 1, "approved": 2}
        assert status_order[strict_result["status"]] <= status_order[flexible_result["status"]]


class TestCombineStatus:
    """combine_status colapsa varios resultados de métricas en un único estado final."""

    def test_all_approved_yields_approved(self):
        results = [{"status": "approved"}, {"status": "approved"}]
        assert combine_status(results) == "approved"

    def test_any_rejected_yields_rejected(self):
        results = [{"status": "approved"}, {"status": "rejected"}, {"status": "warning"}]
        assert combine_status(results) == "rejected"

    def test_warning_without_rejected_yields_warning(self):
        results = [{"status": "approved"}, {"status": "warning"}]
        assert combine_status(results) == "warning"

    def test_empty_results_yields_approved(self):
        assert combine_status([]) == "approved"


class TestRelativeImprovement:
    """relative_improvement debe ser positivo cuando hay mejora real, negativo si empeora."""

    def test_positive_improvement_when_higher_is_better(self):
        result = relative_improvement(initial_value=10.0, final_value=20.0, higher_is_better=True)
        assert result > 0

    def test_negative_improvement_when_higher_is_better_and_value_drops(self):
        result = relative_improvement(initial_value=20.0, final_value=10.0, higher_is_better=True)
        assert result < 0

    def test_positive_improvement_when_lower_is_better_and_value_drops(self):
        # collisions: bajar de 10 a 5 es una mejora, debe ser positivo.
        result = relative_improvement(initial_value=10.0, final_value=5.0, higher_is_better=False)
        assert result > 0

    def test_zero_initial_value_does_not_crash(self):
        result = relative_improvement(initial_value=0.0, final_value=5.0, higher_is_better=True)
        assert result > 0


class TestSplitWindows:
    """split_windows debe dividir en tramo inicial/final con tamaño consistente."""

    def test_window_size_respects_minimum(self):
        df = pd.DataFrame({"episode_reward": range(8), "success_rate": [0.1] * 8})
        initial, final = split_windows(df)

        assert len(initial) == 5  # MIN_WINDOW_SIZE
        assert len(final) == 5

    def test_window_size_scales_with_episode_count(self):
        df = pd.DataFrame({"episode_reward": range(1000), "success_rate": [0.1] * 1000})
        initial, final = split_windows(df)

        assert len(initial) == 200  # 20% de 1000
        assert len(final) == 200

    def test_initial_and_final_are_correct_ends(self):
        df = pd.DataFrame({"episode_reward": range(100), "success_rate": [0.1] * 100})
        initial, final = split_windows(df)

        assert initial["episode_reward"].iloc[0] == 0
        assert final["episode_reward"].iloc[-1] == 99


class TestEvaluateConvergence:
    """evaluate_convergence debe clasificar correctamente casos conocidos."""

    def _make_df(self, n_episodes: int, reward_fn, success_fn) -> pd.DataFrame:
        episodes = list(range(1, n_episodes + 1))
        return pd.DataFrame(
            {
                "episode_reward": [reward_fn(e, n_episodes) for e in episodes],
                "success_rate": [success_fn(e, n_episodes) for e in episodes],
            }
        )

    def test_clear_improvement_is_converged(self):
        df = self._make_df(
            n_episodes=200,
            reward_fn=lambda e, n: -100 + 200 * (e / n),
            success_fn=lambda e, n: 0.05 + 0.9 * (e / n),
        )

        result = evaluate_convergence(df)

        assert result["training_status"] == "converged"
        assert result["reward_improvement_pct"] > 0
        assert result["success_improvement_pct"] > 0

    def test_flat_curve_is_not_converged(self):
        df = self._make_df(
            n_episodes=200,
            reward_fn=lambda e, n: 30.0,
            success_fn=lambda e, n: 0.2,
        )

        result = evaluate_convergence(df)

        assert result["training_status"] == "not_converged"

    def test_only_reward_improves_is_partially_converged(self):
        df = self._make_df(
            n_episodes=200,
            reward_fn=lambda e, n: -100 + 200 * (e / n),
            success_fn=lambda e, n: 0.2,  # se mantiene plano
        )

        result = evaluate_convergence(df)

        assert result["training_status"] == "partially_converged"

    def test_short_run_includes_overlap_note(self):
        df = self._make_df(
            n_episodes=8,
            reward_fn=lambda e, n: e,
            success_fn=lambda e, n: 0.1,
        )

        result = evaluate_convergence(df)

        assert len(result["evaluation_notes"]) > 0


class TestCombineRecommendation:
    """La tabla de combinación debe reflejar exactamente el diseño documentado."""

    @pytest.mark.parametrize(
        "training_status,historical_status,expected",
        [
            ("converged", "approved", "approved"),
            ("converged", "warning", "approved"),
            ("converged", "rejected", "warning"),
            ("partially_converged", "approved", "approved"),
            ("partially_converged", "warning", "warning"),
            ("partially_converged", "rejected", "warning"),
            ("not_converged", "approved", "warning"),
            ("not_converged", "warning", "rejected"),
            ("not_converged", "rejected", "rejected"),
        ],
    )
    def test_recommendation_table_matches_design(self, training_status, historical_status, expected):
        result = combine_recommendation(training_status, historical_status)
        assert result == expected

    def test_unknown_combination_raises_value_error(self):
        with pytest.raises(ValueError):
            combine_recommendation("not_a_real_status", "approved")
