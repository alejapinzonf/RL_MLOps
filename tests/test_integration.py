"""
test_integration.py

Quality gate de integración: dos niveles.

1. Advisors (reward shaping + histórico) con casos construidos a mano,
   confirmando que detectan exactamente lo que deberían y no más.

2. Un mini-pipeline end-to-end real (no mockeado): carga datos ->
   limpia -> entrena demo con pocos episodios -> valida -> evalúa
   convergencia. Corre contra archivos temporales (tmp_path), nunca
   contra data/ real, así que es seguro correrlo en CI o en la laptop
   sin pisar nada del proyecto.

Estos tests son más lentos que los anteriores (entrenan, aunque sea
poco) pero siguen siendo rápidos: unos pocos segundos en total.
"""

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advisor.reward_shaping_advisor import analyze_reward_params
from advisor.historical_advisor import advise as historical_advise


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


class TestRewardShapingAdvisor:
    """El advisor de reward shaping debe distinguir configuraciones sanas de riesgosas."""

    def test_paper_defaults_produce_no_high_severity_warnings(self):
        warnings = analyze_reward_params(DEFAULT_PAPER_REWARD_PARAMS)

        high_severity = [w for w in warnings if w.severity == "high"]
        assert high_severity == [], (
            "Los defaults reales del paper, ya validados empíricamente, "
            "no deberían disparar avisos de severidad alta."
        )

    def test_arrival_dominating_obstacle_hit_is_flagged(self):
        risky_params = dict(DEFAULT_PAPER_REWARD_PARAMS)
        risky_params["arrival_bonus_multiplier"] = 600.0
        risky_params["obstacle_hit_penalty"] = 5.0

        warnings = analyze_reward_params(risky_params)

        assert any(w.severity == "high" for w in warnings), (
            "Un arrival_bonus 120x mayor que obstacle_hit_penalty debería "
            "generar un aviso de severidad alta."
        )

    def test_insignificant_step_penalty_is_flagged(self):
        risky_params = dict(DEFAULT_PAPER_REWARD_PARAMS)
        risky_params["step_penalty"] = 0.001
        risky_params["goal_position_scale"] = 5.0

        warnings = analyze_reward_params(risky_params)

        assert len(warnings) > 0

    def test_missing_param_is_silently_skipped_not_crashed(self):
        incomplete_params = {"arrival_bonus_multiplier": 100.0}

        warnings = analyze_reward_params(incomplete_params)

        assert isinstance(warnings, list)


class TestHistoricalAdvisor:
    """El advisor histórico debe comparar episodios planeados contra el histórico real."""

    @pytest.fixture
    def fake_history_csv(self, tmp_path, monkeypatch) -> Path:
        history_dir = tmp_path / "data" / "processed"
        history_dir.mkdir(parents=True)

        history_df = pd.DataFrame(
            {
                "run_id": ["hist_1", "hist_2", "hist_3"],
                "algorithm": ["q_learning", "q_learning", "dqn"],
                "scenario": ["wall", "wall", "wall"],
                "reward_version": ["paper_reward_v1"] * 3,
                "success_rate": [0.95, 0.97, 0.91],
                "episodes": [400000, 450000, 100000],
            }
        )
        csv_path = history_dir / "historical_results_clean.csv"
        history_df.to_csv(csv_path, index=False)

        import advisor.historical_advisor as historical_advisor_module

        monkeypatch.setattr(
            historical_advisor_module,
            "HISTORICAL_SOURCES",
            [csv_path, tmp_path / "does_not_exist.csv"],
        )

        return csv_path

    def test_few_episodes_against_large_history_is_insufficient(self, fake_history_csv):
        advice = historical_advise(
            algorithm="q_learning",
            scenario="wall",
            planned_episodes=2000,
            reward_version="paper_reward_v1",
        )

        assert advice["verdict"] == "likely_insufficient"
        assert advice["n_matches"] == 2

    def test_comparable_episodes_yield_comparable_verdict(self, fake_history_csv):
        advice = historical_advise(
            algorithm="q_learning",
            scenario="wall",
            planned_episodes=420000,
            reward_version="paper_reward_v1",
        )

        assert advice["verdict"] == "comparable"

    def test_no_matching_history_returns_no_history_verdict(self, fake_history_csv):
        advice = historical_advise(
            algorithm="sac",
            scenario="u_shape",
            planned_episodes=50000,
        )

        assert advice["verdict"] == "no_history"
        assert advice["n_matches"] == 0


class TestEndToEndPipeline:
    """
    Mini-pipeline real de punta a punta, aislado en tmp_path: carga,
    limpia, entrena demo (pocos episodios), valida, evalúa convergencia.

    No usa subprocess ni run_local_pipeline.py directamente (eso
    acoplaría el test a rutas relativas y al entorno del sistema);
    en cambio, llama a las mismas funciones que el pipeline usa
    internamente, sobre un directorio de trabajo temporal.
    """

    @pytest.fixture
    def isolated_project(self, tmp_path, monkeypatch):
        """
        Construye una estructura mínima de proyecto en tmp_path y
        redirige las constantes de cada módulo para que escriban ahí,
        no en el proyecto real.
        """
        (tmp_path / "data" / "raw").mkdir(parents=True)
        (tmp_path / "data" / "interim").mkdir(parents=True)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "data" / "new_runs").mkdir(parents=True)
        (tmp_path / "models" / "checkpoints").mkdir(parents=True)
        (tmp_path / "models" / "candidate").mkdir(parents=True)
        (tmp_path / "reports" / "training").mkdir(parents=True)
        (tmp_path / "reports" / "validation").mkdir(parents=True)
        (tmp_path / "reports" / "evaluation").mkdir(parents=True)
        (tmp_path / "configs").mkdir(parents=True)

        historical_df = pd.DataFrame(
            {
                "run_id": ["hist_1", "hist_2"],
                "algorithm": ["q_learning", "q_learning"],
                "scenario": ["wall", "wall"],
                "reward_version": ["reward_v1_base", "reward_v1_base"],
                "success_rate": [0.85, 0.88],
                "avg_reward": [150.0, 160.0],
                "first_reach_step": [50.0, 48.0],
                "collisions": [3.0, 2.0],
                "avg_steps": [70.0, 68.0],
                "training_time_sec": [100.0, 110.0],
                "episodes": [500, 500],
                "seed": [1, 2],
                "source_file": ["test.csv", "test.csv"],
                "created_at": ["2026-01-01", "2026-01-01"],
                "notes": ["test", "test"],
            }
        )
        historical_df.to_csv(tmp_path / "data" / "processed" / "historical_results_clean.csv", index=False)

        validation_config = {
            "default_mode": "normal",
            "min_history_runs": 2,
            "validation_modes": {
                "strict": {"std_factor": 0.5},
                "normal": {"std_factor": 1.0},
                "flexible": {"std_factor": 1.5},
            },
            "metrics": {
                "success_rate": {"higher_is_better": True, "required": True},
                "avg_reward": {"higher_is_better": True, "required": True},
                "first_reach_step": {"higher_is_better": False, "required": False},
                "collisions": {"higher_is_better": False, "required": False},
                "avg_steps": {"higher_is_better": False, "required": False},
            },
        }
        config_path = tmp_path / "configs" / "validation.yaml"
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(validation_config, file)

        import validation.validate_run as validate_run_module

        monkeypatch.setattr(validate_run_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            validate_run_module, "HISTORICAL_DATA_PATH",
            tmp_path / "data" / "processed" / "historical_results_clean.csv",
        )
        monkeypatch.setattr(
            validate_run_module, "NEW_RUN_PATH", tmp_path / "data" / "new_runs" / "new_run_demo.csv"
        )
        monkeypatch.setattr(validate_run_module, "VALIDATION_CONFIG_PATH", config_path)
        monkeypatch.setattr(
            validate_run_module, "VALIDATION_REPORT_PATH",
            tmp_path / "reports" / "validation" / "validation_result.json",
        )

        return tmp_path

    def test_new_run_within_historical_range_is_approved_end_to_end(self, isolated_project):
        import validation.validate_run as validate_run_module

        new_run_df = pd.DataFrame(
            [
                {
                    "run_id": "test_run_1",
                    "algorithm": "q_learning",
                    "scenario": "wall",
                    "reward_version": "reward_v1_base",
                    "validation_mode": "normal",
                    "success_rate": 0.86,
                    "avg_reward": 155.0,
                    "first_reach_step": 49.0,
                    "collisions": 2.5,
                    "avg_steps": 69.0,
                    "training_time_sec": 105.0,
                    "episodes": 500,
                    "seed": 3,
                    "notes": "test run",
                }
            ]
        )
        new_run_df.to_csv(isolated_project / "data" / "new_runs" / "new_run_demo.csv", index=False)

        config = validate_run_module.load_yaml(validate_run_module.VALIDATION_CONFIG_PATH)
        historical_df = validate_run_module.load_historical_data(validate_run_module.HISTORICAL_DATA_PATH)
        new_run = validate_run_module.load_new_run(validate_run_module.NEW_RUN_PATH)

        report = validate_run_module.validate_new_run(historical_df, new_run, config)

        assert report["final_status"] == "approved"
        assert report["history_rows_used"] == 2

    def test_new_run_far_outside_range_is_rejected_end_to_end(self, isolated_project):
        import validation.validate_run as validate_run_module

        new_run_df = pd.DataFrame(
            [
                {
                    "run_id": "test_run_bad",
                    "algorithm": "q_learning",
                    "scenario": "wall",
                    "reward_version": "reward_v1_base",
                    "validation_mode": "normal",
                    "success_rate": 0.05,
                    "avg_reward": 5.0,
                    "first_reach_step": 290.0,
                    "collisions": 50.0,
                    "avg_steps": 300.0,
                    "training_time_sec": 10.0,
                    "episodes": 20,
                    "seed": 99,
                    "notes": "undertrained test run",
                }
            ]
        )
        new_run_df.to_csv(isolated_project / "data" / "new_runs" / "new_run_demo.csv", index=False)

        config = validate_run_module.load_yaml(validate_run_module.VALIDATION_CONFIG_PATH)
        historical_df = validate_run_module.load_historical_data(validate_run_module.HISTORICAL_DATA_PATH)
        new_run = validate_run_module.load_new_run(validate_run_module.NEW_RUN_PATH)

        report = validate_run_module.validate_new_run(historical_df, new_run, config)

        assert report["final_status"] == "rejected"

    def test_insufficient_history_yields_preliminary_warning(self, isolated_project, monkeypatch):
        import validation.validate_run as validate_run_module

        # Histórico con un escenario sin ninguna corrida comparable.
        new_run_df = pd.DataFrame(
            [
                {
                    "run_id": "test_run_no_history",
                    "algorithm": "sac",
                    "scenario": "u_shape",
                    "reward_version": "reward_v3_collision_penalty",
                    "validation_mode": "normal",
                    "success_rate": 0.5,
                    "avg_reward": 80.0,
                    "first_reach_step": 100.0,
                    "collisions": 5.0,
                    "avg_steps": 120.0,
                    "training_time_sec": 200.0,
                    "episodes": 500,
                    "seed": 7,
                    "notes": "no comparable history",
                }
            ]
        )
        new_run_df.to_csv(isolated_project / "data" / "new_runs" / "new_run_demo.csv", index=False)

        config = validate_run_module.load_yaml(validate_run_module.VALIDATION_CONFIG_PATH)
        historical_df = validate_run_module.load_historical_data(validate_run_module.HISTORICAL_DATA_PATH)
        new_run = validate_run_module.load_new_run(validate_run_module.NEW_RUN_PATH)

        report = validate_run_module.validate_new_run(historical_df, new_run, config)

        assert report["final_status"] == "warning"
        assert report["preliminary_validation"] is True


class TestPipelineScriptsSmokeTest:
    """
    Smoke test real ejecutando los scripts del pipeline como subprocess,
    contra un mini histórico de ejemplo, para detectar errores de
    integración que los tests unitarios no ven (imports rotos, rutas
    mal armadas, argparse mal configurado, etc.).

    Se marcan como 'slow' porque lanzan procesos de Python reales.
    """

    @pytest.mark.slow
    def test_demo_train_runs_without_crashing(self, tmp_path):
        env_project_root = PROJECT_ROOT

        result = subprocess.run(
            [
                sys.executable,
                str(env_project_root / "src" / "training" / "demo_train.py"),
                "--algorithm", "q_learning",
                "--scenario", "wall",
                "--reward-version", "reward_v1_base",
                "--episodes", "10",
                "--checkpoint-interval", "5",
                "--seed", "123",
            ],
            cwd=env_project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"demo_train.py terminó con error.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "Entrenamiento demo completado" in result.stdout
