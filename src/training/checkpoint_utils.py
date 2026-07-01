from pathlib import Path
from datetime import datetime
import json
import pickle


def save_checkpoint(
    checkpoint_dir: Path,
    run_id: str,
    episode: int,
    algorithm: str,
    scenario: str,
    reward_version: str,
    metrics: dict,
):


    run_checkpoint_dir = checkpoint_dir / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "run_id": run_id,
        "episode": episode,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": reward_version,
        "metrics": metrics,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_type": "demo_training_checkpoint",
    }

    checkpoint_path = run_checkpoint_dir / f"checkpoint_ep_{episode}.pkl"

    with open(checkpoint_path, "wb") as file:
        pickle.dump(checkpoint_data, file)

    return checkpoint_path


def save_final_model(
    model_dir: Path,
    run_id: str,
    algorithm: str,
    scenario: str,
    reward_version: str,
    final_metrics: dict,
):
    """
    Guarda un modelo demo/simulado.
    Esta función la usa demo_train.py.
    """

    run_model_dir = model_dir / run_id
    run_model_dir.mkdir(parents=True, exist_ok=True)

    model_data = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": reward_version,
        "final_metrics": final_metrics,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "demo_model",
    }

    model_path = run_model_dir / "final_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model_data, file)

    metadata = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": reward_version,
        "final_metrics": final_metrics,
        "model_type": "demo_model",
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = run_model_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return model_path, metadata_path


def save_q_table_checkpoint(
    checkpoint_dir: Path,
    run_id: str,
    episode: int,
    algorithm: str,
    scenario: str,
    reward_version: str,
    q_table,
    metrics: dict,
    epsilon: float,
):
    """
    Guarda un checkpoint real de Q-learning.
    Incluye la Q-table aprendida hasta cierto episodio.
    """

    run_checkpoint_dir = checkpoint_dir / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "run_id": run_id,
        "episode": episode,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": reward_version,
        "q_table": q_table,
        "metrics": metrics,
        "epsilon": epsilon,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_type": "real_q_learning_checkpoint",
    }

    checkpoint_path = run_checkpoint_dir / f"q_table_checkpoint_ep_{episode}.pkl"

    with open(checkpoint_path, "wb") as file:
        pickle.dump(checkpoint_data, file)

    return checkpoint_path


def save_q_table_model(
    model_dir: Path,
    run_id: str,
    algorithm: str,
    scenario: str,
    reward_version: str,
    q_table,
    final_metrics: dict,
    training_config: dict,
):
    """
    Guarda el modelo final real de Q-learning.
    """

    run_model_dir = model_dir / run_id
    run_model_dir.mkdir(parents=True, exist_ok=True)

    model_data = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": reward_version,
        "q_table": q_table,
        "final_metrics": final_metrics,
        "training_config": training_config,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "real_q_learning_q_table",
    }

    model_path = run_model_dir / "q_table_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model_data, file)

    metadata = {
        "run_id": run_id,
        "algorithm": algorithm,
        "scenario": scenario,
        "reward_version": reward_version,
        "final_metrics": final_metrics,
        "training_config": training_config,
        "model_type": "real_q_learning_q_table",
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_path = run_model_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return model_path, metadata_path
