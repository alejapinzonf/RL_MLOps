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
    """
    Guarda un checkpoint simulado del entrenamiento.

    En un entrenamiento real aquí guardaríamos:
    - Q-table para Q-learning
    - pesos de red para DQN
    - actor/critic para SAC

    Por ahora guardamos un diccionario serializado con pickle.
    """

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
        "checkpoint_type": "simulated_training_checkpoint",
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
    Guarda un modelo final simulado.

    Esto todavía no es un modelo real de RL.
    Es un artefacto placeholder para probar el flujo MLOps.
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
        "model_type": "simulated_rl_policy",
    }

    model_path = run_model_dir / "final_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model_data, file)

    metadata_path = run_model_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(model_data, file, indent=4, ensure_ascii=False)

    return model_path, metadata_path
