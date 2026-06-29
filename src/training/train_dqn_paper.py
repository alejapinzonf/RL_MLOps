from pathlib import Path
from datetime import datetime
import argparse
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment.paper_grid_world import PaperGridWorldEnv
from training.dqn_agent import DQNAgent


ALGORITHM = "dqn"
REWARD_VERSION = "paper_reward_v1"

NEW_RUNS_DIR = PROJECT_ROOT / "data" / "new_runs"
NEW_RUN_PATH = NEW_RUNS_DIR / "new_run_demo.csv"

CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
CANDIDATE_MODELS_DIR = PROJECT_ROOT / "models" / "candidate"
TRAINING_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"

VALID_SCENARIOS = ["wall", "l_shape", "u_shape"]
STATE_DIM = 4  # (dx, dy, dxo, dyo)


def generate_run_id(scenario: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"paper_real_dqn_{scenario}_{timestamp}"


def get_epsilon(episode: int, total_episodes: int, args: argparse.Namespace) -> float:
    decay_episodes = max(1, total_episodes * args.epsilon_decay_fraction)
    return max(
        args.epsilon_end,
        args.epsilon_start - (episode * args.epsilon_start / decay_episodes),
    )


def train_one_episode(
    env: PaperGridWorldEnv,
    agent: DQNAgent,
    epsilon: float,
) -> dict:
    state = env.reset()

    total_reward = 0.0
    collisions = 0
    reached_goal = False
    first_reach_step = None
    min_distance = env._euclidean(state[0], state[1])

    losses = []
    td_errors = []
    mean_qs = []

    step = 0
    for step in range(1, env.max_steps + 1):
        action = agent.choose_action(state, epsilon)

        result = env.step(action)

        agent.store_transition(state, action, result.reward, result.next_state)
        train_info = agent.train_step()

        if train_info is not None:
            losses.append(train_info["loss"])
            td_errors.append(train_info["td_error"])
            mean_qs.append(train_info["mean_q_value"])

        total_reward += result.reward

        if result.info["collision"]:
            collisions += 1

        distance_now = result.info["distance_to_goal"]
        if distance_now < min_distance:
            min_distance = distance_now

        if result.info["reached_goal"] and not reached_goal:
            reached_goal = True
            first_reach_step = step

        state = result.next_state

        if result.done:
            break

    episode_metrics = {
        "episode_reward": total_reward,
        "success": int(reached_goal),
        "min_distance_to_goal": float(min_distance),
        "first_reach_step": first_reach_step if first_reach_step is not None else env.max_steps,
        "collisions": collisions,
        "steps": step,
        "td_error": float(np.mean(td_errors)) if td_errors else 0.0,
        "dqn_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_q_value": float(np.mean(mean_qs)) if mean_qs else 0.0,
        "replay_size": len(agent.replay_buffer),
    }

    return episode_metrics


def summarize_final_metrics(
    episode_metrics: list[dict],
    training_time_sec: float,
    window_size: int,
) -> dict:
    window = episode_metrics[-window_size:]

    return {
        "success_rate": round(float(np.mean([item["success"] for item in window])), 4),
        "avg_reward": round(float(np.mean([item["episode_reward"] for item in window])), 4),
        "first_reach_step": round(float(np.mean([item["first_reach_step"] for item in window])), 4),
        "collisions": round(float(np.mean([item["collisions"] for item in window])), 4),
        "avg_steps": round(float(np.mean([item["steps"] for item in window])), 4),
        "training_time_sec": round(float(training_time_sec), 4),
    }


def save_episode_metrics(run_id: str, episode_metrics: list[dict]) -> Path:
    TRAINING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_REPORTS_DIR / f"{run_id}_episode_metrics.csv"
    pd.DataFrame(episode_metrics).to_csv(output_path, index=False)
    return output_path


def save_new_run_csv(run_id: str, args: argparse.Namespace, final_metrics: dict) -> Path:
    NEW_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    new_run = {
        "run_id": run_id,
        "algorithm": ALGORITHM,
        "scenario": args.scenario,
        "reward_version": REWARD_VERSION,
        "validation_mode": args.validation_mode,
        "success_rate": final_metrics["success_rate"],
        "avg_reward": final_metrics["avg_reward"],
        "first_reach_step": final_metrics["first_reach_step"],
        "collisions": final_metrics["collisions"],
        "avg_steps": final_metrics["avg_steps"],
        "training_time_sec": final_metrics["training_time_sec"],
        "episodes": args.episodes,
        "seed": args.seed,
        "notes": "real dqn training run (paper environment/reward)",
    }

    pd.DataFrame([new_run]).to_csv(NEW_RUN_PATH, index=False)
    return NEW_RUN_PATH


def save_model_checkpoint(run_id: str, episode: int, agent: DQNAgent, metrics: dict):
    run_dir = CHECKPOINTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / f"dqn_checkpoint_ep_{episode}.pt"
    torch.save(
        {
            "episode": episode,
            "model_state": agent.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_final_model(run_id: str, agent: DQNAgent, final_metrics: dict, training_config: dict):
    run_dir = CANDIDATE_MODELS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "dqn_model.pt"
    torch.save(agent.state_dict(), model_path)

    import json

    metadata = {
        "run_id": run_id,
        "algorithm": ALGORITHM,
        "model_type": "dqn_pytorch",
        "final_metrics": final_metrics,
        "training_config": training_config,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = run_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return model_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento real DQN usando el entorno y reward del paper."
    )

    parser.add_argument("--scenario", choices=VALID_SCENARIOS, default="wall")
    parser.add_argument("--validation-mode", choices=["strict", "normal", "flexible"], default="normal")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--obstacle-length", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-goal-start-distance", type=int, default=2)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--target-update-every", type=int, default=2_000)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.90)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval debe ser mayor que 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = generate_run_id(args.scenario)

    env = PaperGridWorldEnv(
        grid_size=args.grid_size,
        scenario=args.scenario,
        obstacle_length=args.obstacle_length,
        max_steps=args.max_steps,
        min_goal_start_distance=args.min_goal_start_distance,
        seed=args.seed,
    )

    agent = DQNAgent(
        state_dim=STATE_DIM,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        target_update_every=args.target_update_every,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    print(f"Usando device: {agent.device}")

    epsilon = args.epsilon_start
    episode_metrics = []
    checkpoint_paths = []

    start_time = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        metrics = train_one_episode(env, agent, epsilon)
        metrics["episode"] = episode
        metrics["epsilon"] = epsilon
        episode_metrics.append(metrics)

        if episode % args.checkpoint_interval == 0 or episode == args.episodes:
            checkpoint_path = save_model_checkpoint(run_id, episode, agent, metrics)
            checkpoint_paths.append(checkpoint_path)

        epsilon = get_epsilon(episode, args.episodes, args)

        print_every = max(1, args.episodes // 20)
        if episode % print_every == 0:
            recent = episode_metrics[-print_every:]
            recent_success = np.mean([item["success"] for item in recent])
            recent_reward = np.mean([item["episode_reward"] for item in recent])
            recent_loss = np.mean([item["dqn_loss"] for item in recent])

            print(
                f"Episode {episode}/{args.episodes} "
                f"| success_rate_recent={recent_success:.3f} "
                f"| avg_reward_recent={recent_reward:.2f} "
                f"| loss_recent={recent_loss:.4f} "
                f"| epsilon={epsilon:.3f}"
            )

    training_time_sec = time.perf_counter() - start_time

    final_metrics = summarize_final_metrics(
        episode_metrics=episode_metrics,
        training_time_sec=training_time_sec,
        window_size=max(50, int(args.episodes * 0.10)),
    )

    training_config = {
        "algorithm": ALGORITHM,
        "scenario": args.scenario,
        "reward_version": REWARD_VERSION,
        "episodes": args.episodes,
        "grid_size": args.grid_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "hidden_dim": args.hidden_dim,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    episode_metrics_path = save_episode_metrics(run_id, episode_metrics)
    model_path, metadata_path = save_final_model(run_id, agent, final_metrics, training_config)
    new_run_path = save_new_run_csv(run_id, args, final_metrics)

    print("\nEntrenamiento real DQN (paper env) completado.")
    print(f"run_id: {run_id}")
    print(f"Archivo de nueva corrida: {new_run_path}")
    print(f"Métricas por episodio: {episode_metrics_path}")
    print(f"Modelo: {model_path}")
    print(f"Metadata: {metadata_path}")

    print("\nCheckpoints guardados:")
    for path in checkpoint_paths:
        print(f"- {path}")

    print("\nMétricas finales (ventana final):")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()