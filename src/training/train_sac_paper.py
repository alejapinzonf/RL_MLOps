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
from training.sac_agent import DiscreteSACAgent


ALGORITHM = "sac"
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
    return f"paper_real_sac_{scenario}_{timestamp}"


def train_one_episode(env: PaperGridWorldEnv, agent: DiscreteSACAgent) -> dict:
    state = env.reset()

    total_reward = 0.0
    collisions = 0
    reached_goal = False
    first_reach_step = None
    min_distance = env._euclidean(state[0], state[1])

    critic_losses = []
    actor_losses = []
    alpha_losses = []
    td_errors = []
    mean_qs = []
    alphas = []
    entropies = []

    step = 0
    for step in range(1, env.max_steps + 1):
        action = agent.choose_action(state, deterministic=False)

        result = env.step(action)

        agent.store_transition(state, action, result.reward, result.next_state, result.done)
        train_info = agent.train_step()

        if train_info is not None:
            critic_losses.append(train_info["critic_loss"])
            actor_losses.append(train_info["actor_loss"])
            alpha_losses.append(train_info["alpha_loss"])
            td_errors.append(train_info["td_error"])
            mean_qs.append(train_info["mean_q_value"])
            alphas.append(train_info["alpha"])
            entropies.append(train_info["entropy"])

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
        "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "alpha_loss": float(np.mean(alpha_losses)) if alpha_losses else 0.0,
        "mean_q_value": float(np.mean(mean_qs)) if mean_qs else 0.0,
        "alpha": float(np.mean(alphas)) if alphas else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
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
        "notes": "real discrete sac training run (paper environment/reward)",
    }

    pd.DataFrame([new_run]).to_csv(NEW_RUN_PATH, index=False)
    return NEW_RUN_PATH


def save_model_checkpoint(run_id: str, episode: int, agent: DiscreteSACAgent, metrics: dict):
    run_dir = CHECKPOINTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / f"sac_checkpoint_ep_{episode}.pt"
    torch.save(
        {
            "episode": episode,
            "model_state": agent.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_final_model(run_id: str, agent: DiscreteSACAgent, final_metrics: dict, training_config: dict):
    run_dir = CANDIDATE_MODELS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "sac_model.pt"
    torch.save(agent.state_dict(), model_path)

    import json

    metadata = {
        "run_id": run_id,
        "algorithm": ALGORITHM,
        "model_type": "discrete_sac_pytorch",
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
        description="Entrenamiento real Discrete SAC usando el entorno y reward del paper."
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
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--initial-alpha", type=float, default=0.3)
    parser.add_argument("--alpha-lr", type=float, default=1e-5)
    parser.add_argument("--target-entropy-fraction", type=float, default=0.4)

    parser.add_argument("--arrival-reward", type=float, default=150.0)
    parser.add_argument("--goal-stay-reward", type=float, default=50.0)
    parser.add_argument("--goal-stay-out-penalty", type=float, default=100.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--stay-outside-penalty", type=float, default=60.0)
    parser.add_argument("--obstacle-hit-penalty", type=float, default=100.0)
    parser.add_argument("--goal-position-scale", type=float, default=2.0)
    parser.add_argument("--obstacle-position-scale", type=float, default=2.0)
    parser.add_argument("--arrival-bonus-multiplier", type=float, default=100.0)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval debe ser mayor que 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = generate_run_id(args.scenario)

    reward_params = {
        "arrival_reward": args.arrival_reward,
        "goal_stay_reward": args.goal_stay_reward,
        "goal_stay_out_penalty": args.goal_stay_out_penalty,
        "step_penalty": args.step_penalty,
        "stay_outside_penalty": args.stay_outside_penalty,
        "obstacle_hit_penalty": args.obstacle_hit_penalty,
        "goal_position_scale": args.goal_position_scale,
        "obstacle_position_scale": args.obstacle_position_scale,
        "arrival_bonus_multiplier": args.arrival_bonus_multiplier,
    }

    env = PaperGridWorldEnv(
        grid_size=args.grid_size,
        scenario=args.scenario,
        obstacle_length=args.obstacle_length,
        max_steps=args.max_steps,
        min_goal_start_distance=args.min_goal_start_distance,
        seed=args.seed,
        reward_params=reward_params,
    )

    agent = DiscreteSACAgent(
        state_dim=STATE_DIM,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        initial_alpha=args.initial_alpha,
        alpha_lr=args.alpha_lr,
        target_entropy_fraction=args.target_entropy_fraction,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    print(f"Usando device: {agent.device}")

    episode_metrics = []
    checkpoint_paths = []

    start_time = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        metrics = train_one_episode(env, agent)
        metrics["episode"] = episode
        episode_metrics.append(metrics)

        if episode % args.checkpoint_interval == 0 or episode == args.episodes:
            checkpoint_path = save_model_checkpoint(run_id, episode, agent, metrics)
            checkpoint_paths.append(checkpoint_path)

        print_every = max(1, args.episodes // 20)
        if episode % print_every == 0:
            recent = episode_metrics[-print_every:]
            recent_success = np.mean([item["success"] for item in recent])
            recent_reward = np.mean([item["episode_reward"] for item in recent])
            recent_alpha = np.mean([item["alpha"] for item in recent])
            recent_entropy = np.mean([item["entropy"] for item in recent])

            print(
                f"Episode {episode}/{args.episodes} "
                f"| success_rate_recent={recent_success:.3f} "
                f"| avg_reward_recent={recent_reward:.2f} "
                f"| alpha_recent={recent_alpha:.3f} "
                f"| entropy_recent={recent_entropy:.3f}"
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
        "tau": args.tau,
        "initial_alpha": args.initial_alpha,
        "seed": args.seed,
    }

    episode_metrics_path = save_episode_metrics(run_id, episode_metrics)
    model_path, metadata_path = save_final_model(run_id, agent, final_metrics, training_config)
    new_run_path = save_new_run_csv(run_id, args, final_metrics)

    print("\nEntrenamiento real Discrete SAC (paper env) completado.")
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
