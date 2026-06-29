from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    """Política discreta: produce logits sobre las acciones."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    """Crítico Q discreto: produce un Q-value por acción."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Buffer de experiencia (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


def soft_update(source_net: nn.Module, target_net: nn.Module, tau: float):
    for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class DiscreteSACAgent:
    """
    Agente Discrete SAC completo: actor (política categórica), dos
    críticos con sus respectivas target networks, coeficiente de
    entropía adaptativo (log_alpha aprendible), y replay buffer.

    Port fiel de la lógica en sac_wall.py / sac_u.py, encapsulado en
    una clase reutilizable para que train_sac_paper.py pueda
    parametrizarlo en vez de depender de constantes globales del
    script original.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        learning_starts: int = 5_000,
        tau: float = 0.005,
        gradient_clip: float | None = 10.0,
        initial_alpha: float = 0.3,
        alpha_lr: float = 1e-5,
        target_entropy_fraction: float = 0.4,
        grid_size: int = 20,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if seed is not None:
            torch.manual_seed(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gradient_clip = gradient_clip
        self.grid_size = grid_size

        self.actor = ActorNetwork(state_dim, n_actions, hidden_dim).to(self.device)

        self.critic1 = CriticNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, n_actions, hidden_dim).to(self.device)

        self.target_critic1 = CriticNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(
            float(np.log(initial_alpha)), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy_fraction * float(np.log(n_actions))

        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.global_step = 0

    def normalize_state(self, state: tuple[int, int, int, int]) -> np.ndarray:
        scale = self.grid_size - 1
        return np.array([component / scale for component in state], dtype=np.float32)

    def choose_action(self, state: tuple[int, int, int, int], deterministic: bool = False) -> int:
        normalized = self.normalize_state(state)

        with torch.no_grad():
            state_t = torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(state_t)
            probs = F.softmax(logits, dim=-1).squeeze(0)

            if deterministic:
                return int(torch.argmax(probs).item())

            action_dist = torch.distributions.Categorical(probs=probs)
            return int(action_dist.sample().item())

    def store_transition(
        self,
        state: tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int],
        done: bool,
    ):
        self.replay_buffer.push(
            self.normalize_state(state),
            action,
            reward,
            self.normalize_state(next_state),
            float(done),
        )

    def train_step(self) -> dict | None:
        """
        Ejecuta un paso de optimización de actor, críticos y alpha si
        hay suficientes muestras en el buffer. Devuelve métricas de la
        actualización, o None si todavía no se entrena.
        """
        self.global_step += 1

        if len(self.replay_buffer) < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        alpha = self.log_alpha.exp()

        q1_values = self.critic1(states).gather(1, actions)
        q2_values = self.critic2(states).gather(1, actions)

        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)
            next_min_q = torch.min(next_q1, next_q2)

            next_v = (next_probs * (next_min_q - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target = rewards + self.gamma * (1.0 - dones) * next_v

        critic1_loss = self.loss_fn(q1_values, target)
        critic2_loss = self.loss_fn(q2_values, target)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()

        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clip)

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            q1_all = self.critic1(states)
            q2_all = self.critic2(states)
            min_q_all = torch.min(q1_all, q2_all)

        actor_loss = (probs * (alpha.detach() * log_probs - min_q_all)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)

        self.actor_optimizer.step()

        entropy = -(probs * log_probs).sum(dim=1).mean()
        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.critic1, self.target_critic1, self.tau)
        soft_update(self.critic2, self.target_critic2, self.tau)

        td_abs = torch.abs(target - q1_values).mean().item()
        mean_q = q1_values.mean().item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "td_error": td_abs,
            "mean_q_value": mean_q,
            "alpha": alpha.item(),
            "entropy": entropy.item(),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, state: dict):
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))