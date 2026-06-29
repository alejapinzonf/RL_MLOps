from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """
    Red Q de DQN. Arquitectura idéntica a la usada en dqn_wall.py /
    dqn_L.py / dqn_u.py: MLP de 2 capas ocultas con ReLU.
    """

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
    """Buffer de experiencia (state, action, reward, next_state)."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)

        return states, actions, rewards, next_states

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN completo: red de política, red objetivo, optimizador,
    replay buffer y lógica de selección de acción/entrenamiento.

    Port fiel de la lógica en dqn_wall.py, encapsulado en una clase
    reutilizable para que train_dqn_paper.py pueda parametrizarlo en
    vez de depender de constantes globales del script original.
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
        target_update_every: int = 2_000,
        gradient_clip: float | None = 10.0,
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
        self.target_update_every = target_update_every
        self.gradient_clip = gradient_clip
        self.grid_size = grid_size

        self.policy_net = DQNNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.global_step = 0

    def normalize_state(self, state: tuple[int, int, int, int]) -> np.ndarray:
        """
        Normaliza el estado relativo (dx, dy, dxo, dyo) a [-1, 1]
        dividiendo por (grid_size - 1), igual que normalize_state en
        dqn_wall.py.
        """
        scale = self.grid_size - 1
        return np.array([component / scale for component in state], dtype=np.float32)

    def choose_action(self, state: tuple[int, int, int, int], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        normalized = self.normalize_state(state)
        with torch.no_grad():
            state_t = torch.tensor(normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(np.random.choice(best_actions))

    def store_transition(
        self,
        state: tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int],
    ):
        self.replay_buffer.push(
            self.normalize_state(state),
            action,
            reward,
            self.normalize_state(next_state),
        )

    def train_step(self) -> dict | None:
        """
        Ejecuta un paso de optimización si hay suficientes muestras en
        el buffer. Devuelve métricas de la actualización, o None si
        todavía no se entrena (buffer por debajo de learning_starts).
        """
        self.global_step += 1

        if len(self.replay_buffer) < self.learning_starts:
            return None

        states, actions, rewards, next_states = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)

        self.optimizer.step()

        if self.global_step % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        td_abs = torch.abs(target - q_values).mean().item()
        mean_q = q_values.mean().item()

        return {"loss": loss.item(), "td_error": td_abs, "mean_q_value": mean_q}

    def state_dict(self) -> dict:
        """Para checkpointing: pesos de la red de política."""
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.policy_net.load_state_dict(state["policy_net"])
        self.target_net.load_state_dict(state["target_net"])