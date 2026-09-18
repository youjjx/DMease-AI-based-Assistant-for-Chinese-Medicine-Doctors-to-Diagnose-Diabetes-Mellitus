from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .kan import KANPolicy, torch


@dataclass
class Rollout:
    states: list
    actions: list
    old_log_probs: list
    returns: list
    advantages: list
    masks: list


class HerbRecommendationEnv:
    """Sequential herb-selection environment built from expert graph weights."""

    def __init__(self, syndrome_herb_matrix, max_steps: int = 6):
        if torch is None:
            raise RuntimeError("Training requires PyTorch")
        self.matrix = syndrome_herb_matrix.float()
        self.max_steps = max_steps
        self.num_syndromes, self.num_herbs = self.matrix.shape
        self.reset()

    def reset(self, target=None):
        self.target = target if target is not None else torch.rand(self.num_syndromes)
        self.target = self.target / self.target.sum().clamp_min(1e-6)
        self.selected = torch.zeros(self.num_herbs)
        self.steps = 0
        return self.state()

    def state(self):
        return torch.cat([self.target, self.selected])

    def action_mask(self):
        return self.selected == 0

    def step(self, action: int):
        if self.selected[action] > 0:
            return self.state(), -1.0, True
        self.selected[action] = 1.0
        self.steps += 1
        relevance = float((self.target * self.matrix[:, action]).sum())
        redundancy = 0.05 * max(0, int(self.selected.sum()) - 1)
        reward = relevance - redundancy
        done = self.steps >= self.max_steps
        return self.state(), reward, done


class PPOTrainer:
    """Clipped PPO implementation for reproducible KAN policy training."""

    def __init__(self, policy: KANPolicy, lr: float = 3e-4, clip_ratio: float = 0.2):
        if torch is None:
            raise RuntimeError("Training requires PyTorch")
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio

    def collect(self, env: HerbRecommendationEnv, episodes: int = 8, gamma: float = 0.97) -> Rollout:
        states, actions, log_probs, returns, advantages, masks = [], [], [], [], [], []
        for _ in range(episodes):
            state = env.reset()
            episode = []
            done = False
            while not done:
                mask = env.action_mask()
                with torch.no_grad():
                    distribution, value = self.policy.distribution(state.unsqueeze(0), mask.unsqueeze(0))
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action)
                next_state, reward, done = env.step(int(action.item()))
                episode.append((state, action.squeeze(0), log_prob.squeeze(0), value.squeeze(0), reward, mask))
                state = next_state
            discounted = 0.0
            episode_returns = []
            for item in reversed(episode):
                discounted = item[4] + gamma * discounted
                episode_returns.append(discounted)
            for item, ret in zip(episode, reversed(episode_returns)):
                states.append(item[0]); actions.append(item[1]); log_probs.append(item[2])
                returns.append(torch.tensor(ret)); advantages.append(torch.tensor(ret) - item[3]); masks.append(item[5])
        advantage_tensor = torch.stack(advantages)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / advantage_tensor.std().clamp_min(1e-6)
        return Rollout(states, actions, log_probs, returns, list(advantage_tensor), masks)

    def update(self, rollout: Rollout, epochs: int = 4) -> dict[str, float]:
        states = torch.stack(rollout.states)
        actions = torch.stack(rollout.actions)
        old_log_probs = torch.stack(rollout.old_log_probs).detach()
        returns = torch.stack(rollout.returns).float()
        advantages = torch.stack(rollout.advantages).detach()
        masks = torch.stack(rollout.masks)
        metrics = {}
        for _ in range(epochs):
            distribution, values = self.policy.distribution(states, masks)
            new_log_probs = distribution.log_prob(actions)
            ratios = (new_log_probs - old_log_probs).exp()
            clipped = ratios.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
            critic_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + critic_loss - 0.01 * entropy
            self.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            metrics = {"loss": float(loss), "actor_loss": float(actor_loss), "critic_loss": float(critic_loss)}
        return metrics

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(), "metadata": metadata or {}}, path)
