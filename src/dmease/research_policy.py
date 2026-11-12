"""Versioned research improvements; original production checkpoints stay compatible."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .evidence_policy import EvidenceEnvironment


class CubicSplineLayer(nn.Module):
    """Cox-de Boor cubic B-splines on a fixed extended uniform grid."""
    def __init__(self, inputs, outputs, intervals=8, degree=3):
        super().__init__()
        self.degree = degree
        self.register_buffer("knots", torch.arange(-degree, intervals + degree + 1).float() * (2 / intervals) - 1)
        self.coefficients = nn.Parameter(torch.randn(outputs, inputs, intervals + degree) * .02)
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.xavier_uniform_(self.base)

    def basis(self, inputs):
        x = inputs.clamp(-1, 1).unsqueeze(-1)
        t = self.knots
        value = ((x >= t[:-1]) & (x < t[1:])).to(inputs.dtype)
        for k in range(1, self.degree + 1):
            value = ((x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)])) * value[..., :-1] + (
                (t[k + 1:] - x) / (t[k + 1:] - t[1:-k])) * value[..., 1:]
        return value

    def forward(self, inputs):
        return torch.einsum("big,oig->bo", self.basis(inputs), self.coefficients) + torch.nn.functional.silu(inputs) @ self.base.T + self.bias


class SplinePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, width=24):
        super().__init__()
        def network(output):
            return nn.Sequential(CubicSplineLayer(state_dim, width), nn.LayerNorm(width), nn.Tanh(), CubicSplineLayer(width, output))
        self.actor, self.critic = network(action_dim), network(1)

    def forward(self, state):
        return self.actor(state), self.critic(state).squeeze(-1)

    def distribution(self, state, action_mask):
        logits, value = self(state)
        return torch.distributions.Categorical(logits=logits.masked_fill(~action_mask, -1e9)), value


def generalized_advantages(rewards, values, gamma=1., lam=.95):
    """Terminal episodes: V after final transition is zero; no bootstrapped truncation."""
    result = []
    advantage, next_value = 0., 0.
    for reward, value in zip(reversed(rewards), reversed(values)):
        advantage = reward + gamma * next_value - value + gamma * lam * advantage
        result.append(advantage)
        next_value = value
    return list(reversed(result))


class StablePPO:
    def __init__(self, policy, actor_lr=1e-4, critic_lr=5e-4, target_kl=.01):
        self.policy = policy
        self.actor = torch.optim.Adam(policy.actor.parameters(), lr=actor_lr)
        self.critic = torch.optim.Adam(policy.critic.parameters(), lr=critic_lr)
        self.target_kl = target_kl

    def collect(self, env, episodes=24):
        rows = []
        objectives = []
        for _ in range(episodes):
            state = env.reset(); episode = []
            while not env.done:
                mask = env.action_mask()
                with torch.no_grad():
                    distribution, value = self.policy.distribution(state[None], mask[None])
                    action = distribution.sample()
                new_state, reward, _ = env.step(int(action.item()))
                episode.append((state, mask, action[0], distribution.log_prob(action)[0],
                                distribution.probs[0], float(value.item()), reward))
                state = new_state
            objectives.append(env.coverage()[2]["objective"])
            advantages = generalized_advantages([x[6] for x in episode], [x[5] for x in episode])
            for row, advantage in zip(episode, advantages):
                rows.append((*row[:5], advantage, advantage + row[5]))
        if not rows:
            raise ValueError("No actionable training episodes")
        batch = [torch.stack([r[i] for r in rows]) for i in range(5)]
        advantages = torch.tensor([r[5] for r in rows])
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        return (*batch, advantages, torch.tensor([r[6] for r in rows]), float(np.mean(objectives)))

    def update(self, batch, entropy_weight=.001, epochs=4):
        states, masks, actions, old_logs, old_probs, advantages, returns, objective = batch
        updates = 0
        for _ in range(epochs):
            distribution, _ = self.policy.distribution(states, masks)
            kl = (old_probs * (old_probs.clamp_min(1e-12).log() - distribution.probs.clamp_min(1e-12).log())).sum(-1).mean()
            if float(kl.detach()) > self.target_kl:
                break
            ratio = (distribution.log_prob(actions) - old_logs).exp()
            loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean() - entropy_weight * distribution.entropy().mean()
            self.actor.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), .5)
            self.actor.step(); updates += 1
        for _ in range(epochs):
            values = self.policy.critic(states).squeeze(-1)
            value_loss = .5 * (values - returns).square().mean()
            self.critic.zero_grad(); value_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), .5)
            self.critic.step()
        with torch.no_grad():
            distribution, _ = self.policy.distribution(states, masks)
            final_kl = (old_probs * (old_probs.clamp_min(1e-12).log() - distribution.probs.clamp_min(1e-12).log())).sum(-1).mean()
        return {"actor_updates": updates, "kl_after": float(final_kl), "entropy": float(distribution.entropy().mean()),
                "value_loss": float(value_loss.detach()), "sampled_objective": objective}


def refine_selection(graph, task, actions, max_steps=8):
    """Deterministic delete/add/swap local search, identical for all policy seeds."""
    env = EvidenceEnvironment(graph, max_steps=max_steps); env.reset(task)
    chosen = frozenset(actions)
    allowed = np.flatnonzero(env.allowed).tolist()
    cache = {}
    def score(indices):
        if indices not in cache:
            selected = np.zeros(graph.action_dim, dtype=np.float32)
            if indices:
                selected[list(indices)] = 1
            cache[indices] = env.coverage(selected)[2]["objective"]
        return cache[indices]
    def feasible(indices):
        names = {graph.herbs[i]["name"] for i in indices}
        return len(indices) <= max_steps and all(not (a in names and b in names) for a, b in env.pairs)
    if not chosen.issubset(allowed) or not feasible(chosen):
        raise ValueError("Initial candidate set violates a hard constraint")
    rounds = 0
    while True:
        neighbors = {chosen - {i} for i in chosen}
        neighbors |= {chosen | {j} for j in allowed if j not in chosen}
        neighbors |= {(chosen - {i}) | {j} for i in chosen for j in allowed if j not in chosen}
        best, value = chosen, score(chosen)
        for option in sorted(neighbors, key=lambda x: tuple(sorted(x))):
            if feasible(option) and score(option) > value + 1e-7:
                best, value = option, score(option)
        if best == chosen:
            break
        chosen = best; rounds += 1
    return sorted(chosen), {"evaluated_subsets": len(cache), "rounds": rounds}
