"""Conservative action overrides; the network margin is not a probability."""
import torch

from .planning_v4 import greedy_action


def gated_rollout(env, policy=None, margin=None):
    env.reset(); overrides = 0; decisions = 0
    while not env.done:
        greedy = greedy_action(env); action = greedy; decisions += 1
        if policy is not None and margin is not None:
            obs, features, mask = env.state()
            with torch.no_grad():
                scores = policy.actor(obs[None], features[None])[0].masked_fill(~mask, -1e9)
            proposed = int(scores.argmax())
            if proposed != greedy and float(scores[proposed]-scores[greedy]) > margin:
                action = proposed; overrides += 1
        env.step(action)
    return env.selected, env.score(env.selected)[0], {'overrides': overrides, 'decisions': decisions}
