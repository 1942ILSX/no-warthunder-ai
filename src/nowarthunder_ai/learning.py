"""TorchRL PPO + room-level GAE, with explicit multi-agent sample boundaries."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .models import CentralCritic, PolicyNetwork, probabilistic_actor
from .difficulty import DifficultyController, difficulty_actor_loss


@dataclass
class Rollout:
    """All leading dimensions are [room, time]; slots are a feature dimension."""
    observation: torch.Tensor
    action_mask: torch.Tensor
    active: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    critic_observation: torch.Tensor
    next_critic_observation: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    recovery_mask: torch.Tensor | None = None


@torch.no_grad()
def room_returns(rollout: Rollout, critic: CentralCritic, *, gamma: float,
                 gae_lambda: float, minibatch_size: int = 1024) -> TensorDict:
    """Bootstrap final observations on truncation, but stop the GAE recursion.

    Only room termination enters these signals. An enemy death, respawn, slot
    reuse or policy-delay barrier does NOT terminate a room value trajectory.
    """
    size = rollout.reward.shape[:2]
    def values(observation: torch.Tensor) -> torch.Tensor:
        flat = observation.reshape(-1, observation.shape[-1])
        return torch.cat([critic(chunk) for chunk in flat.split(minibatch_size)]).reshape(*size, 1)
    td = TensorDict({
        "state_value": values(rollout.critic_observation),
        "next": {
            "state_value": values(rollout.next_critic_observation),
            "reward": rollout.reward.unsqueeze(-1),
            "done": (rollout.terminated | rollout.truncated).unsqueeze(-1),
            "terminated": rollout.terminated.unsqueeze(-1),
        },
    }, batch_size=size, device=rollout.reward.device)
    # Explicit non-vectorized implementation works on CPU/CUDA/MPS; TorchRL
    # still owns the GAE math. It avoids accelerator-incompatible convolution
    # paths and preserves terminal/truncation semantics without auto-reset.
    GAE(gamma=gamma, lmbda=gae_lambda, value_network=None,
        vectorized=False, average_gae=False)(td, time_dim=1)
    return td


def actor_samples(rollout: Rollout, returns: TensorDict) -> TensorDict:
    """Fan room advantages out to active decisions, never to dormant slots."""
    advantage = returns["advantage"]
    if rollout.active.ndim == 3:  # shared enemy policy, [room,time,slot]
        advantage = advantage.unsqueeze(-2).expand(*rollout.active.shape, 1)
    valid = rollout.active
    values = {
        "observation": rollout.observation[valid],
        "action_mask": rollout.action_mask[valid],
        "action": rollout.action[valid],
        "sample_log_prob": rollout.log_prob[valid],
        "advantage": advantage[valid],
    }
    if rollout.recovery_mask is not None:
        if rollout.recovery_mask.shape != valid.shape:
            raise ValueError("Temporal recovery mask must match actor activity shape")
        values["recovery_mask"] = rollout.recovery_mask[valid]
    return TensorDict(values, batch_size=[int(valid.sum().item())], device=rollout.reward.device)


class PPOLearner:
    def __init__(self, actor: PolicyNetwork, critic: CentralCritic,
                 config: Mapping, difficulty_controller: DifficultyController | None = None):
        self.actor, self.critic = actor, critic
        self.config = config
        self.difficulty_controller = difficulty_controller
        lr = float(config.get("learning_rate", 3e-4))
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        # Value learning is batched separately below. Including it on each
        # enemy actor sample would overweight rooms with more living tanks and
        # omit room transitions that have no active enemies.
        self.loss = ClipPPOLoss(
            actor_network=probabilistic_actor(actor), critic_network=None,
            functional=False, clip_epsilon=float(config.get("clip_epsilon", .2)),
            entropy_coeff=float(config.get("entropy_coef", .01)),
            critic_coeff=None, normalize_advantage=True,
        )

    def update(self, rollout: Rollout) -> dict[str, float]:
        config = self.config
        batch_size = int(config.get("minibatch_size", 1024))
        critic_batch_size = int(config.get("critic_minibatch_size", batch_size))
        returns = room_returns(rollout, self.critic,
                               gamma=float(config.get("gamma", .999)),
                               gae_lambda=float(config.get("gae_lambda", .95)),
                               minibatch_size=critic_batch_size)
        samples = actor_samples(rollout, returns)
        critic_obs = rollout.critic_observation.flatten(0, 1)
        targets = returns["value_target"].flatten(0, 1)
        max_norm = float(config.get("max_grad_norm", .5))
        metrics: dict[str, list[torch.Tensor]] = {}
        actor_steps, critic_steps = 0, 0
        for _ in range(int(config.get("ppo_epochs", 4))):
            permutation = torch.randperm(len(samples), device=critic_obs.device)
            for indices in permutation.split(batch_size):
                if not len(indices):
                    continue
                losses = self.loss(samples[indices])
                loss = losses["loss_objective"] + losses["loss_entropy"]
                if self.difficulty_controller is not None:
                    batch = samples[indices]
                    auxiliary, diagnostic = difficulty_actor_loss(
                        self.actor(batch["observation"]), batch["observation"], batch["action_mask"],
                        self.difficulty_controller.config, self.difficulty_controller.effective_budgets,
                        recovery_mask=batch.get("recovery_mask", None))
                    loss = loss + auxiliary
                    for key, value in diagnostic.items():
                        metrics.setdefault(key, []).append(value)
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.actor.enforce_input_gate()
                actor_steps += 1
                for key in ("loss_objective", "loss_entropy", "entropy", "clip_fraction", "kl_approx"):
                    metrics.setdefault(key, []).append(losses[key].detach())
            for indices in torch.randperm(len(critic_obs), device=critic_obs.device).split(critic_batch_size):
                value_loss = F.smooth_l1_loss(self.critic(critic_obs[indices]), targets[indices])
                self.critic_optimizer.zero_grad(set_to_none=True)
                (float(config.get("value_coef", .5)) * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()
                critic_steps += 1
                metrics.setdefault("loss_critic", []).append(value_loss.detach())
        # Synchronize metrics once per update, not per minibatch/tank.
        keys = list(metrics)
        means = torch.stack([torch.stack(metrics[k]).mean() for k in keys]).cpu().tolist()
        result = dict(zip(keys, means))
        result.update(actor_samples=float(len(samples)), room_samples=float(len(critic_obs)),
                      actor_optimizer_steps=float(actor_steps), critic_optimizer_steps=float(critic_steps))
        return result

    def optimizer_state(self) -> dict:
        return {"actor": self.actor_optimizer.state_dict(), "critic": self.critic_optimizer.state_dict()}

    def load_optimizer_state(self, state: dict) -> None:
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])
