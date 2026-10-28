"""Exportable feed-forward policies and training-only centralized critics.

The 320 observation features are passed through unchanged: in particular, the
enemy role feature is never normalized away or replaced by a slot index.
"""
from __future__ import annotations

from collections.abc import Sequence
import copy

import numpy as np
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MaskedCategorical, ProbabilisticActor


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes: Sequence[int] = (256, 128),
                 observation_size: int = 320, action_size: int = 10):
        super().__init__()
        self.hidden_sizes = tuple(int(n) for n in hidden_sizes)
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        # Nonpersistent keeps legacy state_dicts unchanged. Zero first-layer
        # columns make the same handicap intrinsic to the exported Go MLP.
        self.register_buffer("input_gate", torch.ones(self.observation_size), persistent=False)
        self.masked_features: tuple[int, ...] = ()
        if not self.hidden_sizes or min(self.hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain positive widths")
        layers: list[nn.Module] = []
        dims = (self.observation_size, *self.hidden_sizes, self.action_size)
        for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            linear = nn.Linear(left, right)
            nn.init.orthogonal_(linear.weight, gain=0.01 if i == len(dims) - 2 else 2**0.5)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation * self.input_gate)

    @torch.no_grad()
    def set_masked_features(self, indices: Sequence[int]) -> None:
        if any(type(index) is not int or not 0 <= index < self.observation_size for index in indices):
            raise ValueError("Invalid masked observation feature")
        if self.observation_size == 320 and any(298 <= index <= 308 for index in indices):
            raise ValueError("Legal-action features and personality role cannot be masked")
        self.masked_features = tuple(sorted(set(indices)))
        self.input_gate.fill_(1)
        if self.masked_features:
            self.input_gate[list(self.masked_features)] = 0
        self.enforce_input_gate()

    @torch.no_grad()
    def enforce_input_gate(self) -> None:
        self.net[0].weight.mul_(self.input_gate.unsqueeze(0))


class CentralCritic(nn.Module):
    """One value per room; never one return per enemy lifespan."""
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_sizes = tuple(int(n) for n in hidden_sizes)
        dims = (self.input_size, *self.hidden_sizes, 1)
        layers: list[nn.Module] = []
        for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def probabilistic_actor(policy: PolicyNetwork) -> ProbabilisticActor:
    """TorchRL owns the categorical distribution and PPO likelihood ratio."""
    module = TensorDictModule(policy, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        module=module, in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"], distribution_class=MaskedCategorical,
        return_log_prob=True, log_prob_key="sample_log_prob",
    )


def frozen_copy(policy: PolicyNetwork) -> PolicyNetwork:
    clone = copy.deepcopy(policy).eval()
    clone.requires_grad_(False)
    return clone


@torch.no_grad()
def act(policy: PolicyNetwork | None, observation: np.ndarray,
        mask: np.ndarray, active: np.ndarray, device: torch.device | str,
        *, deterministic: bool = False) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """One batched inference, including all rooms/slots; inactive -> Go fallback.

    All-false masks are only repaired for inactive slots. An active slot without
    a legal action indicates a protocol bug and is rejected before inference.
    Returned tensors stay on the device for rollout assembly, with one batched
    CPU action transfer to the Go bridge (no synchronization per tank).
    """
    shape = active.shape
    if policy is None:
        return (np.full(shape, -1, dtype=np.int64),
                torch.zeros(shape, device=device, dtype=torch.int64),
                torch.zeros(shape, device=device))
    if np.any(active & ~mask.any(axis=-1)):
        raise ValueError("Active policy slot has no legal action")
    safe_mask = np.array(mask, dtype=np.bool_, copy=True)
    safe_mask[~active] = False
    safe_mask[..., 0] |= ~active
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(safe_mask, dtype=torch.bool, device=device)
    dist = MaskedCategorical(logits=policy(obs), mask=mask_tensor)
    actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    log_probs = dist.log_prob(actions)
    sent = actions.cpu().numpy().copy()
    sent[~active] = -1
    return sent, actions.detach(), log_probs.detach()
