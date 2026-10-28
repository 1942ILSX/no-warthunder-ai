"""Bounded historical opponents, plus scripted and frozen latest policies."""
from __future__ import annotations

import random

from .checkpoint import cpu_state
from .models import PolicyNetwork, frozen_copy


class OpponentPool:
    def __init__(self, capacity: int = 8):
        if capacity < 1:
            raise ValueError("league_size must be positive")
        self.capacity = capacity
        self.entries: dict[str, list[dict]] = {"enemy": [], "defender": []}

    def add(self, role: str, policy: PolicyNetwork, iteration: int) -> None:
        self.entries[role].append({"iteration": iteration, "hidden_sizes": policy.hidden_sizes,
                                   "state": cpu_state(policy)})
        del self.entries[role][:-self.capacity]

    def sample(self, role: str, latest: PolicyNetwork, *, scripted_probability: float,
               latest_probability: float, device) -> tuple[PolicyNetwork | None, str]:
        draw = random.random()
        if draw < scripted_probability:
            return None, "scripted"
        entries = self.entries[role]
        if draw < scripted_probability + latest_probability or not entries:
            return frozen_copy(latest), "latest"
        entry = random.choice(entries)
        actor = PolicyNetwork(entry["hidden_sizes"], latest.observation_size, latest.action_size).to(device)
        actor.load_state_dict(entry["state"])
        actor.eval().requires_grad_(False)
        return actor, f"history:{entry['iteration']}"

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "entries": self.entries}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.entries = state["entries"]
