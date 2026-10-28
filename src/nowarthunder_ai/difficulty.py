"""Optional skill constraints for v2 training; no runtime policy wrapper.

The game's positive win reward is unchanged. These bounded auxiliary penalties
shape attack pressure and recovery behavior; they are pragmatic controls, not a
formal constrained-policy-optimization guarantee. Eligibility still requires
independent deterministic evaluation of the exported plain MLP.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch.nn import functional as F


CONTROL_DEFAULTS = {
    "version": 1, "masked_features": [], "fire_budget": .5,
    "turn_fire_budget": .2, "pressure_coef": 2., "recovery_coef": .05,
    "emergency_distance": .12, "initial_budget_scale": 1.,
    "min_budget_scale": .25, "max_budget_scale": 2.,
    "controller_rate": .5, "max_log_step": .15,
}


def control_config(value: Mapping | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("difficulty_control must be a mapping")
    unknown = set(value) - set(CONTROL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown difficulty_control keys: {sorted(unknown)}")
    result = {**CONTROL_DEFAULTS, **dict(value)}
    if type(result["version"]) is not int or result["version"] != 1:
        raise ValueError("Unsupported difficulty_control version")
    features = result["masked_features"]
    if not isinstance(features, (list, tuple)) or any(type(i) is not int or not 0 <= i < 320 for i in features):
        raise ValueError("masked_features must contain observation indices in 0..319")
    if any(298 <= i <= 308 for i in features):
        raise ValueError("Legal-action features and personality role cannot be masked")
    result["masked_features"] = sorted(set(features))
    for key in set(CONTROL_DEFAULTS) - {"version", "masked_features"}:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"difficulty_control.{key} must be finite")
        result[key] = float(value)
    if not all(0 <= result[key] <= 1 for key in ("fire_budget", "turn_fire_budget")):
        raise ValueError("Difficulty fire budgets must lie in [0,1]")
    if result["pressure_coef"] <= 0 or result["recovery_coef"] < 0:
        raise ValueError("pressure_coef must be positive and recovery_coef nonnegative")
    if not 0 < result["emergency_distance"] <= 1:
        raise ValueError("emergency_distance must lie in (0,1]")
    if not 0 < result["min_budget_scale"] <= result["initial_budget_scale"] <= result["max_budget_scale"]:
        raise ValueError("Budget scale bounds must contain a positive initial scale")
    if result["controller_rate"] < 0 or not 0 < result["max_log_step"] <= 1:
        raise ValueError("controller_rate must be nonnegative and max_log_step must lie in (0,1]")
    return result


class DifficultyController:
    """Finite multiplicative adjustment, driven only by qualified calibration."""
    def __init__(self, config: Mapping, state: Mapping | None = None):
        self.config = control_config(config)
        assert self.config is not None
        self.budget_scale = self.config["initial_budget_scale"]
        self.updates = 0
        self.last_status = "initialized"
        if state is not None:
            scale = float(state["budget_scale"])
            if not math.isfinite(scale) or not self.config["min_budget_scale"] <= scale <= self.config["max_budget_scale"]:
                raise ValueError("Saved difficulty budget_scale is invalid")
            updates = state.get("updates", 0)
            if type(updates) is not int or updates < 0:
                raise ValueError("Saved difficulty update count is invalid")
            self.budget_scale, self.updates = scale, updates
            self.last_status = str(state.get("last_status", "restored"))

    @property
    def effective_budgets(self) -> dict[str, float]:
        return {key: min(1., max(0., self.config[key] * self.budget_scale))
                for key in ("fire_budget", "turn_fire_budget")}

    def state_dict(self) -> dict:
        return {"budget_scale": self.budget_scale, "updates": self.updates,
                "last_status": self.last_status}

    def metrics(self) -> dict:
        return {"difficulty_budget_scale": self.budget_scale,
                "difficulty_controller_updates": self.updates,
                "difficulty_controller_status": self.last_status,
                **{f"difficulty_{key}": value for key, value in self.effective_budgets.items()}}

    def update(self, assessment: Mapping) -> dict:
        if not assessment.get("quality_passed", False):
            self.last_status = "frozen_quality_failed"
            return self.metrics()
        rate = float(assessment["defender_win_rate"])
        low, high = (float(v) for v in assessment["target_defender_win_rate"])
        if not all(math.isfinite(v) for v in (rate, low, high)) or not 0 <= low <= high <= 1 or not 0 <= rate <= 1:
            raise ValueError("Controller calibration rates must be finite probabilities")
        if low <= rate <= high:
            self.last_status = "in_band"
            return self.metrics()
        step = self.config["controller_rate"] * (rate - (low + high) / 2)
        step = min(self.config["max_log_step"], max(-self.config["max_log_step"], step))
        next_scale = min(self.config["max_budget_scale"], max(self.config["min_budget_scale"],
                         self.budget_scale * math.exp(step)))
        current_budgets = self.effective_budgets
        def budgets_at(scale):
            return {key: min(1., max(0., self.config[key] * scale)) for key in current_budgets}
        def unchanged(scale):
            return all(math.isclose(value, current_budgets[key], rel_tol=0, abs_tol=1e-12)
                       for key, value in budgets_at(scale).items())
        # Old checkpoints can contain latent scale above the point where both
        # budgets saturated. A reducing step must cross that useless plateau;
        # increasing it must not claim an effective controller update.
        if step < 0 and unchanged(next_scale):
            positive = [self.config[key] for key in current_budgets if self.config[key] > 0]
            if positive and all(current_budgets[key] == 1 for key in current_budgets if self.config[key] > 0):
                useful_scale = max(1 / base for base in positive)
                next_scale = max(self.config["min_budget_scale"], min(self.budget_scale, useful_scale) * math.exp(step))
        if next_scale == self.budget_scale:
            self.last_status = "at_bound"
        elif unchanged(next_scale):
            self.last_status = "at_effective_bound"
        else:
            self.budget_scale = next_scale
            self.updates += 1
            self.last_status = "reduce_pressure" if rate < low else "increase_pressure"
        return self.metrics()


def difficulty_actor_loss(logits: torch.Tensor, observation: torch.Tensor,
                          action_mask: torch.Tensor, config: Mapping,
                          effective_budgets: Mapping, *,
                          recovery_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batch-mean pressure constraints and a movement-marginal recovery CE.

    Only legal shots at visible targets count as pressure. Brick clearing and
    close-range self-defense are exempt. A mean-budget hinge permits strong
    responses in individual states; it does not cap every state's fire chance.
    Recovery CE sums both legal fire variants for its chosen move direction.
    """
    masked_logits = logits.masked_fill(~action_mask, -1e9)
    log_probability = F.log_softmax(masked_logits, dim=-1)
    probability = log_probability.exp()
    aim = observation[:, 313:317]
    facing = observation[:, 2:6].argmax(dim=-1)
    facing_aim = aim.gather(1, facing.unsqueeze(-1))
    action_aim = torch.cat((facing_aim.expand(-1, 2), aim.repeat_interleave(2, dim=-1)), dim=-1)
    moves = torch.arange(10, device=logits.device) // 2
    is_fire = torch.arange(10, device=logits.device).remainder(2).bool()
    emergency = ((observation[:, 217] > .5)
                 & (observation[:, 208:210].square().sum(dim=-1) < float(config["emergency_distance"])**2))
    combat = (action_mask & is_fire & (action_aim >= .8) & ~emergency.unsqueeze(-1))
    turn_combat = combat & (moves > 0) & ((moves - 1) != facing.unsqueeze(-1))
    opportunities = combat.any(dim=-1)
    turn_opportunities = turn_combat.any(dim=-1)
    fire_pressure = (probability * combat).sum(dim=-1)
    turn_pressure = (probability * turn_combat).sum(dim=-1)

    def conditional_mean(values, selected):
        return (values * selected).sum() / selected.sum().clamp_min(1)

    mean_fire = conditional_mean(fire_pressure, opportunities)
    mean_turn = conditional_mean(turn_pressure, turn_opportunities)
    fire_excess = F.relu(mean_fire - float(effective_budgets["fire_budget"]))
    turn_excess = F.relu(mean_turn - float(effective_budgets["turn_fire_budget"]))
    pressure_loss = float(config["pressure_coef"]) * (fire_excess.square() + turn_excess.square())

    stuck = observation[:, 317] > .35
    previous_move = (observation[:, 8] * 4).round().long().clamp(0, 4)
    idle_without_target = (previous_move == 0) & (aim.max(dim=-1).values < .1)
    recover = stuck | idle_without_target
    temporal = torch.zeros_like(recover)
    if recovery_mask is not None:
        if recovery_mask.shape != recover.shape:
            raise ValueError("Temporal recovery mask must match the actor batch")
        temporal = recovery_mask.to(dtype=torch.bool, device=logits.device)
        recover = recover | temporal
    role = (observation[:, 308] * 3).round().long()
    use_base = ((role == 0) | (observation[:, 217] < .5)
                | ((role == 3) & (observation[:, 208:210].abs().sum(dim=-1) > .32)))
    navigation = torch.where(use_base.unsqueeze(-1), observation[:, 294:298], observation[:, 290:294])
    direction_legal = action_mask[:, 2:].reshape(-1, 4, 2).any(dim=-1)
    feasible = direction_legal & (navigation < .999)
    direction = torch.arange(4, device=logits.device)
    alternatives = feasible & (direction != (previous_move - 1).unsqueeze(-1))
    feasible = torch.where((stuck & alternatives.any(dim=-1)).unsqueeze(-1), alternatives, feasible)
    recover = recover & feasible.any(dim=-1)
    target = navigation.masked_fill(~feasible, 10.).argmin(dim=-1)
    direction_logp = torch.logsumexp(log_probability[:, 2:].reshape(-1, 4, 2), dim=-1)
    recovery_ce = -direction_logp.gather(1, target.unsqueeze(-1)).squeeze(-1)
    recovery_loss = float(config["recovery_coef"]) * conditional_mean(recovery_ce, recover)
    return pressure_loss + recovery_loss, {
        "loss_difficulty_pressure": pressure_loss.detach(),
        "loss_difficulty_recovery": recovery_loss.detach(),
        "difficulty_fire_pressure": mean_fire.detach(),
        "difficulty_turn_fire_pressure": mean_turn.detach(),
        "difficulty_combat_opportunities": opportunities.sum().detach().float(),
        "difficulty_turn_opportunities": turn_opportunities.sum().detach().float(),
        "difficulty_recovery_samples": recover.sum().detach().float(),
        "difficulty_temporal_recovery_samples": (temporal & feasible.any(dim=-1)).sum().detach().float(),
    }
