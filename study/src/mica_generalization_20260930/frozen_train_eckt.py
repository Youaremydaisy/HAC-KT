#!/usr/bin/env python3
"""Train ECKT-v2.1: an equation-based, from-scratch cognitive/affect proxy KT model.

The model does not load AKT, QIKT, DKT, or any other KT checkpoint.  It keeps a
separate mastery logit for every concept, applies a positive evidence update and
time decay after each observed response, and adds four named behavior-derived
affect proxy contributions.  A Cui-style recent-innovation gate can hold the
affect state while cognitive mastery continues to update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_model import WindowDataset


STATE_NAMES = ("engagement", "confusion", "frustration", "boredom")
DETECTOR_TO_STATE = (2, 1, 0, 3)  # concentrating, confused, frustrated, bored


def revisit_gaps(skills: np.ndarray, logged_gaps: np.ndarray, num_skills: int) -> tuple[np.ndarray, np.ndarray]:
    """Same-skill recency at the observed time; never use the next event time."""
    times = np.cumsum(np.expm1(np.clip(logged_gaps.astype(np.float64), 0, 20)))
    previous = np.full(num_skills, -1, dtype=np.int64)
    current_gap = np.zeros(len(skills), dtype=np.float32)
    target_gap = np.zeros(len(skills), dtype=np.float32)
    for i, skill in enumerate(skills):
        old = previous[skill]
        if old >= 0:
            current_gap[i] = np.log1p(times[i] - times[old])
        previous[skill] = i
        if i + 1 < len(skills):
            old = previous[skills[i + 1]]
            if old >= 0:
                target_gap[i] = np.log1p(times[i] - times[old])
    return current_gap, target_gap


class ECKTWindowDataset(WindowDataset):
    """WindowDataset plus causal state carried from earlier windows."""

    def __init__(self, data_dir: Path, split: str, max_seq_len: int, num_skills: int, include_revisit: bool = False) -> None:
        super().__init__(data_dir, split, max_seq_len)
        self.num_skills = int(num_skills)
        self.revisit_current = None
        self.revisit_target = None
        if include_revisit:
            self.revisit_current = np.zeros(len(self.skills), dtype=np.float32)
            self.revisit_target = np.zeros(len(self.skills), dtype=np.float32)
            for a, b in zip(self.offsets[:-1], self.offsets[1:]):
                current, target = revisit_gaps(self.skills[a:b], self.delta[a:b], self.num_skills)
                self.revisit_current[a:b] = current
                self.revisit_target[a:b] = target

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = super().__getitem__(index)
        sequence_index, start, _ = (int(x) for x in self.windows[index])
        initial_mastery = np.zeros(self.num_skills, dtype=np.float32)
        initial_affect = np.zeros(4, dtype=np.float32)
        counts = np.zeros(self.num_skills, dtype=np.float64)
        correct_sums = np.zeros(self.num_skills, dtype=np.float64)
        has_history = start > 0
        if has_history:
            base = int(self.offsets[sequence_index])
            prefix_skills = np.asarray(self.skills[base:base + start], dtype=np.int64)
            prefix_correct = np.asarray(self.correct[base:base + start], dtype=np.float64)
            counts = np.bincount(prefix_skills, minlength=self.num_skills).astype(np.float64)
            correct_sums = np.bincount(prefix_skills, weights=prefix_correct, minlength=self.num_skills)
            prior = (correct_sums + 1.0) / (counts + 2.0)
            initial_mastery = np.clip(np.log(prior / (1.0 - prior)), -4.0, 4.0).astype(np.float32)
            if self.behavior_affects is not None:
                prior_affect = np.asarray(self.behavior_affects[base:base + start], dtype=np.float32).mean(axis=0)
                prior_affect = np.clip(prior_affect, 1e-4, 1.0 - 1e-4)
                initial_affect = np.log(prior_affect / (1.0 - prior_affect)).astype(np.float32)
        item["initial_mastery"] = torch.from_numpy(initial_mastery)
        item["initial_affect"] = torch.from_numpy(initial_affect)
        item["has_history"] = torch.tensor(float(has_history), dtype=torch.float32)
        item["window_index"] = torch.tensor(index, dtype=torch.long)
        item["initial_success_counts"] = torch.from_numpy(correct_sums.astype(np.float32))
        item["initial_failure_counts"] = torch.from_numpy((counts - correct_sums).astype(np.float32))
        if self.revisit_current is not None:
            base = int(self.offsets[sequence_index]) + start
            n = int(item["mask"].sum())
            for key, values in [("revisit_delta", self.revisit_current), ("target_revisit_delta", self.revisit_target)]:
                padded = np.zeros(self.max_steps, dtype=np.float32)
                padded[:n] = values[base:base + n]
                item[key] = torch.from_numpy(padded)
        if has_history:
            global_prior = float((prefix_correct.sum() + 1.0) / (len(prefix_correct) + 2.0))
            initial_ability = float(np.clip(np.log(global_prior / (1.0 - global_prior)), -4.0, 4.0))
        else:
            initial_ability = 0.0
        item["initial_ability"] = torch.tensor(initial_ability, dtype=torch.float32)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ECKT(nn.Module):
    """Explicit coupled state equations with exact additive explanations."""

    def __init__(
        self,
        num_skills: int,
        num_problems: int,
        behavior_dim: int = 8,
        mode: str = "affect_always",
        history: int = 4,
        event_threshold: float = 0.15,
        event_temperature: float = 0.08,
        affect_max: float = 0.75,
        periodic_period: int = 10,
        random_rate: float = 0.10,
        use_affect_gain: bool = False,
        random_seed: int = 42,
        affect_target: str = "behavior",
        use_discrimination: bool = False,
        response_mode: str = "linear",
        weight_mode: str = "linear",
        weight_tau: float = 2.0,
        gaussian_uncertainty: bool = False,
        behavior_gain: bool = False,
        behavior_gain_hinge: bool = False,
        affect_forgetting: bool = False,
        pfa_signed_learning: bool = False,
        affect_difficulty_interaction: bool = False,
        multiscale_affect: bool = False,
        student_ability: bool = False,
        normalize_behavior: bool = False,
        smooth_behavior_gain: bool = False,
        fixed_learning_gain: float = 1.0,
        practice_readout: bool = False,
        revisit_forgetting: bool = False,
        hawkes_readout: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"cognitive", "affect_always", "affect_periodic", "affect_random", "affect_event"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.num_skills = int(num_skills)
        self.num_problems = max(1, int(num_problems))
        self.behavior_dim = int(behavior_dim)
        self.history = max(1, int(history))
        self.event_threshold = float(event_threshold)
        self.event_temperature = float(event_temperature)
        self.affect_max = float(affect_max)
        self.periodic_period = max(1, int(periodic_period))
        self.random_rate = float(random_rate)
        self.use_affect_gain = bool(use_affect_gain)
        self.random_seed = int(random_seed)
        if affect_target not in {"behavior", "detector"}:
            raise ValueError(f"unknown affect_target: {affect_target}")
        self.affect_target = affect_target
        if response_mode not in {"linear", "hinge"}:
            raise ValueError(f"unknown response_mode: {response_mode}")
        self.use_discrimination = bool(use_discrimination)
        self.response_mode = response_mode
        if weight_mode not in {"linear", "exponential", "gaussian", "dynamic"}:
            raise ValueError(f"unknown weight_mode: {weight_mode}")
        self.weight_mode = weight_mode
        self.weight_tau = max(float(weight_tau), 1e-3)
        self.gaussian_uncertainty = bool(gaussian_uncertainty)
        self.behavior_gain = bool(behavior_gain)
        self.behavior_gain_hinge = bool(behavior_gain_hinge)
        self.affect_forgetting = bool(affect_forgetting)
        self.pfa_signed_learning = bool(pfa_signed_learning)
        self.affect_difficulty_interaction = bool(affect_difficulty_interaction)
        self.multiscale_affect = bool(multiscale_affect)
        self.student_ability = bool(student_ability)
        self.normalize_behavior = bool(normalize_behavior)
        self.smooth_behavior_gain = bool(smooth_behavior_gain)
        self.fixed_learning_gain = float(fixed_learning_gain)
        self.practice_readout = bool(practice_readout)
        self.revisit_forgetting = bool(revisit_forgetting)
        self.hawkes_readout = bool(hawkes_readout)
        self.cognitive_part_names = ["intercept", "mastery", "skill_bias", "difficulty"]
        if self.student_ability:
            self.cognitive_part_names.append("student_ability")
        if self.practice_readout:
            self.cognitive_part_names += ["success_history", "failure_history"]
        if self.hawkes_readout:
            self.cognitive_part_names += ["correct_cross_effect", "incorrect_cross_effect"]
        if not 0.5 <= self.fixed_learning_gain <= 1.5:
            raise ValueError("fixed_learning_gain must be in [0.5, 1.5]")
        if self.normalize_behavior:
            self.register_buffer("behavior_center", torch.zeros(behavior_dim))
            self.register_buffer("behavior_scale", torch.ones(behavior_dim))

        # Cognitive state parameters.  u[k] is a concept mastery logit.
        self.initial_mastery = nn.Parameter(torch.zeros(num_skills))
        self.practice_raw = nn.Parameter(torch.zeros(num_skills, 2)) if self.practice_readout else None
        self.cross_effect_raw = nn.Parameter(torch.zeros(2, num_skills, num_skills)) if self.hawkes_readout else None
        self.cross_decay_raw = nn.Parameter(torch.tensor(-0.5)) if self.hawkes_readout else None
        self.skill_bias = nn.Parameter(torch.zeros(num_skills))
        self.learning_raw = nn.Parameter(torch.full((num_skills,), -1.5))
        self.learning_pos_raw = nn.Parameter(torch.full((num_skills,), -1.5)) if self.pfa_signed_learning else None
        self.learning_neg_raw = nn.Parameter(torch.full((num_skills,), -1.5)) if self.pfa_signed_learning else None
        self.forgetting_raw = nn.Parameter(torch.full((num_skills,), -4.0))
        self.affect_forgetting_raw = nn.Parameter(torch.zeros(4)) if self.affect_forgetting else None
        self.item_difficulty = nn.Parameter(torch.zeros(self.num_problems))
        self.item_discrimination_raw = nn.Parameter(torch.zeros(self.num_problems)) if self.use_discrimination else None
        self.intercept = nn.Parameter(torch.tensor(0.0))
        self.mastery_scale_raw = nn.Parameter(torch.tensor(1.5))
        self.difficulty_scale_raw = nn.Parameter(torch.tensor(1.0))
        self.ability_scale_raw = nn.Parameter(torch.tensor(0.5)) if self.student_ability else None

        # Affect state equation in logit space:
        # a*_t = persistence * a_{t-1} + B [behavior_t, response_t] + b.
        self.initial_affect = nn.Parameter(torch.zeros(4))
        self.affect_persistence_raw = nn.Parameter(torch.zeros(4))
        affect_input_dim = behavior_dim + 1 + (4 if self.multiscale_affect else 0)
        self.affect_behavior = nn.Parameter(torch.zeros(4, affect_input_dim))
        self.affect_bias = nn.Parameter(torch.zeros(4))
        self.affect_coeff_raw = nn.Parameter(torch.zeros(4))
        self.affect_hinge_raw = nn.Parameter(torch.zeros(4, 3))
        self.affect_difficulty_raw = nn.Parameter(torch.zeros(4)) if self.affect_difficulty_interaction else None
        self.short_affect_decay_raw = nn.Parameter(torch.zeros(4)) if self.multiscale_affect else None
        self.gain_affect_raw = nn.Parameter(torch.zeros(4))
        self.affect_logvar = nn.Parameter(torch.zeros(4)) if self.gaussian_uncertainty else None
        self.behavior_gain_raw = nn.Parameter(torch.zeros(behavior_dim + 1)) if self.behavior_gain else None
        self.behavior_gain_hinge_raw = nn.Parameter(torch.zeros(behavior_dim + 1, 3)) if self.behavior_gain and self.behavior_gain_hinge else None

    def _bounded_behavior(self, features: Tensor, correct: Tensor) -> Tensor:
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=5.0, neginf=-5.0)
        if self.normalize_behavior:
            features = (features - self.behavior_center) / self.behavior_scale
        # Features come from logs and empirical-Bayes proxies; tanh prevents a
        # single very long delay from dominating the state equation.
        return torch.tanh(torch.cat((features, correct.float().unsqueeze(-1)), dim=-1))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        skill = batch["skill"].long().clamp(0, self.num_skills - 1)
        target_skill = batch["target_skill"].long().clamp(0, self.num_skills - 1)
        problem = batch["problem"].long().clamp(0, self.num_problems - 1)
        target_problem = batch["target_problem"].long().clamp(0, self.num_problems - 1)
        correct = batch["correct"].float()
        target_correct = batch["target_correct"].float()
        mask = batch["mask"].float()
        behavior = batch["behavior_features"].float()
        behavior_affect = batch["behavior_affect"].float().clamp(0.0, 1.0)
        delta = batch["delta"].float().clamp(0.0, 20.0)
        batch_size, steps = skill.shape
        rows = torch.arange(batch_size, device=skill.device)

        global_mastery = self.initial_mastery.unsqueeze(0).expand(batch_size, -1)
        global_affect = self.initial_affect.unsqueeze(0).expand(batch_size, -1)
        has_history = batch.get("has_history", torch.zeros(batch_size, device=skill.device)).bool()
        prefix_mastery = batch.get("initial_mastery", torch.zeros_like(global_mastery)).to(global_mastery.dtype)
        prefix_affect = batch.get("initial_affect", torch.zeros_like(global_affect)).to(global_affect.dtype)
        initial_ability = batch.get("initial_ability", torch.zeros(batch_size, device=skill.device)).to(global_mastery.dtype)
        mastery = torch.where(has_history[:, None], prefix_mastery, global_mastery).clone()
        affect_logit = torch.where(has_history[:, None], prefix_affect, global_affect).clone()
        innovation_history = torch.zeros(batch_size, self.history, device=skill.device)
        cognitive_logits: list[Tensor] = []
        full_logits: list[Tensor] = []
        affect_states: list[Tensor] = []
        affect_contribs: list[Tensor] = []
        gates: list[Tensor] = []
        innovations: list[Tensor] = []
        scores: list[Tensor] = []
        cognitive_parts: list[Tensor] = []
        gain_coeffs: list[Tensor] = []
        affect_gain_parts: list[Tensor] = []
        affect_gain_raws: list[Tensor] = []
        affect_gain_bases: list[Tensor] = []
        short_affect_emas: list[Tensor] = []

        mastery_scale = F.softplus(self.mastery_scale_raw) + 1e-4
        difficulty_scale = F.softplus(self.difficulty_scale_raw) + 1e-4
        learning_rate = F.softplus(self.learning_raw)
        learning_pos = F.softplus(self.learning_pos_raw) if self.learning_pos_raw is not None else None
        learning_neg = F.softplus(self.learning_neg_raw) if self.learning_neg_raw is not None else None
        forgetting_rate = F.softplus(self.forgetting_raw)
        persistence = torch.sigmoid(self.affect_persistence_raw)
        affect_coeff = self.affect_max * torch.tanh(self.affect_coeff_raw)
        # Unknown/padding item has zero difficulty by construction.
        item_difficulty = self.item_difficulty - self.item_difficulty[0]
        item_discrimination = (0.5 + F.softplus(self.item_discrimination_raw)) if self.item_discrimination_raw is not None else None
        # Centering removes the intercept/skill-bias location ambiguity and
        # makes the reported skill contribution comparable across runs.
        skill_bias = self.skill_bias - self.skill_bias.mean()
        ability_scale = F.softplus(self.ability_scale_raw) if self.ability_scale_raw is not None else None
        short_affect_ema = torch.zeros(batch_size, 4, device=skill.device)
        short_affect_decay = torch.sigmoid(self.short_affect_decay_raw) if self.short_affect_decay_raw is not None else None
        successes = batch.get("initial_success_counts", torch.zeros_like(mastery)).clone()
        failures = batch.get("initial_failure_counts", torch.zeros_like(mastery)).clone()
        practice_coeff = 0.5 * torch.tanh(self.practice_raw) if self.practice_raw is not None else None
        if self.hawkes_readout:
            correct_trace = torch.zeros_like(mastery)
            incorrect_trace = torch.zeros_like(mastery)
            cross_coeff = 0.75 * torch.tanh(self.cross_effect_raw)
            cross_decay = F.softplus(self.cross_decay_raw)

        for t in range(steps):
            active = mask[:, t] > 0.5
            cur_skill = skill[:, t]
            tgt_skill = target_skill[:, t]
            cur_problem = problem[:, t]
            tgt_problem = target_problem[:, t]

            # Forget only the skill revisited at this interaction.  This is an
            # explicit, cheap approximation to a time-decay mastery model.
            current_u = mastery.gather(1, cur_skill[:, None]).squeeze(1)
            previous_affect = torch.sigmoid(affect_logit)
            if self.affect_forgetting:
                forgetting_multiplier = (1.0 + 0.5 * (torch.sigmoid(self.affect_forgetting_raw) @ (previous_affect - 0.5).T)).clamp(0.5, 1.5)
            else:
                forgetting_multiplier = torch.ones(batch_size, device=skill.device)
            elapsed = batch["revisit_delta"][:, t].clamp(0, 20) if self.revisit_forgetting else delta[:, t]
            decay = torch.exp(-forgetting_rate[cur_skill] * forgetting_multiplier * elapsed / 10.0)
            current_u = current_u * decay
            current_mastery = torch.sigmoid(current_u)
            current_cognitive_effect = mastery_scale * (current_mastery - 0.5) - difficulty_scale * item_difficulty[cur_problem]
            if item_discrimination is not None:
                current_cognitive_effect = item_discrimination[cur_problem] * current_cognitive_effect
            current_cog_logit = self.intercept + skill_bias[cur_skill] + current_cognitive_effect
            if self.hawkes_readout:
                # Window-local Hawkes-inspired sufficient statistics. A shared
                # positive decay gives an exponential kernel in elapsed days.
                fade = torch.exp(-cross_decay * torch.expm1(delta[:, t]) / 86400.0)
                correct_trace = correct_trace * torch.where(active, fade, torch.ones_like(fade))[:, None]
                incorrect_trace = incorrect_trace * torch.where(active, fade, torch.ones_like(fade))[:, None]
                support = 1 + correct_trace.sum(-1) + incorrect_trace.sum(-1)
                current_cog_logit = current_cog_logit + (
                    (correct_trace * cross_coeff[0, :, cur_skill].T).sum(-1)
                    + (incorrect_trace * cross_coeff[1, :, cur_skill].T).sum(-1)
                ) / support
            if practice_coeff is not None:
                current_history = torch.stack((successes.gather(1, cur_skill[:, None]).squeeze(1),
                                               failures.gather(1, cur_skill[:, None]).squeeze(1)), dim=-1)
                current_cog_logit = current_cog_logit + (torch.log1p(current_history) * practice_coeff[cur_skill]).sum(-1)
            if ability_scale is not None:
                current_cog_logit = current_cog_logit + ability_scale * initial_ability
            current_cog_prob = torch.sigmoid(current_cog_logit)

            # The observed proxy is available only after the response.  Its
            # mismatch with the previous state supplies the affect innovation.
            affect_innovation = (behavior_affect[:, t] - previous_affect).abs().mean(dim=-1)
            innovation = 0.5 * ((correct[:, t] - current_cog_prob).abs() + affect_innovation)

            shifted = torch.cat((innovation_history[:, 1:], innovation[:, None]), dim=1)
            innovation_history = torch.where(active[:, None], shifted, innovation_history)
            count = min(self.history, t + 1)
            recent = innovation_history[:, -count:]
            if self.weight_mode == "linear":
                weights = torch.arange(self.history - count + 1, self.history + 1, device=skill.device, dtype=innovation_history.dtype)[None, :].expand(batch_size, -1)
            else:
                age = torch.arange(count - 1, -1, -1, device=skill.device, dtype=innovation_history.dtype)[None, :]
                if self.weight_mode == "exponential":
                    weights = torch.exp(-age / self.weight_tau)
                elif self.weight_mode == "gaussian":
                    weights = torch.exp(-0.5 * (age / self.weight_tau) ** 2)
                else:
                    recency = -age / self.weight_tau
                    weights = torch.softmax(recent / max(self.event_temperature, 1e-3) + recency, dim=-1)
            score = (recent * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
            if self.mode == "affect_event":
                if self.training:
                    gate = torch.sigmoid((score - self.event_threshold) / self.event_temperature)
                else:
                    gate = (score > self.event_threshold).float()
            elif self.mode == "affect_periodic":
                gate = torch.full_like(score, float((t + 1) % self.periodic_period == 0))
            elif self.mode == "affect_random":
                window_index = batch.get("window_index", torch.arange(batch_size, device=skill.device)).long()
                key = (window_index * 1103515245 + (t + 1) * 12345 + self.random_seed) % 10000
                gate = (key.float() < self.random_rate * 10000.0).to(score.dtype)
            else:
                gate = torch.ones_like(score)

            if "gate_override" in batch:
                if self.training:
                    raise ValueError("gate_override is reserved for frozen evaluation")
                gate = batch["gate_override"][:, t].to(score.dtype)

            # Cognitive mastery always incorporates the observed response.  An
            # optional bounded affect gain is evaluated from the pre-feedback
            # affect state, so it cannot use the current label as a shortcut.
            if self.use_affect_gain and self.mode != "cognitive":
                affect_gain_part = torch.tanh(self.gain_affect_raw)[None, :] * (previous_affect - 0.5)
                affect_gain_raw = 1.0 + affect_gain_part.sum(dim=-1)
                gain_coeff = affect_gain_raw
                gain_coeff = gain_coeff.clamp(0.5, 1.5)
            else:
                gain_coeff = torch.ones(batch_size, device=skill.device)
                affect_gain_part = torch.zeros(batch_size, 4, device=skill.device)
                affect_gain_raw = torch.ones(batch_size, device=skill.device)
            affect_gain_base = gain_coeff.clone()
            if self.behavior_gain:
                behavior_signal = self._bounded_behavior(behavior[:, t], correct[:, t])
                if self.behavior_gain_hinge:
                    knots_b = torch.tensor((-0.3, 0.0, 0.3), device=behavior_signal.device, dtype=behavior_signal.dtype)
                    hinges_b = F.relu(behavior_signal[..., None] - knots_b)
                    if self.smooth_behavior_gain:
                        # A neutral standardized feature has zero contribution.
                        hinges_b = hinges_b - F.relu(-knots_b)
                    behavior_coeff = 0.5 * torch.tanh(self.behavior_gain_hinge_raw)
                    behavior_effect = (behavior_signal * torch.tanh(self.behavior_gain_raw)[None, :] + (hinges_b * behavior_coeff[None, :, :]).sum(dim=-1)).sum(dim=-1)
                else:
                    behavior_coeff = torch.tanh(self.behavior_gain_raw)
                    behavior_effect = (behavior_signal * behavior_coeff[None, :]).sum(dim=-1)
                if self.smooth_behavior_gain:
                    behavior_multiplier = 1.0 + 0.5 * torch.tanh(
                        behavior_effect / math.sqrt(self.behavior_dim + 1)
                    )
                else:
                    behavior_multiplier = (1.0 + 0.5 * behavior_effect).clamp(0.5, 1.5)
                gain_coeff = gain_coeff * behavior_multiplier
                gain_coeff = gain_coeff.clamp(0.5, 1.5)
            gain_coeff = (gain_coeff * self.fixed_learning_gain).clamp(0.5, 1.5)
            if learning_pos is not None and learning_neg is not None:
                # PFA-style signed evidence: correct and incorrect responses
                # have separately learned, positive rates while retaining the
                # prediction-error factor.  This keeps the update auditable.
                signed_evidence = torch.where(
                    correct[:, t] > 0.5,
                    learning_pos[cur_skill] * (1.0 - current_cog_prob),
                    -learning_neg[cur_skill] * current_cog_prob,
                )
            else:
                signed_evidence = learning_rate[cur_skill] * (correct[:, t] - current_cog_prob)
            evidence = gain_coeff * signed_evidence
            updated_u = current_u + evidence
            mastery = mastery.scatter(
                1, cur_skill[:, None], torch.where(active, updated_u, mastery.gather(1, cur_skill[:, None]).squeeze(1))[:, None]
            )
            if practice_coeff is not None:
                successes = successes.scatter_add(1, cur_skill[:, None], (correct[:, t] * active)[:, None])
                failures = failures.scatter_add(1, cur_skill[:, None], ((1.0 - correct[:, t]) * active)[:, None])
            if self.hawkes_readout:
                correct_trace = correct_trace.scatter_add(1, cur_skill[:, None], (correct[:, t] * active)[:, None])
                incorrect_trace = incorrect_trace.scatter_add(1, cur_skill[:, None], ((1.0 - correct[:, t]) * active)[:, None])

            affect_input = self._bounded_behavior(behavior[:, t], correct[:, t])
            if short_affect_decay is not None:
                short_affect_ema = torch.where(
                    active[:, None],
                    short_affect_decay[None, :] * short_affect_ema
                    + (1.0 - short_affect_decay[None, :]) * behavior_affect[:, t],
                    short_affect_ema,
                )
                affect_input = torch.cat((affect_input, short_affect_ema), dim=-1)
            candidate_affect = (
                affect_logit * persistence[None, :]
                + F.linear(affect_input, self.affect_behavior)
                + self.affect_bias[None, :]
            )
            affect_logit = torch.where(
                active[:, None],
                affect_logit + gate[:, None] * (candidate_affect - affect_logit),
                affect_logit,
            )
            affect_state = torch.sigmoid(affect_logit)

            target_u = mastery.gather(1, tgt_skill[:, None]).squeeze(1)
            if self.revisit_forgetting:
                # Project to the current observed time, not the future target
                # time. The state itself is decayed when this skill is observed.
                target_u = target_u * torch.exp(-forgetting_rate[tgt_skill] * batch["target_revisit_delta"][:, t].clamp(0, 20) / 10.0)
            target_mastery = torch.sigmoid(target_u)
            mastery_part = mastery_scale * (target_mastery - 0.5)
            skill_part = skill_bias[tgt_skill]
            difficulty_part = -difficulty_scale * item_difficulty[tgt_problem]
            if item_discrimination is not None:
                mastery_part = item_discrimination[tgt_problem] * mastery_part
                difficulty_part = item_discrimination[tgt_problem] * difficulty_part
            base_logit = self.intercept + mastery_part + skill_part + difficulty_part
            if self.hawkes_readout:
                support = 1 + correct_trace.sum(-1) + incorrect_trace.sum(-1)
                correct_cross = (correct_trace * cross_coeff[0, :, tgt_skill].T).sum(-1) / support
                incorrect_cross = (incorrect_trace * cross_coeff[1, :, tgt_skill].T).sum(-1) / support
                base_logit = base_logit + correct_cross + incorrect_cross
            if practice_coeff is not None:
                target_history = torch.stack((successes.gather(1, tgt_skill[:, None]).squeeze(1),
                                              failures.gather(1, tgt_skill[:, None]).squeeze(1)), dim=-1)
                practice_parts = torch.log1p(target_history) * practice_coeff[tgt_skill]
                base_logit = base_logit + practice_parts.sum(-1)
            if ability_scale is not None:
                base_logit = base_logit + ability_scale * initial_ability
            if self.mode == "cognitive":
                contributions = torch.zeros_like(affect_state)
            else:
                centered = affect_state - 0.5
                if self.response_mode == "hinge":
                    knots = torch.tensor((-0.2, 0.0, 0.2), device=affect_state.device, dtype=affect_state.dtype)
                    hinges = F.relu(centered[..., None] - knots)
                    contributions = centered * affect_coeff[None, :] + (hinges * (self.affect_max * torch.tanh(self.affect_hinge_raw))[None, :, :]).sum(dim=-1)
                else:
                    contributions = centered * affect_coeff[None, :]
                if self.affect_difficulty_interaction:
                    difficulty_context = torch.tanh(item_difficulty[tgt_problem])
                    contributions = contributions + (
                        0.25 * centered * difficulty_context[:, None]
                        * torch.tanh(self.affect_difficulty_raw)[None, :]
                    )
            full_logit = base_logit + contributions.sum(dim=-1)

            cognitive_logits.append(base_logit)
            full_logits.append(full_logit)
            affect_states.append(affect_state)
            affect_contribs.append(contributions)
            gates.append(gate)
            innovations.append(innovation)
            scores.append(score)
            parts = [self.intercept.expand_as(base_logit), mastery_part, skill_part, difficulty_part]
            if ability_scale is not None:
                parts.append(ability_scale * initial_ability)
            if practice_coeff is not None:
                parts += [practice_parts[:, 0], practice_parts[:, 1]]
            if self.hawkes_readout:
                parts += [correct_cross, incorrect_cross]
            cognitive_parts.append(torch.stack(parts, dim=-1))
            gain_coeffs.append(gain_coeff)
            affect_gain_parts.append(affect_gain_part)
            affect_gain_raws.append(affect_gain_raw)
            affect_gain_bases.append(affect_gain_base)
            short_affect_emas.append(short_affect_ema)

        return {
            "correct_logits": torch.stack(full_logits, dim=1),
            "cognitive_logits": torch.stack(cognitive_logits, dim=1),
            "affect_state": torch.stack(affect_states, dim=1),
            "affect_contributions": torch.stack(affect_contribs, dim=1),
            "gate": torch.stack(gates, dim=1),
            "innovation": torch.stack(innovations, dim=1),
            "event_score": torch.stack(scores, dim=1),
            "cognitive_parts": torch.stack(cognitive_parts, dim=1),
            "gain_coeff": torch.stack(gain_coeffs, dim=1),
            "affect_gain_parts": torch.stack(affect_gain_parts, dim=1),
            "affect_gain_raw": torch.stack(affect_gain_raws, dim=1),
            "affect_gain_base": torch.stack(affect_gain_bases, dim=1),
            "short_affect_ema": torch.stack(short_affect_emas, dim=1),
            "affect_logvar": self.affect_logvar.detach().expand(batch_size, steps, -1) if self.affect_logvar is not None else torch.zeros(batch_size, steps, 4, device=skill.device),
        }


def _metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return {
        "count": int(y.size),
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(((p >= 0.5) == y).mean()),
    }


def collect(model: ECKT, loader: DataLoader, device: torch.device, collect_arrays: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    values: dict[str, list[np.ndarray]] = {k: [] for k in ("target", "full", "cognitive", "full_logit", "cognitive_logit", "affect_state", "affect_target", "contrib", "gate", "innovation", "event_score", "cognitive_parts", "gain_coeff", "affect_gain_parts", "affect_gain_raw", "affect_gain_base", "short_affect_ema", "affect_logvar")}
    with torch.no_grad():
        for batch in loader:
            device_batch = {k: v.to(device) for k, v in batch.items()}
            out = model(device_batch)
            active = device_batch["mask"].bool()
            values["target"].append(device_batch["target_correct"][active].cpu().numpy())
            values["full"].append(torch.sigmoid(out["correct_logits"])[active].cpu().numpy())
            values["cognitive"].append(torch.sigmoid(out["cognitive_logits"])[active].cpu().numpy())
            values["full_logit"].append(out["correct_logits"][active].cpu().numpy())
            values["cognitive_logit"].append(out["cognitive_logits"][active].cpu().numpy())
            values["affect_state"].append(out["affect_state"][active].cpu().numpy())
            affect_target = device_batch["target_behavior_affect"]
            if model.affect_target == "detector":
                affect_target = device_batch["target_affect"][..., list(DETECTOR_TO_STATE)]
            values["affect_target"].append(affect_target[active].cpu().numpy())
            values["contrib"].append(out["affect_contributions"][active].cpu().numpy())
            values["gate"].append(out["gate"][active].cpu().numpy())
            values["innovation"].append(out["innovation"][active].cpu().numpy())
            values["event_score"].append(out["event_score"][active].cpu().numpy())
            values["cognitive_parts"].append(out["cognitive_parts"][active].cpu().numpy())
            values["gain_coeff"].append(out["gain_coeff"][active].cpu().numpy())
            values["affect_gain_parts"].append(out["affect_gain_parts"][active].cpu().numpy())
            values["affect_gain_raw"].append(out["affect_gain_raw"][active].cpu().numpy())
            values["affect_gain_base"].append(out["affect_gain_base"][active].cpu().numpy())
            values["short_affect_ema"].append(out["short_affect_ema"][active].cpu().numpy())
            values["affect_logvar"].append(out["affect_logvar"][active].cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in values.items()}
    metrics = _metric_row(arrays["target"], arrays["full"])
    metrics.update({
        "cognitive_auc": float(roc_auc_score(arrays["target"], arrays["cognitive"])),
        "cognitive_brier": float(brier_score_loss(arrays["target"], arrays["cognitive"])),
        "affect_rmse": float(np.sqrt(np.mean((arrays["affect_state"] - arrays["affect_target"]) ** 2))),
        "event_rate": float((arrays["gate"] >= 0.5).mean()),
        "mean_abs_affect_logit": float(np.abs(arrays["contrib"].sum(axis=1)).mean()),
        "max_logit_reconstruction_error": float(np.max(np.abs(arrays["full_logit"] - (arrays["cognitive_parts"].sum(axis=1) + arrays["contrib"].sum(axis=1))))),
    })
    return metrics, (arrays if collect_arrays else None)


def write_explanations(path: Path, arrays: dict[str, np.ndarray], rows: int = 200, cognitive_names: list[str] | None = None) -> None:
    order = np.argsort(-np.abs(arrays["contrib"].sum(axis=1)))[:rows]
    columns = ["target", "full_probability", "cognitive_probability", "cognitive_logit", "full_logit", "affect_residual_logit", "gain_coefficient", "gate", "innovation", "event_score"]
    if cognitive_names is None:
        cognitive_names = ["intercept", "mastery", "skill_bias", "difficulty"]
        if arrays["cognitive_parts"].shape[1] in (5, 7):
            cognitive_names.append("student_ability")
        if arrays["cognitive_parts"].shape[1] in (6, 7):
            cognitive_names += ["success_history", "failure_history"]
    if len(cognitive_names) != arrays["cognitive_parts"].shape[1]:
        raise ValueError("cognitive names do not cover all exported contributions")
    columns += [f"cognitive_{name}" for name in cognitive_names]
    columns += [f"state_{name}" for name in STATE_NAMES]
    columns += [f"affect_contribution_{name}" for name in STATE_NAMES]
    columns += [f"affect_gain_contribution_{name}" for name in STATE_NAMES]
    columns += ["affect_gain_raw", "affect_gain_base"]
    columns += [f"short_affect_ema_{name}" for name in STATE_NAMES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in order:
            row: dict[str, object] = {
                "target": int(arrays["target"][index]),
                "full_probability": float(arrays["full"][index]),
                "cognitive_probability": float(arrays["cognitive"][index]),
                "cognitive_logit": float(arrays["cognitive_logit"][index]),
                "full_logit": float(arrays["full_logit"][index]),
                "affect_residual_logit": float(arrays["contrib"][index].sum()),
                "gain_coefficient": float(arrays["gain_coeff"][index]),
                "gate": float(arrays["gate"][index]),
                "innovation": float(arrays["innovation"][index]),
                "event_score": float(arrays["event_score"][index]),
            }
            row.update({f"cognitive_{name}": float(value) for name, value in zip(cognitive_names, arrays["cognitive_parts"][index])})
            row.update({f"state_{name}": float(value) for name, value in zip(STATE_NAMES, arrays["affect_state"][index])})
            row.update({f"affect_contribution_{name}": float(value) for name, value in zip(STATE_NAMES, arrays["contrib"][index])})
            row.update({f"affect_gain_contribution_{name}": float(value) for name, value in zip(STATE_NAMES, arrays["affect_gain_parts"][index])})
            row["affect_gain_raw"] = float(arrays["affect_gain_raw"][index])
            row["affect_gain_base"] = float(arrays["affect_gain_base"][index])
            row.update({f"short_affect_ema_{name}": float(value) for name, value in zip(STATE_NAMES, arrays["short_affect_ema"][index])})
            writer.writerow(row)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_skills = int(metadata["num_skills_with_padding"])
    train_ds = ECKTWindowDataset(args.data_dir, "train", int(metadata["max_seq_len"]), num_skills, args.revisit_forgetting)
    val_ds = ECKTWindowDataset(args.data_dir, "val", int(metadata["max_seq_len"]), num_skills, args.revisit_forgetting)
    test_ds = None if args.skip_test else ECKTWindowDataset(args.data_dir, "test", int(metadata["max_seq_len"]), num_skills, args.revisit_forgetting)
    loaders = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = ECKT(
        num_skills=num_skills,
        num_problems=int(metadata.get("num_problems_with_padding", 1)),
        behavior_dim=int(len(metadata.get("behavior_feature_names", [])) or 8),
        mode=args.mode, history=args.history, event_threshold=args.event_threshold,
        event_temperature=args.event_temperature, affect_max=args.affect_max,
        periodic_period=args.periodic_period,
        random_rate=args.random_rate,
        use_affect_gain=args.affect_gain,
        random_seed=args.seed,
        affect_target=args.affect_target,
        use_discrimination=args.use_discrimination,
        response_mode=args.response_mode,
        weight_mode=args.weight_mode,
        weight_tau=args.weight_tau,
        gaussian_uncertainty=args.gaussian_uncertainty,
        behavior_gain=args.behavior_gain,
        behavior_gain_hinge=args.behavior_gain_hinge,
        student_ability=args.student_ability,
        affect_forgetting=args.affect_forgetting,
        pfa_signed_learning=args.pfa_signed_learning,
        affect_difficulty_interaction=args.affect_difficulty_interaction,
        multiscale_affect=args.multiscale_affect,
        normalize_behavior=args.normalize_behavior,
        smooth_behavior_gain=args.smooth_behavior_gain,
        fixed_learning_gain=args.fixed_learning_gain,
        practice_readout=args.practice_readout,
        revisit_forgetting=args.revisit_forgetting,
        hawkes_readout=args.hawkes_readout,
    ).to(device)
    initialization: dict[str, Any] = {"scope": "training students only"}
    if args.normalize_behavior:
        feature_path = args.data_dir / "train_behavior_features.npy"
        features = np.load(feature_path, mmap_mode="r")
        total = np.zeros(model.behavior_dim, dtype=np.float64)
        squares = np.zeros_like(total)
        count = np.zeros_like(total)
        for start in range(0, len(features), 200_000):
            chunk = np.asarray(features[start:start + 200_000], dtype=np.float64)
            finite = np.isfinite(chunk)
            chunk = np.where(finite, chunk, 0.0)
            total += chunk.sum(axis=0)
            squares += np.square(chunk).sum(axis=0)
            count += finite.sum(axis=0)
        center = total / np.maximum(count, 1)
        scale = np.sqrt(np.maximum(squares / np.maximum(count, 1) - center ** 2, 0))
        scale = np.maximum(scale, 0.05)
        with torch.no_grad():
            model.behavior_center.copy_(torch.as_tensor(center, device=device))
            model.behavior_scale.copy_(torch.as_tensor(scale, device=device))
        initialization["behavior_normalization"] = {
            "source": str(feature_path), "center": center.tolist(),
            "scale": scale.tolist(), "finite_counts": count.astype(int).tolist(),
        }
    if args.init_item_prior:
        # Empirical-Bayes initialization is fitted to training responses only;
        # it introduces no held-out labels and no pretrained KT parameters.
        problem = np.asarray(train_ds.problems, dtype=np.int64)
        response = np.asarray(train_ds.correct, dtype=np.float64)
        n = model.num_problems
        counts = np.bincount(problem, minlength=n)
        successes = np.bincount(problem, weights=response, minlength=n)
        global_p = float(np.clip(response.mean(), 1e-4, 1 - 1e-4))
        prior_p = (successes + 20.0 * global_p) / (counts + 20.0)
        prior_p = np.clip(prior_p, 1e-4, 1 - 1e-4)
        global_logit = math.log(global_p / (1.0 - global_p))
        prior_logit = np.log(prior_p / (1.0 - prior_p))
        with torch.no_grad():
            difficulty_scale = float(F.softplus(model.difficulty_scale_raw) + 1e-4)
            discrimination = 0.5 + math.log(2.0) if model.use_discrimination else 1.0
            difficulty = (global_logit - prior_logit) / (difficulty_scale * discrimination)
            difficulty[0] = 0.0
            model.intercept.fill_(global_logit)
            model.item_difficulty.copy_(torch.as_tensor(difficulty, device=device))
        initialization["item_prior"] = {
            "prior_strength": 20.0, "training_rows": int(len(response)),
            "known_training_items": int(np.count_nonzero(counts[1:])),
            "global_correct": global_p, "unknown_item_difficulty": 0.0,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders(train_ds, True):
            device_batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(device_batch)
            mask = device_batch["mask"]
            den = mask.sum().clamp_min(1.0)
            correct_loss = (F.binary_cross_entropy_with_logits(out["correct_logits"], device_batch["target_correct"], reduction="none") * mask).sum() / den
            cognitive_loss = (F.binary_cross_entropy_with_logits(out["cognitive_logits"], device_batch["target_correct"], reduction="none") * mask).sum() / den
            affect_target = device_batch["target_behavior_affect"]
            if model.affect_target == "detector":
                affect_target = device_batch["target_affect"][..., list(DETECTOR_TO_STATE)]
            affect_residual = out["affect_state"] - affect_target
            if model.gaussian_uncertainty:
                logvar = model.affect_logvar.clamp(-5.0, 3.0)[None, None, :]
                affect_loss = (0.5 * (torch.exp(-logvar) * affect_residual.square() + logvar).mean(dim=-1) * mask).sum() / den
            else:
                affect_loss = (affect_residual.square().mean(dim=-1) * mask).sum() / den
            smooth_mask = mask[:, 1:] * mask[:, :-1]
            smooth_loss = (((out["affect_state"][:, 1:] - out["affect_state"][:, :-1]) ** 2).mean(dim=-1) * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
            residual_penalty = (out["affect_contributions"].abs().sum(dim=-1) * mask).sum() / den
            loss = correct_loss + args.cognitive_anchor_weight * cognitive_loss
            if args.mode != "cognitive":
                loss = loss + args.affect_loss_weight * affect_loss + args.smoothness_weight * smooth_loss + args.residual_penalty_weight * residual_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _ = collect(model, loaders(val_ds, False), device, collect_arrays=False)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["auc"] > best_auc + args.min_delta:
            best_auc = float(val_metrics["auc"])
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        elif args.patience > 0 and epoch - best_epoch >= args.patience:
            break
    if not (args.output_dir / "best.pt").exists():
        raise RuntimeError("best.pt was not written")
    model.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=device), strict=True)
    event_selection: dict[str, Any] | None = None
    if args.mode == "affect_event":
        # Select the trigger boundary on validation only.  The model weights
        # stay fixed while the replay is repeated with each candidate.
        model.mode = "affect_always"
        always_metrics, _ = collect(model, loaders(val_ds, False), device, collect_arrays=False)
        always_val_auc = float(always_metrics["auc"])
        model.mode = "affect_event"
        candidates = np.linspace(0.05, 0.45, 9)
        rows: list[dict[str, float]] = []
        for threshold in candidates:
            model.event_threshold = float(threshold)
            candidate_metrics, _ = collect(model, loaders(val_ds, False), device, collect_arrays=False)
            rows.append({"threshold": float(threshold), "val_auc": float(candidate_metrics["auc"]), "event_rate": float(candidate_metrics["event_rate"])})
        eligible = [row for row in rows if row["val_auc"] >= always_val_auc - args.max_event_auc_loss]
        selected = min(eligible, key=lambda row: (row["event_rate"], -row["val_auc"])) if eligible else max(rows, key=lambda row: (row["val_auc"], -row["event_rate"]))
        model.event_threshold = float(selected["threshold"])
        event_selection = {"selection_split": "validation", "always_val_auc": always_val_auc, "max_event_auc_loss": args.max_event_auc_loss, "selected": selected, "candidates": rows}
    val_metrics, val_arrays = collect(model, loaders(val_ds, False), device, collect_arrays=True)
    test_metrics, test_arrays = (None, None) if test_ds is None else collect(model, loaders(test_ds, False), device, collect_arrays=True)
    if val_arrays is not None:
        np.savez_compressed(args.output_dir / "val_events.npz", **val_arrays)
        write_explanations(args.output_dir / "val_explanations.csv", val_arrays, cognitive_names=model.cognitive_part_names)
    if test_arrays is not None:
        np.savez_compressed(args.output_dir / "test_events.npz", **test_arrays)
        write_explanations(args.output_dir / "test_explanations.csv", test_arrays, cognitive_names=model.cognitive_part_names)
    report = {
        "model": "ECKT-from-scratch",
        "mode": args.mode,
        "affect_target": args.affect_target,
        "behavior_gain": args.behavior_gain,
        "state_names": list(STATE_NAMES),
        "cognitive_component_names": model.cognitive_part_names,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "initialization": initialization,
        "event_selection": event_selection,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "train_windows": len(train_ds), "val_windows": len(val_ds),
        "test_windows": None if test_ds is None else len(test_ds),
        "elapsed_seconds": time.time() - started,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "label_boundary": "Affect states are behavior-derived proxy states, not human emotion gold labels.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["cognitive", "affect_always", "affect_periodic", "affect_random", "affect_event"], default="affect_always")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--affect-loss-weight", type=float, default=0.5)
    parser.add_argument("--cognitive-anchor-weight", type=float, default=0.25)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--residual-penalty-weight", type=float, default=0.001)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--event-threshold", type=float, default=0.15)
    parser.add_argument("--event-temperature", type=float, default=0.08)
    parser.add_argument("--max-event-auc-loss", type=float, default=0.0005)
    parser.add_argument("--affect-target", choices=["behavior", "detector"], default="behavior")
    parser.add_argument("--use-discrimination", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--response-mode", choices=["linear", "hinge"], default="linear")
    parser.add_argument("--weight-mode", choices=["linear", "exponential", "gaussian", "dynamic"], default="linear")
    parser.add_argument("--weight-tau", type=float, default=2.0)
    parser.add_argument("--gaussian-uncertainty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--behavior-gain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--behavior-gain-hinge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--affect-forgetting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pfa-signed-learning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--affect-difficulty-interaction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiscale-affect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize-behavior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-item-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smooth-behavior-gain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed-learning-gain", type=float, default=1.0)
    parser.add_argument("--practice-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--revisit-forgetting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hawkes-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--student-ability", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--periodic-period", type=int, default=10)
    parser.add_argument("--random-rate", type=float, default=0.10)
    parser.add_argument("--affect-gain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--affect-max", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
