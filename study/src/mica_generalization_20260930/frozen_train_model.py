#!/usr/bin/env python3
"""Train leakage-aware cognitive/affective sequence models on prepared arrays."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    r2_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    """Memory-mapped student windows; each target is the following interaction."""

    def __init__(self, data_dir: Path, split: str, max_seq_len: int) -> None:
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_steps = max_seq_len - 1
        self.skills = np.load(data_dir / f"{split}_skills.npy", mmap_mode="r")
        self.correct = np.load(data_dir / f"{split}_correct.npy", mmap_mode="r")
        self.response = np.load(data_dir / f"{split}_response.npy", mmap_mode="r")
        self.affects = np.load(data_dir / f"{split}_affects.npy", mmap_mode="r")
        behavior_affect_path = data_dir / f"{split}_behavior_affects.npy"
        self.behavior_affects = (
            np.load(behavior_affect_path, mmap_mode="r")
            if behavior_affect_path.exists() else None
        )
        behavior_features_path = data_dir / f"{split}_behavior_features.npy"
        self.behavior_features = (
            np.load(behavior_features_path, mmap_mode="r")
            if behavior_features_path.exists() else None
        )
        self.delta = np.load(data_dir / f"{split}_delta.npy", mmap_mode="r")
        problem_path = data_dir / f"{split}_problem_index.npy"
        self.problems = np.load(problem_path, mmap_mode="r") if problem_path.exists() else None
        problem_difficulty_path = data_dir / f"{split}_problem_difficulty.npy"
        self.problem_difficulty = (
            np.load(problem_difficulty_path, mmap_mode="r")
            if problem_difficulty_path.exists() else None
        )
        self.offsets = np.load(data_dir / f"{split}_offsets.npy", mmap_mode="r")
        self.windows = np.load(data_dir / f"{split}_windows.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sequence_index, start, window_length = (int(x) for x in self.windows[index])
        base = int(self.offsets[sequence_index]) + start
        length = min(window_length, self.max_seq_len)
        steps = length - 1

        skill = np.zeros(self.max_steps, dtype=np.int64)
        correct = np.zeros(self.max_steps, dtype=np.int64)
        response = np.zeros(self.max_steps, dtype=np.float32)
        delta = np.zeros(self.max_steps, dtype=np.float32)
        affect = np.zeros((self.max_steps, 4), dtype=np.float32)
        behavior_affect = np.zeros((self.max_steps, 4), dtype=np.float32)
        behavior_feature_dim = (
            int(self.behavior_features.shape[1]) if self.behavior_features is not None else 0
        )
        behavior_features = np.zeros(
            (self.max_steps, behavior_feature_dim), dtype=np.float32
        )
        problem = np.zeros(self.max_steps, dtype=np.int64)
        problem_difficulty = np.zeros(self.max_steps, dtype=np.float32)
        target_skill = np.zeros(self.max_steps, dtype=np.int64)
        target_problem = np.zeros(self.max_steps, dtype=np.int64)
        target_problem_difficulty = np.zeros(self.max_steps, dtype=np.float32)
        target_correct = np.zeros(self.max_steps, dtype=np.float32)
        target_affect = np.zeros((self.max_steps, 4), dtype=np.float32)
        target_behavior_affect = np.zeros((self.max_steps, 4), dtype=np.float32)
        mask = np.zeros(self.max_steps, dtype=np.float32)

        if steps > 0:
            current = slice(base, base + steps)
            following = slice(base + 1, base + steps + 1)
            skill[:steps] = self.skills[current]
            correct[:steps] = self.correct[current]
            response[:steps] = self.response[current]
            delta[:steps] = self.delta[current]
            affect[:steps] = self.affects[current]
            if self.behavior_affects is not None:
                behavior_affect[:steps] = self.behavior_affects[current]
            if self.behavior_features is not None:
                behavior_features[:steps] = self.behavior_features[current]
            if self.problems is not None:
                problem[:steps] = self.problems[current]
            target_skill[:steps] = self.skills[following]
            if self.problems is not None:
                target_problem[:steps] = self.problems[following]
            if self.problem_difficulty is not None:
                problem_difficulty[:steps] = self.problem_difficulty[current]
                target_problem_difficulty[:steps] = self.problem_difficulty[following]
            target_correct[:steps] = self.correct[following]
            target_affect[:steps] = self.affects[following]
            if self.behavior_affects is not None:
                target_behavior_affect[:steps] = self.behavior_affects[following]
            mask[:steps] = 1.0

        return {
            "skill": torch.from_numpy(skill),
            "correct": torch.from_numpy(correct),
            "response": torch.from_numpy(response),
            "delta": torch.from_numpy(delta),
            "affect": torch.from_numpy(affect),
            "behavior_affect": torch.from_numpy(behavior_affect),
            "behavior_features": torch.from_numpy(behavior_features),
            "problem": torch.from_numpy(problem),
            "problem_difficulty": torch.from_numpy(problem_difficulty),
            "target_skill": torch.from_numpy(target_skill),
            "target_problem": torch.from_numpy(target_problem),
            "target_problem_difficulty": torch.from_numpy(target_problem_difficulty),
            "target_correct": torch.from_numpy(target_correct),
            "target_affect": torch.from_numpy(target_affect),
            "target_behavior_affect": torch.from_numpy(target_behavior_affect),
            "mask": torch.from_numpy(mask),
        }


class CognitiveAffectiveModel(nn.Module):
    """Two coupled latent states with an optional innovation-triggered update gate."""

    def __init__(self, num_skills: int, mode: str, skill_dim: int = 32,
                 hidden_cognitive: int = 64, hidden_affect: int = 32,
                 num_problems: int = 1, use_problem: bool = False,
                 use_difficulty: bool = False,
                 affect_source: str = "detector",
                 explicit_state: bool = False) -> None:
        super().__init__()
        if mode not in {"cognitive", "joint", "event"}:
            raise ValueError(f"unknown mode {mode}")
        if affect_source not in {"detector", "behavior", "both"}:
            raise ValueError(f"unknown affect source {affect_source}")
        self.mode = mode
        self.use_affect = mode != "cognitive"
        self.use_problem = bool(use_problem and num_problems > 1)
        self.use_difficulty = bool(use_difficulty)
        self.affect_source = affect_source
        self.explicit_state = bool(explicit_state)
        self.skill_embedding = nn.Embedding(num_skills, skill_dim, padding_idx=0)
        if self.use_problem:
            self.problem_embedding = nn.Embedding(num_problems, skill_dim, padding_idx=0)
        self.correct_embedding = nn.Embedding(2, 4)

        problem_dim = skill_dim if self.use_problem else 0
        difficulty_dim = 1 if self.use_difficulty else 0
        cognitive_input = skill_dim + problem_dim + difficulty_dim + 4 + 2
        if self.use_affect:
            cognitive_input += hidden_affect
        self.cognitive_cell = nn.GRUCell(cognitive_input, hidden_cognitive)

        if self.use_affect:
            observed_affect_dim = 8 if affect_source == "both" else 4
            affect_input = observed_affect_dim + 4 + 2 + hidden_cognitive
            self.affect_cell = nn.GRUCell(affect_input, hidden_affect)
            joint_dim = hidden_cognitive + hidden_affect
            self.affect_head = nn.Sequential(
                nn.Linear(joint_dim, hidden_affect), nn.ReLU(), nn.Linear(hidden_affect, 4)
            )
            self.correct_head = nn.Sequential(
                nn.Linear(joint_dim + skill_dim + problem_dim + difficulty_dim, hidden_cognitive),
                nn.ReLU(), nn.Linear(hidden_cognitive, 1),
            )
            if mode == "event":
                self.gate_net = nn.Sequential(
                    nn.Linear(joint_dim + 2, hidden_affect), nn.Tanh(), nn.Linear(hidden_affect, 1)
                )
            self.behavior_affect_head = nn.Sequential(
                nn.Linear(joint_dim, hidden_affect), nn.Tanh(), nn.Linear(hidden_affect, 4)
            )
            if self.explicit_state:
                self.affect_response_coeff = nn.Parameter(torch.zeros(4))
        else:
            self.correct_head = nn.Sequential(
                nn.Linear(hidden_cognitive + skill_dim + problem_dim + difficulty_dim, hidden_cognitive),
                nn.ReLU(), nn.Linear(hidden_cognitive, 1),
            )
        if self.explicit_state:
            self.mastery_head = nn.Linear(hidden_cognitive, 1)
            self.mastery_scale = nn.Parameter(torch.tensor(1.0))
        self.hidden_cognitive = hidden_cognitive
        self.hidden_affect = hidden_affect

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        skill = batch["skill"]
        correct = batch["correct"]
        response = batch["response"]
        delta = batch["delta"]
        affect = batch["affect"]
        behavior_affect = batch["behavior_affect"]
        target_skill = batch["target_skill"]
        problem = batch["problem"]
        target_problem = batch["target_problem"]
        problem_difficulty = batch["problem_difficulty"]
        target_problem_difficulty = batch["target_problem_difficulty"]
        mask = batch["mask"]
        batch_size, steps = skill.shape
        device = skill.device
        h_c = torch.zeros(batch_size, self.hidden_cognitive, device=device)
        h_a = torch.zeros(batch_size, self.hidden_affect, device=device) if self.use_affect else None
        correct_outputs: list[Tensor] = []
        affect_outputs: list[Tensor] = []
        behavior_affect_outputs: list[Tensor] = []
        mastery_outputs: list[Tensor] = []
        gate_outputs: list[Tensor] = []
        innovation_outputs: list[Tensor] = []

        for index in range(steps):
            active = mask[:, index].bool()
            skill_embedding = self.skill_embedding(skill[:, index])
            problem_embedding = (
                self.problem_embedding(problem[:, index]) if self.use_problem else None
            )
            difficulty_value = (
                problem_difficulty[:, index].unsqueeze(-1) if self.use_difficulty else None
            )
            correct_embedding = self.correct_embedding(correct[:, index].clamp(0, 1))
            numeric = torch.stack((response[:, index], delta[:, index]), dim=-1)

            if self.use_affect:
                assert h_a is not None
                joint_before = torch.cat((h_c, h_a), dim=-1)
                current_affect = torch.sigmoid(self.affect_head(joint_before))
                current_behavior_affect = torch.sigmoid(
                    self.behavior_affect_head(joint_before)
                )
                current_base_logit = self.correct_head(
                    torch.cat(
                        tuple(x for x in (joint_before, skill_embedding, problem_embedding,
                                          difficulty_value)
                              if x is not None), dim=-1
                    )
                ).squeeze(-1)
                if self.affect_source == "detector":
                    current_observed_affect = affect[:, index]
                    current_predicted_affect = current_affect
                elif self.affect_source == "behavior":
                    current_observed_affect = behavior_affect[:, index]
                    current_predicted_affect = current_behavior_affect
                else:
                    current_observed_affect = torch.cat(
                        (affect[:, index], behavior_affect[:, index]), dim=-1
                    )
                    current_predicted_affect = torch.cat(
                        (current_affect, current_behavior_affect), dim=-1
                    )
                current_logit = current_base_logit
                if self.explicit_state:
                    current_logit = current_logit + self.mastery_scale * self.mastery_head(
                        h_c
                    ).squeeze(-1)
                    current_logit = current_logit + (
                        (current_affect - 0.5) * self.affect_response_coeff
                    ).sum(dim=-1)
                current_correct = torch.sigmoid(current_logit)
                innovation = (
                    (correct[:, index].float() - current_correct).abs()
                    + (current_observed_affect - current_predicted_affect).abs().mean(dim=-1)
                ) * 0.5
                if self.mode == "event":
                    gate_input = torch.cat(
                        (joint_before, innovation.unsqueeze(-1), delta[:, index].unsqueeze(-1)), dim=-1
                    )
                    gate = torch.sigmoid(self.gate_net(gate_input).squeeze(-1))
                else:
                    gate = torch.ones_like(innovation)

                cognitive_input = torch.cat(
                    tuple(x for x in (skill_embedding, problem_embedding, difficulty_value,
                                      correct_embedding,
                                      numeric, h_a) if x is not None), dim=-1
                )
                affect_input = torch.cat(
                    (current_observed_affect, correct_embedding, numeric, h_c), dim=-1
                )
                candidate_c = self.cognitive_cell(cognitive_input, h_c)
                candidate_a = self.affect_cell(affect_input, h_a)
                new_c = gate.unsqueeze(-1) * candidate_c + (1.0 - gate.unsqueeze(-1)) * h_c
                new_a = gate.unsqueeze(-1) * candidate_a + (1.0 - gate.unsqueeze(-1)) * h_a
                h_c = torch.where(active.unsqueeze(-1), new_c, h_c)
                h_a = torch.where(active.unsqueeze(-1), new_a, h_a)
                joint_after = torch.cat((h_c, h_a), dim=-1)
                affect_prediction = torch.sigmoid(self.affect_head(joint_after))
                behavior_affect_prediction = torch.sigmoid(
                    self.behavior_affect_head(joint_after)
                )
                affect_outputs.append(affect_prediction)
                behavior_affect_outputs.append(behavior_affect_prediction)
                gate_outputs.append(gate)
                innovation_outputs.append(innovation)
                output_state = joint_after
            else:
                cognitive_input = torch.cat(
                    tuple(x for x in (skill_embedding, problem_embedding, difficulty_value,
                                      correct_embedding,
                                      numeric) if x is not None), dim=-1
                )
                candidate_c = self.cognitive_cell(cognitive_input, h_c)
                h_c = torch.where(active.unsqueeze(-1), candidate_c, h_c)
                output_state = h_c

            target_embedding = self.skill_embedding(target_skill[:, index])
            target_problem_embedding = (
                self.problem_embedding(target_problem[:, index]) if self.use_problem else None
            )
            target_difficulty_value = (
                target_problem_difficulty[:, index].unsqueeze(-1)
                if self.use_difficulty else None
            )
            correct_outputs.append(
                self.correct_head(
                    torch.cat(
                        tuple(x for x in (output_state, target_embedding, target_problem_embedding,
                                          target_difficulty_value)
                              if x is not None), dim=-1
                    )
                ).squeeze(-1)
            )
            if self.explicit_state:
                mastery_logit = self.mastery_head(h_c).squeeze(-1)
                correct_outputs[-1] = correct_outputs[-1] + self.mastery_scale * mastery_logit
                if self.use_affect:
                    correct_outputs[-1] = correct_outputs[-1] + (
                        (affect_outputs[-1] - 0.5) * self.affect_response_coeff
                    ).sum(dim=-1)
                mastery_outputs.append(torch.sigmoid(mastery_logit))

        outputs: dict[str, Tensor | None] = {
            "correct_logits": torch.stack(correct_outputs, dim=1),
            "affect_pred": torch.stack(affect_outputs, dim=1) if affect_outputs else None,
            "behavior_affect_pred": (
                torch.stack(behavior_affect_outputs, dim=1)
                if behavior_affect_outputs else None
            ),
            "mastery": torch.stack(mastery_outputs, dim=1) if mastery_outputs else None,
            "gate": torch.stack(gate_outputs, dim=1) if gate_outputs else None,
            "innovation": torch.stack(innovation_outputs, dim=1) if innovation_outputs else None,
        }
        return outputs


class ResidualCognitiveAffectiveModel(nn.Module):
    """Strong item-aware KT backbone with an interpretable affect residual.

    The cognitive state is updated on every observed interaction.  The
    affective state is a separate, low-dimensional state and can be updated
    continuously or only when an innovation score crosses a learned,
    validation-budgeted gate.  The response logit is explicitly decomposed:

        full_logit = base_cognitive_logit + bounded_affect_residual

    The residual head is zero-initialised, so adding the affect module cannot
    destroy the cognitive baseline at the start of training.
    """

    def __init__(
        self,
        num_skills: int,
        mode: str = "residual_joint",
        skill_dim: int = 48,
        hidden_cognitive: int = 96,
        hidden_affect: int = 32,
        num_problems: int = 1,
        use_problem: bool = True,
        use_difficulty: bool = True,
        affect_source: str = "behavior",
        event_budget: float = 0.10,
        event_temperature: float = 0.15,
        event_policy: str = "soft",
        affect_residual_max: float = 0.75,
    ) -> None:
        super().__init__()
        if mode not in {"residual_joint", "residual_event"}:
            raise ValueError(f"unknown residual mode {mode}")
        if affect_source not in {"detector", "behavior", "both"}:
            raise ValueError(f"unknown affect source {affect_source}")
        if event_policy not in {"soft", "hard"}:
            raise ValueError(f"unknown event policy {event_policy}")
        if not 0.0 <= event_budget <= 1.0:
            raise ValueError("event_budget must be between 0 and 1")
        if event_temperature <= 0.0:
            raise ValueError("event_temperature must be positive")
        if affect_residual_max <= 0.0:
            raise ValueError("affect_residual_max must be positive")

        self.mode = mode
        self.event_policy = "always" if mode == "residual_joint" else event_policy
        self.affect_source = affect_source
        self.event_budget = float(event_budget)
        self.event_temperature = float(event_temperature)
        self.affect_residual_max = float(affect_residual_max)
        self.use_problem = bool(use_problem and num_problems > 1)
        self.use_difficulty = bool(use_difficulty)

        self.skill_embedding = nn.Embedding(num_skills, skill_dim, padding_idx=0)
        if self.use_problem:
            self.problem_embedding = nn.Embedding(
                num_problems, skill_dim, padding_idx=0
            )
        self.correct_embedding = nn.Embedding(2, 8)

        problem_dim = skill_dim if self.use_problem else 0
        difficulty_dim = 1 if self.use_difficulty else 0
        cognitive_input = skill_dim + problem_dim + difficulty_dim + 8 + 2
        self.cognitive_cell = nn.GRUCell(cognitive_input, hidden_cognitive)

        observed_dim = 8 if affect_source == "both" else 4
        affect_input = observed_dim + 8 + 2 + hidden_cognitive
        self.affect_cell = nn.GRUCell(affect_input, hidden_affect)

        state_dim = hidden_cognitive + hidden_affect
        target_dim = skill_dim + problem_dim + difficulty_dim
        base_input = hidden_cognitive + target_dim
        self.base_head = nn.Sequential(
            nn.Linear(base_input, hidden_cognitive),
            nn.ReLU(),
            nn.Linear(hidden_cognitive, 1),
        )
        # Target-KC-specific mastery readout.  It is auxiliary-supervised and
        # never confused with the full response logit.
        self.mastery_head = nn.Sequential(
            nn.Linear(hidden_cognitive + skill_dim, hidden_cognitive // 2),
            nn.Tanh(),
            nn.Linear(hidden_cognitive // 2, 1),
        )

        self.affect_state_head = nn.Sequential(
            nn.Linear(hidden_affect, hidden_affect),
            nn.Tanh(),
            nn.Linear(hidden_affect, 4),
        )
        self.detector_head = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_affect),
            nn.Tanh(),
            nn.Linear(hidden_affect, 4),
        )
        self.behavior_head = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_affect),
            nn.Tanh(),
            nn.Linear(hidden_affect, 4),
        )

        residual_input = state_dim + target_dim + 4
        self.affect_residual_head = nn.Sequential(
            nn.Linear(residual_input, hidden_affect),
            nn.Tanh(),
            nn.Linear(hidden_affect, 1),
        )
        # A zero residual is the exact starting point of the joint model.
        nn.init.zeros_(self.affect_residual_head[-1].weight)
        nn.init.zeros_(self.affect_residual_head[-1].bias)

        # Positive event weights keep the trigger score monotone in
        # interpretable innovation components.
        self.event_weights_raw = nn.Parameter(
            torch.tensor([1.5, 2.0, 0.5], dtype=torch.float32)
        )
        # Start from a conservative threshold; the validation-selected budget
        # penalty learns the operating point rather than hard-coding a rate.
        inverse_softplus_one = math.log(math.expm1(1.0))
        self.event_threshold_raw = nn.Parameter(
            torch.tensor(inverse_softplus_one, dtype=torch.float32)
        )
        self.hidden_cognitive = hidden_cognitive
        self.hidden_affect = hidden_affect

    def _observed_affect(
        self, batch: dict[str, Tensor], index: int
    ) -> tuple[Tensor, Tensor]:
        detector = batch["affect"][:, index]
        behavior = batch["behavior_affect"][:, index]
        if self.affect_source == "detector":
            return detector, detector
        if self.affect_source == "behavior":
            return behavior, behavior
        return torch.cat((detector, behavior), dim=-1), torch.cat(
            (detector, behavior), dim=-1
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        skill = batch["skill"]
        correct = batch["correct"]
        response = batch["response"]
        delta = batch["delta"]
        target_skill = batch["target_skill"]
        problem = batch["problem"]
        target_problem = batch["target_problem"]
        problem_difficulty = batch["problem_difficulty"]
        target_problem_difficulty = batch["target_problem_difficulty"]
        mask = batch["mask"]
        batch_size, steps = skill.shape
        device = skill.device

        h_c = torch.zeros(batch_size, self.hidden_cognitive, device=device)
        h_a = torch.zeros(batch_size, self.hidden_affect, device=device)
        full_logits: list[Tensor] = []
        base_logits: list[Tensor] = []
        residual_logits: list[Tensor] = []
        detector_predictions: list[Tensor] = []
        behavior_predictions: list[Tensor] = []
        affect_states: list[Tensor] = []
        mastery_logits: list[Tensor] = []
        gates: list[Tensor] = []
        soft_gates: list[Tensor] = []
        innovations: list[Tensor] = []
        event_components: list[Tensor] = []

        for index in range(steps):
            active = mask[:, index].bool()
            skill_t = self.skill_embedding(skill[:, index])
            problem_t = (
                self.problem_embedding(problem[:, index])
                if self.use_problem else None
            )
            difficulty_t = (
                problem_difficulty[:, index].unsqueeze(-1)
                if self.use_difficulty else None
            )
            correct_embedding = self.correct_embedding(correct[:, index].clamp(0, 1))
            numeric = torch.stack((response[:, index], delta[:, index]), dim=-1)

            state_before = torch.cat((h_c, h_a), dim=-1)
            target_skill_embedding = self.skill_embedding(target_skill[:, index])
            target_problem_embedding = (
                self.problem_embedding(target_problem[:, index])
                if self.use_problem else None
            )
            target_difficulty = (
                target_problem_difficulty[:, index].unsqueeze(-1)
                if self.use_difficulty else None
            )
            target_context = torch.cat(
                tuple(x for x in (
                    target_skill_embedding,
                    target_problem_embedding,
                    target_difficulty,
                ) if x is not None),
                dim=-1,
            )

            # Predict the current observed event from the state before update.
            affect_state_before = torch.sigmoid(self.affect_state_head(h_a))
            detector_before = torch.sigmoid(
                self.detector_head(torch.cat((state_before, skill_t), dim=-1))
            )
            behavior_before = torch.sigmoid(
                self.behavior_head(torch.cat((state_before, skill_t), dim=-1))
            )
            observed, observed_for_residual = self._observed_affect(batch, index)
            predicted_for_residual = (
                detector_before if self.affect_source == "detector"
                else behavior_before if self.affect_source == "behavior"
                else torch.cat((detector_before, behavior_before), dim=-1)
            )
            current_base_logit = self.base_head(
                torch.cat(tuple(x for x in (h_c, skill_t, problem_t, difficulty_t)
                                if x is not None), dim=-1)
            ).squeeze(-1)
            current_response = torch.sigmoid(current_base_logit)
            cognitive_residual = (
                correct[:, index].float() - current_response
            ).abs()
            affect_residual = (
                observed_for_residual - predicted_for_residual
            ).abs().mean(dim=-1)
            time_feature = (delta[:, index] / 13.312655).clamp(0.0, 1.0)
            event_feature = torch.stack(
                (cognitive_residual.detach(), affect_residual.detach(),
                 time_feature.detach()), dim=-1
            )
            event_weights = F.softplus(self.event_weights_raw)
            # The threshold is represented as a logit to keep optimization
            # stable; budget regularization is applied to the resulting gate.
            event_threshold = F.softplus(self.event_threshold_raw)
            event_logit = event_feature.mul(event_weights).sum(dim=-1)
            event_logit = event_logit - event_threshold
            soft_gate = torch.sigmoid(event_logit / self.event_temperature)
            if self.mode == "residual_joint":
                gate = torch.ones_like(soft_gate)
            elif self.event_policy == "hard":
                hard_gate = (soft_gate >= 0.5).to(soft_gate.dtype)
                gate = hard_gate + soft_gate - soft_gate.detach()
            else:
                gate = soft_gate

            cognitive_input = torch.cat(
                tuple(x for x in (
                    skill_t, problem_t, difficulty_t, correct_embedding, numeric
                ) if x is not None),
                dim=-1,
            )
            affect_input = torch.cat(
                (observed, correct_embedding, numeric, h_c), dim=-1
            )
            # Cognition always advances.  Only the affect state is event-gated.
            candidate_c = self.cognitive_cell(cognitive_input, h_c)
            candidate_a = self.affect_cell(affect_input, h_a)
            h_c = torch.where(active.unsqueeze(-1), candidate_c, h_c)
            new_a = h_a + gate.unsqueeze(-1) * (candidate_a - h_a)
            h_a = torch.where(active.unsqueeze(-1), new_a, h_a)

            state_after = torch.cat((h_c, h_a), dim=-1)
            affect_state = torch.sigmoid(self.affect_state_head(h_a))
            detector_prediction = torch.sigmoid(
                self.detector_head(torch.cat((state_after, target_skill_embedding), dim=-1))
            )
            behavior_prediction = torch.sigmoid(
                self.behavior_head(torch.cat((state_after, target_skill_embedding), dim=-1))
            )
            residual_input = torch.cat(
                (state_after, target_context, affect_state), dim=-1
            )
            response_residual = self.affect_residual_max * torch.tanh(
                self.affect_residual_head(residual_input).squeeze(-1)
            )
            base_logit = self.base_head(
                torch.cat(tuple(x for x in (h_c, target_context) if x is not None), dim=-1)
            ).squeeze(-1)
            mastery_logit = self.mastery_head(
                torch.cat((h_c, target_skill_embedding), dim=-1)
            ).squeeze(-1)
            full_logit = base_logit + response_residual

            full_logits.append(full_logit)
            base_logits.append(base_logit)
            residual_logits.append(response_residual)
            detector_predictions.append(detector_prediction)
            behavior_predictions.append(behavior_prediction)
            affect_states.append(affect_state)
            mastery_logits.append(mastery_logit)
            gates.append(gate)
            soft_gates.append(soft_gate)
            innovations.append(0.5 * (cognitive_residual + affect_residual))
            event_components.append(event_feature)

        return {
            "correct_logits": torch.stack(full_logits, dim=1),
            "base_correct_logits": torch.stack(base_logits, dim=1),
            "affect_residual_logits": torch.stack(residual_logits, dim=1),
            "signed_affect_contribution": torch.stack(residual_logits, dim=1),
            "detector_affect_pred": torch.stack(detector_predictions, dim=1),
            "behavior_affect_pred": torch.stack(behavior_predictions, dim=1),
            # Backward-compatible alias: ``affect_pred`` always denotes the
            # detector-derived target stored in ``target_affect``.  The
            # selected affect *input* source must not change supervision
            # semantics; behavior predictions have their own explicit head.
            "affect_pred": torch.stack(detector_predictions, dim=1),
            "affect_state": torch.stack(affect_states, dim=1),
            "mastery_logit": torch.stack(mastery_logits, dim=1),
            "mastery": torch.sigmoid(torch.stack(mastery_logits, dim=1)),
            "gate": torch.stack(gates, dim=1),
            "gate_soft": torch.stack(soft_gates, dim=1),
            "innovation": torch.stack(innovations, dim=1),
            "event_components": torch.stack(event_components, dim=1),
        }


class DualStateEventModel(nn.Module):
    """Leakage-aware dual-state KT with a residual-driven update budget.

    The model predicts the current interaction from the state *before* the
    update.  Those prediction residuals define an event score, so a trigger
    cannot inspect the next response or next affect label.  The cognitive
    state receives a cheap recurrent update at every step, while the event
    gate controls the additional affective-state update.  The
    response head receives the model's prediction of the next affect state,
    which makes the cognitive-affective path explicit and testable.
    """

    def __init__(
        self,
        num_skills: int,
        skill_dim: int = 48,
        hidden_cognitive: int = 96,
        hidden_affect: int = 48,
        event_temperature: float = 0.15,
        event_policy: str = "soft",
    ) -> None:
        super().__init__()
        if event_temperature <= 0:
            raise ValueError("event_temperature must be positive")
        if event_policy not in {"soft", "hard", "always"}:
            raise ValueError(f"unknown event policy {event_policy}")
        self.mode = "dual_always" if event_policy == "always" else "dual_event"
        self.event_policy = event_policy
        self.skill_embedding = nn.Embedding(num_skills, skill_dim, padding_idx=0)
        self.correct_embedding = nn.Embedding(2, 8)
        self.affect_encoder = nn.Sequential(
            nn.Linear(4, hidden_affect), nn.LayerNorm(hidden_affect), nn.Tanh()
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(2, hidden_affect), nn.LayerNorm(hidden_affect), nn.Tanh()
        )

        cognitive_input = skill_dim + 8 + hidden_affect + hidden_affect + hidden_affect
        self.cognitive_cell = nn.GRUCell(cognitive_input, hidden_cognitive)
        affect_input = hidden_affect + 8 + 2 + hidden_cognitive
        self.affect_cell = nn.GRUCell(affect_input, hidden_affect)

        self.pre_response_head = nn.Sequential(
            nn.Linear(hidden_cognitive + hidden_affect + skill_dim, hidden_cognitive),
            nn.Tanh(), nn.Linear(hidden_cognitive, 1),
        )
        self.pre_affect_head = nn.Sequential(
            nn.Linear(hidden_cognitive + hidden_affect + skill_dim, hidden_affect),
            nn.Tanh(), nn.Linear(hidden_affect, 4),
        )
        self.affect_head = nn.Sequential(
            nn.Linear(hidden_cognitive + hidden_affect + skill_dim, hidden_affect),
            nn.Tanh(), nn.Linear(hidden_affect, 4),
        )
        self.affect_to_response = nn.Sequential(
            nn.Linear(4, hidden_affect), nn.Tanh()
        )
        self.response_head = nn.Sequential(
            nn.Linear(hidden_cognitive + hidden_affect + skill_dim + hidden_affect, hidden_cognitive),
            nn.ReLU(), nn.Linear(hidden_cognitive, 1),
        )

        # Positive weights make the event score monotone in each interpretable
        # residual feature.  The threshold is learned under an explicit budget.
        self.event_weights_raw = nn.Parameter(torch.tensor([1.5, 2.0, 0.5]))
        self.event_bias = nn.Parameter(torch.tensor(-0.8))
        self.event_threshold = nn.Parameter(torch.tensor(0.8))
        self.event_temperature = float(event_temperature)
        self.hidden_cognitive = hidden_cognitive
        self.hidden_affect = hidden_affect

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        skill = batch["skill"]
        correct = batch["correct"]
        response = batch["response"]
        delta = batch["delta"]
        affect = batch["affect"]
        target_skill = batch["target_skill"]
        mask = batch["mask"]
        batch_size, steps = skill.shape
        device = skill.device
        h_c = torch.zeros(batch_size, self.hidden_cognitive, device=device)
        h_a = torch.zeros(batch_size, self.hidden_affect, device=device)
        correct_outputs: list[Tensor] = []
        affect_outputs: list[Tensor] = []
        gate_outputs: list[Tensor] = []
        soft_gate_outputs: list[Tensor] = []
        innovation_outputs: list[Tensor] = []

        # Seven days is the clipping horizon used while preparing the arrays.
        delta_scale = 13.312655
        for index in range(steps):
            active = mask[:, index].bool()
            skill_t = self.skill_embedding(skill[:, index])
            target_embedding = self.skill_embedding(target_skill[:, index])
            correct_embedding = self.correct_embedding(correct[:, index].clamp(0, 1))
            numeric = torch.stack((response[:, index], delta[:, index] / delta_scale), dim=-1)
            affect_embedding = self.affect_encoder(affect[:, index])
            numeric_embedding = self.numeric_encoder(numeric)

            state_before = torch.cat((h_c, h_a), dim=-1)
            current_response = torch.sigmoid(
                self.pre_response_head(torch.cat((state_before, skill_t), dim=-1)).squeeze(-1)
            )
            current_affect = torch.sigmoid(
                self.pre_affect_head(torch.cat((state_before, skill_t), dim=-1))
            )
            cognitive_residual = (correct[:, index].float() - current_response).abs()
            affect_residual = (affect[:, index] - current_affect).abs().mean(dim=-1)
            time_feature = (delta[:, index] / delta_scale).clamp(0.0, 1.0)
            innovation = 0.5 * (cognitive_residual + affect_residual)
            event_features = torch.stack(
                (cognitive_residual.detach(), affect_residual.detach(), time_feature.detach()), dim=-1
            )
            event_weights = F.softplus(self.event_weights_raw)
            event_logit = event_features.mul(event_weights).sum(dim=-1)
            event_logit = event_logit + self.event_bias - self.event_threshold
            soft_gate = torch.sigmoid(event_logit / self.event_temperature)
            if self.event_policy == "always":
                gate = torch.ones_like(soft_gate)
            elif self.event_policy == "hard":
                hard_gate = (soft_gate >= 0.5).to(soft_gate.dtype)
                # Forward values are discrete while gradients follow the soft
                # relaxation. This gives the event budget a usable gradient.
                gate = hard_gate + soft_gate - soft_gate.detach()
            else:
                gate = soft_gate

            cognitive_input = torch.cat(
                (skill_t, correct_embedding, numeric_embedding, affect_embedding, h_a), dim=-1
            )
            affect_input = torch.cat(
                (affect_embedding, correct_embedding, numeric, h_c), dim=-1
            )
            candidate_c = self.cognitive_cell(cognitive_input, h_c)
            # The cognitive state has a cheap recurrent update at every
            # interaction.  The event budget controls the additional affective
            # state update, so sparse triggering does not discard ordinary
            # learning evidence.
            new_c = candidate_c
            if self.event_policy == "hard" and not self.training:
                # In deployment-style evaluation, do not run the affect cell
                # for samples whose discrete event gate is closed.
                triggered = active & (soft_gate >= 0.5)
                new_a = h_a.clone()
                if bool(triggered.any()):
                    new_a[triggered] = self.affect_cell(
                        affect_input[triggered], h_a[triggered]
                    )
            else:
                candidate_a = self.affect_cell(affect_input, h_a)
                new_a = h_a + gate.unsqueeze(-1) * (candidate_a - h_a)
            h_c = torch.where(active.unsqueeze(-1), new_c, h_c)
            h_a = torch.where(active.unsqueeze(-1), new_a, h_a)

            next_state = torch.cat((h_c, h_a, target_embedding), dim=-1)
            next_affect = torch.sigmoid(self.affect_head(next_state))
            response_context = self.affect_to_response(next_affect)
            response_input = torch.cat((h_c, h_a, target_embedding, response_context), dim=-1)
            correct_outputs.append(self.response_head(response_input).squeeze(-1))
            affect_outputs.append(next_affect)
            gate_outputs.append(gate)
            soft_gate_outputs.append(soft_gate)
            innovation_outputs.append(innovation)

        return {
            "correct_logits": torch.stack(correct_outputs, dim=1),
            "affect_pred": torch.stack(affect_outputs, dim=1),
            "gate": torch.stack(gate_outputs, dim=1),
            "gate_soft": torch.stack(soft_gate_outputs, dim=1),
            "innovation": torch.stack(innovation_outputs, dim=1),
        }


def _loss(
    outputs: dict[str, Tensor | None],
    batch: dict[str, Tensor],
    affect_weight: float,
    behavior_affect_weight: float = 0.0,
    event_penalty_weight: float = 0.0,
    event_budget: float = 0.1,
    event_budget_weight: float = 0.0,
    mastery_loss_weight: float = 0.0,
    affect_state_loss_weight: float = 0.0,
    residual_penalty_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    mask = batch["mask"]
    denominator = mask.sum().clamp_min(1.0)
    correct_loss_all = F.binary_cross_entropy_with_logits(
        outputs["correct_logits"], batch["target_correct"], reduction="none"
    )
    correct_loss = (correct_loss_all * mask).sum() / denominator
    values: dict[str, float] = {"correct_loss": float(correct_loss.detach().cpu())}
    total = correct_loss
    # Detector and behavior proxies are different targets.  Prefer the
    # explicit detector head when available and retain ``affect_pred`` only
    # for older model variants that expose a single detector head.
    detector_prediction = outputs.get("detector_affect_pred")
    if detector_prediction is None:
        detector_prediction = outputs.get("affect_pred")
    if detector_prediction is not None:
        affect_loss_all = (
            (detector_prediction - batch["target_affect"]) ** 2
        ).mean(dim=-1)
        affect_loss = (affect_loss_all * mask).sum() / denominator
        total = total + affect_weight * affect_loss
        values["affect_loss"] = float(affect_loss.detach().cpu())
    if outputs.get("behavior_affect_pred") is not None and behavior_affect_weight > 0.0:
        behavior_loss_all = (
            (outputs["behavior_affect_pred"] - batch["target_behavior_affect"]) ** 2
        ).mean(dim=-1)
        behavior_loss = (behavior_loss_all * mask).sum() / denominator
        total = total + behavior_affect_weight * behavior_loss
        values["behavior_affect_loss"] = float(behavior_loss.detach().cpu())
    if outputs.get("mastery_logit") is not None and mastery_loss_weight > 0.0:
        mastery_loss_all = F.binary_cross_entropy_with_logits(
            outputs["mastery_logit"], batch["target_correct"], reduction="none"
        )
        mastery_loss = (mastery_loss_all * mask).sum() / denominator
        total = total + mastery_loss_weight * mastery_loss
        values["mastery_loss"] = float(mastery_loss.detach().cpu())
    if outputs.get("affect_state") is not None and affect_state_loss_weight > 0.0:
        state_loss_all = (
            (outputs["affect_state"] - batch["target_behavior_affect"]) ** 2
        ).mean(dim=-1)
        state_loss = (state_loss_all * mask).sum() / denominator
        total = total + affect_state_loss_weight * state_loss
        values["affect_state_loss"] = float(state_loss.detach().cpu())
    if outputs.get("affect_residual_logits") is not None and residual_penalty_weight > 0.0:
        residual_penalty = (
            outputs["affect_residual_logits"].abs() * mask
        ).sum() / denominator
        total = total + residual_penalty_weight * residual_penalty
        values["residual_penalty"] = float(residual_penalty.detach().cpu())
    if outputs["gate"] is not None and event_penalty_weight > 0.0:
        gate_loss = (outputs["gate"] * mask).sum() / denominator
        total = total + event_penalty_weight * gate_loss
        values["event_gate_loss"] = float(gate_loss.detach().cpu())
    if outputs["gate"] is not None and event_budget_weight > 0.0:
        gate_mean = (outputs["gate"] * mask).sum() / denominator
        budget_loss = (gate_mean - event_budget).square()
        total = total + event_budget_weight * budget_loss
        values["event_budget_loss"] = float(budget_loss.detach().cpu())
    values["loss"] = float(total.detach().cpu())
    return total, values


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else float("nan")


def _expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(y_score, edges[1:-1]), bins - 1)
    total = max(len(y_true), 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            error += selected.sum() / total * abs(
                float(y_true[selected].mean()) - float(y_score[selected].mean())
            )
    return float(error)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             affect_weight: float, behavior_affect_weight: float = 0.0,
             max_batches: int = 0, mastery_loss_weight: float = 0.0,
             affect_state_loss_weight: float = 0.0,
             residual_penalty_weight: float = 0.0) -> dict[str, Any]:
    model.eval()
    correct_scores: list[np.ndarray] = []
    correct_targets: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    affect_scores: list[np.ndarray] = []
    affect_targets: list[np.ndarray] = []
    behavior_affect_scores: list[np.ndarray] = []
    behavior_affect_targets: list[np.ndarray] = []
    mastery_scores: list[np.ndarray] = []
    affect_state_scores: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    soft_gates: list[np.ndarray] = []
    innovations: list[np.ndarray] = []
    losses: list[float] = []
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        outputs = model(batch)
        loss, _ = _loss(
            outputs, batch, affect_weight, behavior_affect_weight,
            mastery_loss_weight=mastery_loss_weight,
            affect_state_loss_weight=affect_state_loss_weight,
            residual_penalty_weight=residual_penalty_weight,
        )
        losses.append(float(loss.detach().cpu()))
        mask = batch["mask"].bool()
        scores = torch.sigmoid(outputs["correct_logits"])[mask].detach().cpu().numpy()
        targets = batch["target_correct"][mask].detach().cpu().numpy()
        correct_scores.append(scores)
        correct_targets.append(targets)
        if outputs.get("base_correct_logits") is not None:
            base_scores.append(
                torch.sigmoid(outputs["base_correct_logits"])[mask]
                .detach().cpu().numpy()
            )
        if outputs.get("affect_residual_logits") is not None:
            residual_values.append(
                outputs["affect_residual_logits"][mask].detach().cpu().numpy()
            )
        total_count += int(mask.sum().item())
        if outputs["affect_pred"] is not None:
            affect_scores.append(outputs["affect_pred"][mask].detach().cpu().numpy())
            affect_targets.append(batch["target_affect"][mask].detach().cpu().numpy())
            gates.append(outputs["gate"][mask].detach().cpu().numpy())
            if outputs.get("gate_soft") is not None:
                soft_gates.append(outputs["gate_soft"][mask].detach().cpu().numpy())
            innovations.append(outputs["innovation"][mask].detach().cpu().numpy())
            if outputs.get("behavior_affect_pred") is not None:
                behavior_affect_scores.append(
                    outputs["behavior_affect_pred"][mask].detach().cpu().numpy()
                )
                behavior_affect_targets.append(
                    batch["target_behavior_affect"][mask].detach().cpu().numpy()
                )
        if outputs.get("mastery") is not None:
            mastery_scores.append(outputs["mastery"][mask].detach().cpu().numpy())
        if outputs.get("affect_state") is not None:
            affect_state_scores.append(
                outputs["affect_state"][mask].detach().cpu().numpy()
            )

    y_score = np.concatenate(correct_scores) if correct_scores else np.empty(0)
    y_true = np.concatenate(correct_targets) if correct_targets else np.empty(0)
    metrics: dict[str, Any] = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "count": total_count,
        "accuracy": float(accuracy_score(y_true, y_score >= 0.5)) if total_count else float("nan"),
        "auc": _safe_auc(y_true, y_score) if total_count else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)) if total_count else float("nan"),
        "brier": float(brier_score_loss(y_true, y_score)) if total_count else float("nan"),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])) if total_count else float("nan"),
        "ece_15": _expected_calibration_error(y_true, y_score) if total_count else float("nan"),
    }
    if base_scores:
        base_score = np.concatenate(base_scores)
        metrics["base_auc"] = _safe_auc(y_true, base_score)
        metrics["base_accuracy"] = float(
            accuracy_score(y_true, base_score >= 0.5)
        )
    if residual_values:
        residual = np.concatenate(residual_values)
        metrics.update({
            "affect_residual_mean": float(residual.mean()),
            "affect_residual_abs_mean": float(np.abs(residual).mean()),
            "affect_residual_std": float(residual.std()),
            "affect_residual_positive_rate": float((residual > 0).mean()),
        })
    if affect_scores:
        affect_score = np.concatenate(affect_scores)
        affect_true = np.concatenate(affect_targets)
        metrics.update({
            "affect_rmse": float(math.sqrt(mean_squared_error(affect_true, affect_score))),
            "affect_mae": float(mean_absolute_error(affect_true, affect_score)),
            "affect_label_accuracy": float(
                accuracy_score(np.argmax(affect_true, axis=1), np.argmax(affect_score, axis=1))
            ),
            "affect_macro_f1": float(
                f1_score(np.argmax(affect_true, axis=1), np.argmax(affect_score, axis=1), average="macro")
            ),
        })
        affect_names = ("frustrated", "confused", "concentrating", "bored")
        per_affect: dict[str, dict[str, float]] = {}
        for index, name in enumerate(affect_names):
            truth = affect_true[:, index]
            prediction = affect_score[:, index]
            correlation = (
                float(np.corrcoef(truth, prediction)[0, 1])
                if np.std(truth) > 0 and np.std(prediction) > 0 else float("nan")
            )
            per_affect[name] = {
                "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
                "mae": float(mean_absolute_error(truth, prediction)),
                "r2": float(r2_score(truth, prediction)),
                "pearson": correlation,
            }
        metrics["affect_by_dimension"] = per_affect
        gate = np.concatenate(gates)
        innovation = np.concatenate(innovations)
        metrics.update({
            "gate_mean": float(gate.mean()),
            "event_rate_gt_0_5": float((gate > 0.5).mean()),
            "event_count_gt_0_5": int((gate > 0.5).sum()),
            "event_candidate_count": int(gate.size),
            "innovation_mean": float(innovation.mean()),
            "gate_innovation_corr": float(np.corrcoef(gate, innovation)[0, 1])
            if gate.size > 1 and np.std(gate) > 0 and np.std(innovation) > 0 else float("nan"),
        })
        if soft_gates:
            metrics["gate_soft_mean"] = float(np.concatenate(soft_gates).mean())
        event_policy = getattr(model, "event_policy", None)
        if event_policy is not None:
            metrics.update({
                "event_policy": event_policy,
                "affect_state_update_count": int((gate > 0.5).sum()),
                "affect_state_skip_rate": float(1.0 - (gate > 0.5).mean()),
            })
    if behavior_affect_scores:
        behavior_score = np.concatenate(behavior_affect_scores)
        behavior_true = np.concatenate(behavior_affect_targets)
        metrics.update({
            "behavior_affect_rmse": float(
                math.sqrt(mean_squared_error(behavior_true, behavior_score))
            ),
            "behavior_affect_mae": float(
                mean_absolute_error(behavior_true, behavior_score)
            ),
        })
    if mastery_scores:
        mastery = np.concatenate(mastery_scores)
        metrics["mastery_mean"] = float(mastery.mean())
        metrics["mastery_std"] = float(mastery.std())
        metrics["mastery_response_corr"] = (
            float(np.corrcoef(mastery, y_true)[0, 1])
            if mastery.size > 1 and np.std(mastery) > 0 and np.std(y_true) > 0
            else float("nan")
        )
    if affect_state_scores:
        state = np.concatenate(affect_state_scores)
        state_true = np.concatenate(behavior_affect_targets) if behavior_affect_targets else None
        metrics.update({
            "affect_state_mean": float(state.mean()),
            "affect_state_std": float(state.std()),
        })
        if state_true is not None and len(state_true) == len(state):
            metrics["affect_state_rmse"] = float(
                math.sqrt(mean_squared_error(state_true, state))
            )
    return metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    max_seq_len = int(metadata["max_seq_len"])
    num_skills = int(metadata["num_skills_with_padding"])
    num_problems = int(metadata.get("num_problems_with_padding", 1))
    train_data = WindowDataset(data_dir, "train", max_seq_len)
    val_data = WindowDataset(data_dir, "val", max_seq_len)
    test_data = None if args.skip_test else WindowDataset(data_dir, "test", max_seq_len)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = (
        DataLoader(test_data, shuffle=False, drop_last=False, **loader_kwargs)
        if test_data is not None else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.mode in {"residual_joint", "residual_event"}:
        model = ResidualCognitiveAffectiveModel(
            num_skills=num_skills,
            mode=args.mode,
            skill_dim=args.skill_dim,
            hidden_cognitive=args.hidden_cognitive,
            hidden_affect=args.hidden_affect,
            num_problems=num_problems,
            use_problem=args.use_problem,
            use_difficulty=args.use_difficulty,
            affect_source=args.affect_source,
            event_budget=args.event_budget,
            event_temperature=args.event_temperature,
            event_policy=args.event_policy,
            affect_residual_max=args.affect_residual_max,
        ).to(device)
    elif args.mode in {"dual_event", "dual_always"}:
        model = DualStateEventModel(
            num_skills=num_skills,
            skill_dim=args.skill_dim,
            hidden_cognitive=args.hidden_cognitive,
            hidden_affect=args.hidden_affect,
            event_temperature=args.event_temperature,
            event_policy="always" if args.mode == "dual_always" else args.event_policy,
        ).to(device)
    else:
        model = CognitiveAffectiveModel(
            num_skills=num_skills, mode=args.mode, skill_dim=args.skill_dim,
            hidden_cognitive=args.hidden_cognitive, hidden_affect=args.hidden_affect,
            num_problems=num_problems, use_problem=args.use_problem,
            use_difficulty=args.use_difficulty,
            affect_source=args.affect_source,
            explicit_state=args.explicit_state,
        ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({
            "init_checkpoint": str(args.init_checkpoint),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }, ensure_ascii=False), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=args.lr * 0.01
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_correct: list[float] = []
        epoch_affect: list[float] = []
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch)
                loss, loss_values = _loss(
                    outputs,
                    batch,
                    args.affect_loss_weight,
                    args.behavior_affect_loss_weight,
                    args.event_penalty_weight if args.mode == "event" else 0.0,
                    args.event_budget,
                    args.event_budget_weight
                    if args.mode in {"dual_event", "residual_event"} else 0.0,
                    args.mastery_loss_weight,
                    args.affect_state_loss_weight,
                    args.residual_penalty_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss_values["loss"])
            epoch_correct.append(loss_values["correct_loss"])
            if "affect_loss" in loss_values:
                epoch_affect.append(loss_values["affect_loss"])

        val_metrics = evaluate(
            model, val_loader, device, args.affect_loss_weight,
            args.behavior_affect_loss_weight, args.max_eval_batches,
            args.mastery_loss_weight, args.affect_state_loss_weight,
            args.residual_penalty_weight,
        )
        scheduler.step(val_metrics["auc"] if np.isfinite(val_metrics["auc"]) else -1.0)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "train_correct_loss": float(np.mean(epoch_correct)) if epoch_correct else float("nan"),
            "train_affect_loss": float(np.mean(epoch_affect)) if epoch_affect else None,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - epoch_start,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        val_auc = val_metrics["auc"] if np.isfinite(val_metrics["auc"]) else -float("inf")
        if val_auc > best_auc + args.min_delta:
            best_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model": model.state_dict(), "args": vars(args),
                "metadata": metadata, "epoch": epoch, "val": val_metrics,
            }, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(json.dumps({
                    "early_stop": True, "epoch": epoch,
                    "best_epoch": best_epoch, "best_val_auc": best_auc,
                    "patience": args.patience,
                }, ensure_ascii=False), flush=True)
                break

    best_path = output_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            model, test_loader, device, args.affect_loss_weight,
            args.behavior_affect_loss_weight, args.max_eval_batches,
            args.mastery_loss_weight, args.affect_state_loss_weight,
            args.residual_penalty_weight,
        )
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "mode": args.mode,
        "data_dir": str(data_dir),
        "device": str(device),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "test": test_metrics,
        "test_evaluated": test_metrics is not None,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
        "args": serializable_args,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"test": test_metrics, "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=[
            "cognitive", "joint", "event", "dual_event", "dual_always",
            "residual_joint", "residual_event",
        ],
        default="joint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--skill-dim", type=int, default=32)
    parser.add_argument("--hidden-cognitive", type=int, default=64)
    parser.add_argument("--hidden-affect", type=int, default=32)
    parser.add_argument(
        "--use-problem", action=argparse.BooleanOptionalAction, default=False,
        help="Use the strictly observed problem identity as an additional feature.",
    )
    parser.add_argument(
        "--use-difficulty", action=argparse.BooleanOptionalAction, default=False,
        help="Use training-only empirical-Bayes problem difficulty features.",
    )
    parser.add_argument(
        "--affect-source", choices=["detector", "behavior", "both"],
        default="detector",
        help="Use detector estimates, causal behavior profiles, or both as affect inputs.",
    )
    parser.add_argument(
        "--explicit-state", action=argparse.BooleanOptionalAction, default=False,
        help="Expose mastery and direct affect-to-response readouts.",
    )
    parser.add_argument("--affect-loss-weight", type=float, default=0.5)
    parser.add_argument("--behavior-affect-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--mastery-loss-weight", type=float, default=0.15,
        help="Auxiliary BCE weight for target-KC mastery readout.",
    )
    parser.add_argument(
        "--affect-state-loss-weight", type=float, default=0.10,
        help="Auxiliary MSE weight for the four-dimensional affect-state readout.",
    )
    parser.add_argument(
        "--residual-penalty-weight", type=float, default=0.001,
        help="L1 penalty on the bounded affect contribution.",
    )
    parser.add_argument(
        "--affect-residual-max", type=float, default=0.75,
        help="Maximum absolute affect contribution in response-logit units.",
    )
    parser.add_argument(
        "--event-penalty-weight",
        type=float,
        default=0.0,
        help="Optional penalty on the mean event gate; used only in event mode.",
    )
    parser.add_argument(
        "--event-budget",
        type=float,
        default=0.10,
        help="Target mean soft trigger probability for dual_event mode.",
    )
    parser.add_argument(
        "--event-budget-weight",
        type=float,
        default=5.0,
        help="Penalty weight for deviation from event-budget in dual_event mode.",
    )
    parser.add_argument(
        "--event-temperature",
        type=float,
        default=0.15,
        help="Temperature of the differentiable event gate in dual_event mode.",
    )
    parser.add_argument(
        "--event-policy",
        choices=["soft", "hard"],
        default="soft",
        help="Soft relaxation or a straight-through hard gate for dual_event.",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Validation epochs without a min-delta improvement before stopping.",
    )
    parser.add_argument(
        "--min-delta", type=float, default=1e-4,
        help="Minimum validation AUC improvement to reset early stopping.",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Do not read or evaluate the test split during model exploration.",
    )
    parser.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="Optional checkpoint used for a non-strict warm start.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    if not 0.0 <= args.event_budget <= 1.0:
        parser.error("event-budget must be between 0 and 1")
    if args.event_temperature <= 0:
        parser.error("event-temperature must be positive")
    if args.min_delta < 0 or args.affect_residual_max <= 0:
        parser.error("min-delta must be non-negative and affect-residual-max positive")
    if args.init_checkpoint is not None and not args.init_checkpoint.exists():
        parser.error(f"init-checkpoint does not exist: {args.init_checkpoint}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
