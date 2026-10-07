"""Optional frozen legacy Scaler actor for a lateral-only auxiliary loss.

This module is deliberately not imported by PPO, runners or module __init__.
Inputs are CURRENT, RAW, frame-major Scaler policy histories (10 x 86).
The old actor receives six-yaw-adapted histories before its original normalizer.
Outputs are raw action means in CURRENT coordinates, exactly as in the explicit
checkpoint evaluator, before its action clamp. No critic or optimizer is restored.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from torch import nn

from rsl_rl.networks import EmpiricalNormalization, MLP

_SCHEMA = "scaler_legacy_lateral_teacher_v1"
_YAW_INDICES = (5, 6, 14, 15, 22, 23)
_CONTRACT = {
    "current_input": "raw_frame_major_10x86",
    "teacher_coordinates": "legacy_six_yaw",
    "output_coordinates": "current_raw_action_mean",
    "yaw_indices": list(_YAW_INDICES),
    "adapted_frame_slices": [[9, 33], [33, 57], [57, 81]],
    "layer_dimensions": [860, 512, 256, 128, 24],
    "activation": "elu",
    "normalizer_eps": .01,
    "action_clipping": None,
}


def _checked_sha(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected an explicit lowercase SHA256 digest")
    return value


def _contract(nominal_morphology: bool) -> dict:
    return {**copy.deepcopy(_CONTRACT),
            "morphology_conditioning": ("zero_all_10x5_slots_before_normalizer" if nominal_morphology
                                        else "preserve_current_raw_morphology_features")}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Stable digest of names, tensor dtypes, shapes and exact CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Teacher state must contain only tensors: {name}")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _frames(raw_observations: torch.Tensor) -> torch.Tensor:
    if raw_observations.ndim != 2 or raw_observations.shape[-1] != 860 or not raw_observations.is_floating_point():
        raise ValueError("Expected floating current raw observations with shape (N, 860)")
    return raw_observations.reshape(-1, 10, 86)


class LegacyLateralTeacher(nn.Module):
    """An actor plus its frozen normalizer, with an explicit six-yaw adapter."""

    def __init__(self, normalized: bool, identity: dict, *, nominal_morphology: bool = False):
        super().__init__()
        if not isinstance(nominal_morphology, bool):
            raise ValueError("nominal_morphology must be boolean")
        self._nominal_morphology = nominal_morphology
        # Loading/restoring a frozen actor must not advance training's RNG.
        # Construct on CPU and touch no accelerator generator here.
        with torch.random.fork_rng(devices=[]):
            self.actor = MLP(860, 24, [512, 256, 128], activation="elu")
        self.actor_obs_normalizer = EmpiricalNormalization(860, eps=.01) if normalized else nn.Identity()
        signs = torch.ones(24)
        signs[list(_YAW_INDICES)] = -1.
        self.register_buffer("_yaw_signs", signs, persistent=False)
        self._identity = copy.deepcopy(identity)
        self.normalized = bool(normalized)
        self._loaded = False
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """Remain frozen even if a containing module switches to training."""
        return super().train(False)

    @property
    def identity(self) -> dict:
        return copy.deepcopy(self._identity)

    @property
    def nominal_morphology(self) -> bool:
        return self._nominal_morphology

    @classmethod
    def from_checkpoint(cls, path: str | Path, expected_sha256: str, *, device="cpu",
                        nominal_morphology: bool = False):
        expected_sha256 = _checked_sha(expected_sha256)
        source = Path(path).expanduser().resolve(strict=True)
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise ValueError("Legacy teacher source checkpoint SHA256 mismatch")
        # Load the exact bytes hashed above, not a second read of a mutable path.
        checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)
        full_state = checkpoint["model_state_dict"]
        state = {k: v for k, v in full_state.items()
                 if k.startswith(("actor.", "actor_obs_normalizer."))}
        normalized = any(k.startswith("actor_obs_normalizer.") for k in state)
        identity = {"source_checkpoint_path": str(source), "source_checkpoint_sha256": expected_sha256,
                    "source_iteration": checkpoint.get("iter"), "actor_state_sha256": _state_sha256(state)}
        teacher = cls(normalized, identity, nominal_morphology=nominal_morphology)
        teacher._load_actor_state(state)
        return teacher.to(device)

    def _load_actor_state(self, state):
        expected = self.state_dict()
        if state.keys() != expected.keys():
            raise ValueError("Teacher actor/normalizer state keys disagree with the 860/24 contract")
        for key, tensor in state.items():
            if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
                raise ValueError(f"Teacher actor/normalizer shape or dtype mismatch: {key}")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite teacher tensor: {key}")
        if self.normalized:
            if torch.any(state["actor_obs_normalizer._std"] < 0) or torch.any(state["actor_obs_normalizer._var"] < 0):
                raise ValueError("Teacher normalizer variance/std must be non-negative")
            if state["actor_obs_normalizer.count"].item() < 0:
                raise ValueError("Teacher normalizer count must be non-negative")
        self.load_state_dict(state, strict=True)
        if _state_sha256(self.state_dict()) != self._identity["actor_state_sha256"]:
            raise ValueError("Teacher actor state digest mismatch")
        self._loaded = True

    @torch.no_grad()
    def forward(self, current_raw_observations: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            raise ValueError("Construct a verified teacher with from_checkpoint or from_checkpoint_state")
        frames = _frames(current_raw_observations).clone()
        if frames.device != self._yaw_signs.device or frames.dtype != self._yaw_signs.dtype:
            raise ValueError("Teacher and raw observations must share device and dtype")
        if not torch.isfinite(frames).all():
            raise ValueError("Nonfinite teacher observation")
        for start in (9, 33, 57):
            frames[..., start:start + 24] *= self._yaw_signs
        if self.nominal_morphology:
            # Optional conditioning, not a change to the physical student's
            # morphology. Defaults to preserving all observed morphology slots.
            frames[..., 81:86] = 0.
        legacy = frames.reshape(-1, 860)
        result = self.actor(self.actor_obs_normalizer(legacy)) * self._yaw_signs
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite teacher action")
        return result

    def checkpoint_state(self) -> dict:
        """Self-contained teacher payload; no dependence on source-file presence.

        Its source identity records the original verified checkpoint. The state
        digest detects accidental payload changes; this is not an independent
        re-verification of unavailable original checkpoint bytes.
        """
        if not self._loaded:
            raise ValueError("Cannot save an unverified teacher")
        state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        if _state_sha256(state) != self._identity["actor_state_sha256"]:
            raise ValueError("Frozen teacher state changed after loading")
        return {"schema": _SCHEMA, "contract": _contract(self.nominal_morphology), "normalized": self.normalized,
                "nominal_morphology": self.nominal_morphology,
                "identity": self.identity, "model_state_dict": state}

    @classmethod
    def from_checkpoint_state(cls, payload: dict, expected_source_sha256: str, *, device="cpu",
                              expected_actor_state_sha256: str | None = None,
                              expected_nominal_morphology: bool | None = None):
        """Restore a trusted training-checkpoint payload, retaining its identity."""
        expected_source_sha256 = _checked_sha(expected_source_sha256)
        nominal = payload.get("nominal_morphology")
        if not isinstance(nominal, bool):
            raise ValueError("Teacher checkpoint must explicitly record nominal_morphology")
        if expected_nominal_morphology is not None and nominal is not expected_nominal_morphology:
            raise ValueError("Teacher restore morphology-conditioning mismatch")
        if payload.get("schema") != _SCHEMA or payload.get("contract") != _contract(nominal):
            raise ValueError("Teacher checkpoint schema/coordinate contract mismatch")
        identity = payload["identity"]
        if identity["source_checkpoint_sha256"] != expected_source_sha256:
            raise ValueError("Teacher restore source identity mismatch")
        state_sha = _checked_sha(identity["actor_state_sha256"])
        if expected_actor_state_sha256 is not None and state_sha != _checked_sha(expected_actor_state_sha256):
            raise ValueError("Teacher restore actor identity mismatch")
        if not isinstance(payload["normalized"], bool):
            raise ValueError("Teacher normalization flag must be boolean")
        teacher = cls(payload["normalized"], identity, nominal_morphology=nominal)
        teacher._load_actor_state(payload["model_state_dict"])
        return teacher.to(device)


def pure_lateral_mask(current_raw_observations: torch.Tensor, *, command_threshold: float = .1,
                      other_axis_threshold: float = .1) -> torch.Tensor:
    """Gate using only the latest raw command; older history cannot enable it."""
    if any(not math.isfinite(x) or x < 0. for x in (command_threshold, other_axis_threshold)):
        raise ValueError("Command thresholds must be finite and non-negative")
    command = _frames(current_raw_observations)[:, -1, 6:9].detach()
    if not torch.isfinite(command).all():
        raise ValueError("Nonfinite latest command")
    return ((command[:, 1].abs() > command_threshold)
            & (command[:, 0].abs() < other_axis_threshold)
            & (command[:, 2].abs() < other_axis_threshold))


def weighted_lateral_teacher_mse(student_mean: torch.Tensor, current_raw_observations: torch.Tensor,
                                 teacher: LegacyLateralTeacher | None, *, coefficient: float = 0.,
                                 sample_weights: torch.Tensor | None = None,
                                 command_threshold: float = .1, other_axis_threshold: float = .1) -> torch.Tensor:
    """Optional MSE, normalized by active positive weights, with default zero.

    No nonlateral samples or teacher gradients enter this loss. The caller owns
    morphology/confidence weights and serialization of these loss settings.
    Comparisons use raw action means, without a hidden clamp or a phase target.
    """
    if not math.isfinite(coefficient) or coefficient < 0.:
        raise ValueError("Teacher coefficient must be finite and non-negative")
    frames = _frames(current_raw_observations)
    if student_mean.shape != (frames.shape[0], 24) or student_mean.device != frames.device:
        raise ValueError("Student actions must have shape (N, 24) on the observation device")
    if not torch.isfinite(student_mean).all():
        raise ValueError("Nonfinite student actions")
    if coefficient == 0.:
        return student_mean.sum() * 0.
    mask = pure_lateral_mask(current_raw_observations, command_threshold=command_threshold,
                             other_axis_threshold=other_axis_threshold)
    weights = student_mean.new_ones(frames.shape[0]) if sample_weights is None else sample_weights.detach().to(student_mean)
    if weights.shape != mask.shape or not torch.isfinite(weights).all() or torch.any(weights < 0.):
        raise ValueError("Teacher sample weights must be finite, non-negative and shaped (N,)")
    active = mask & (weights > 0.)
    if not torch.any(active):
        return student_mean.sum() * 0.
    if teacher is None:
        raise ValueError("A teacher is required for an enabled lateral loss")
    target = teacher(current_raw_observations[active])
    error = (student_mean[active] - target).square().mean(dim=-1)
    return coefficient * (weights[active] * error).sum() / weights[active].sum()
