"""Training-only balanced raw-observation action rehearsal for canonical AMP.

This data never enters rollout storage or discriminator replay. Targets are
computed once by an independently frozen teacher; checkpoint restore preserves
their exact bytes instead of recomputing device-dependent predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

import torch

from rsl_rl.modules.actor_critic import ActorCritic


FAMILIES = ("left", "right", "backward")
TARGET_GENERATION_BACKEND = "cpu_float32"
TARGET_GENERATION_BATCH_SIZE = 32


def _json_copy(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Rehearsal {name} must be a finite JSON dictionary")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Rehearsal {name} must be a finite JSON dictionary") from error
    if result != value:
        raise ValueError(f"Rehearsal {name} must roundtrip as JSON without conversion")
    return result


def tensor_sha256(value: torch.Tensor) -> str:
    """Portable identity including dtype, shape, and contiguous raw bytes."""
    h = hashlib.sha256()
    h.update(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
    h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def teacher_identity(teacher: ActorCritic, source_metadata: dict) -> dict:
    """Bind complete frozen weights plus an available original checkpoint SHA."""
    source = _json_copy(source_metadata, "teacher source_metadata")
    states = {key: tensor_sha256(value) for key, value in sorted(teacher.state_dict().items())}
    source_checkpoint = source.get("source_checkpoint", {})
    if not isinstance(source_checkpoint, dict):
        raise ValueError("Rehearsal teacher source_checkpoint must be a dictionary")
    source_sha = source_checkpoint.get("sha256")
    if source_sha is not None and (not isinstance(source_sha, str) or len(source_sha) != 64
                                   or any(c not in "0123456789abcdef" for c in source_sha)):
        raise ValueError("Rehearsal teacher source checkpoint SHA must be lowercase SHA-256")
    return {"state_sha256": hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
            "source_checkpoint_sha256": source_sha}


def validate_models(student, teacher, include_backward):
    if include_backward is not True or teacher is None:
        raise ValueError("Rehearsal requires a lateral teacher with include_backward=True")
    for model in (student, teacher):
        if type(model) is not ActorCritic:
            raise ValueError("Rehearsal requires an unadapted raw ActorCritic")
        actor = [x for x in model.actor.modules() if isinstance(x, torch.nn.Linear)]
        critic = [x for x in model.critic.modules() if isinstance(x, torch.nn.Linear)]
        if (model.actor_obs_normalization or model.critic_obs_normalization or model.state_dependent_std
                or not isinstance(model.actor_obs_normalizer, torch.nn.Identity)
                or not isinstance(model.critic_obs_normalizer, torch.nn.Identity)
                or actor[0].in_features != 860 or actor[-1].out_features != 24
                or critic[0].in_features != 1490
                or any(p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError("Rehearsal requires raw float32 860/1490 observations and 24 actions")
    if (teacher is student or teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters())
            or {id(p) for p in teacher.parameters()} & {id(p) for p in student.parameters()}
            or any(not torch.isfinite(v).all() for v in teacher.state_dict().values())):
        raise ValueError("Rehearsal teacher must be independent, finite, frozen and in eval mode")


def _config(coefficient, batch_per_family):
    if (type(coefficient) not in (float, int) or not math.isfinite(coefficient) or coefficient <= 0
            or type(batch_per_family) is not int or batch_per_family <= 0):
        raise ValueError("Rehearsal requires a finite positive coefficient and positive integer batch_per_family")
    return float(coefficient), batch_per_family


def _tensors(values, width, name, device, counts=None):
    if not isinstance(values, Mapping) or set(values) != set(FAMILIES):
        raise ValueError(f"Rehearsal {name} requires exactly left/right/backward families")
    result = {}
    for family in FAMILIES:
        value = values[family]
        if (not torch.is_tensor(value) or value.layout != torch.strided or value.dtype != torch.float32
                or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != width
                or (counts is not None and value.shape[0] != counts[family])
                or not torch.isfinite(value).all()):
            raise ValueError(f"Rehearsal {name}.{family} must be finite nonempty float32 [N,{width}]")
        result[family] = value.detach().to(device).clone().contiguous()
    return result


class LateralPolicyRehearsal:
    """Immutable-to-caller fixed targets, sampled equally per command family."""

    @classmethod
    def prepare(cls, student, teacher, include_backward, observations_by_family, *, coefficient=.25,
                batch_per_family=64, source_metadata=None, teacher_source_metadata=None):
        validate_models(student, teacher, include_backward)
        coefficient, batch_per_family = _config(coefficient, batch_per_family)
        source = _json_copy({} if source_metadata is None else source_metadata, "source_metadata")
        identity = teacher_identity(teacher, {} if teacher_source_metadata is None else teacher_source_metadata)
        device = next(student.parameters()).device
        cpu_observations = _tensors(observations_by_family, 860, "observations", "cpu")
        # Installation is independent of the student's CUDA/TF32 mode. Use a
        # separate CPU clone so neither the live teacher nor global precision
        # flags are mutated. Match the source replay's 32-row GEMM shape even
        # for a short final batch; this actor is strictly row-independent.
        cpu_teacher = copy.deepcopy(teacher).to(device="cpu", dtype=torch.float32)
        cpu_teacher.eval().requires_grad_(False)
        targets = {}
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            for family, value in cpu_observations.items():
                predictions = []
                for chunk in value.split(TARGET_GENERATION_BATCH_SIZE):
                    padded = torch.zeros(TARGET_GENERATION_BATCH_SIZE, 860, dtype=torch.float32, device="cpu")
                    padded[:len(chunk)] = chunk
                    predictions.append(cpu_teacher.actor(padded)[:len(chunk)].clone())
                targets[family] = torch.cat(predictions)
        observations = _tensors(cpu_observations, 860, "observations", device)
        targets = _tensors(targets, 24, "targets", device,
                           {family: len(value) for family, value in observations.items()})
        return cls(observations, targets, coefficient, batch_per_family, source, identity)

    def __init__(self, observations, targets, coefficient, batch_per_family, source, identity):
        self.observations_by_family = observations
        self.targets_by_family = targets
        self.coefficient = coefficient
        self.batch_per_family = batch_per_family
        self.source_metadata = source
        self.teacher_identity = identity

    def validate_binding(self, student, teacher, include_backward, teacher_source_metadata):
        validate_models(student, teacher, include_backward)
        if self.teacher_identity != teacher_identity(teacher, teacher_source_metadata):
            raise ValueError("Rehearsal frozen teacher identity mismatch; clear/reinstall rehearsal explicitly")

    def loss(self, student):
        observations, targets = [], []
        for family in FAMILIES:
            pool = self.observations_by_family[family]
            indices = torch.randint(len(pool), (self.batch_per_family,), device=pool.device)
            observations.append(pool[indices])
            targets.append(self.targets_by_family[family][indices])
        # Calling actor directly leaves the on-policy action distribution intact.
        error = (student.actor(torch.cat(observations)) - torch.cat(targets)).square()
        return self.coefficient * error.reshape(3, self.batch_per_family, 24).mean(dim=(1, 2)).mean()

    def state_dict(self):
        observations = {k: v.detach().cpu().clone() for k, v in self.observations_by_family.items()}
        targets = {k: v.detach().cpu().clone() for k, v in self.targets_by_family.items()}
        return {"schema_version": 2, "coefficient": self.coefficient, "batch_per_family": self.batch_per_family,
                "target_generation_backend": TARGET_GENERATION_BACKEND,
                "target_generation_batch_size": TARGET_GENERATION_BATCH_SIZE,
                "observations_by_family": observations, "targets_by_family": targets,
                "source_metadata": copy.deepcopy(self.source_metadata),
                "teacher_identity": copy.deepcopy(self.teacher_identity),
                "tensor_sha256": {"observations": {k: tensor_sha256(v) for k, v in observations.items()},
                                  "targets": {k: tensor_sha256(v) for k, v in targets.items()}}}

    @classmethod
    def from_state_dict(cls, state, *, student, teacher, include_backward, teacher_source_metadata):
        """Prepare restored state without mutating any runner, model or optimizer."""
        keys = {"schema_version", "coefficient", "batch_per_family", "observations_by_family", "targets_by_family",
                "source_metadata", "teacher_identity", "tensor_sha256", "target_generation_backend",
                "target_generation_batch_size"}
        if isinstance(state, dict) and type(state.get("schema_version")) is int and state["schema_version"] == 1:
            raise ValueError("Rehearsal schema1 used unverified device-dependent targets; regenerate CPU float32 schema2 targets")
        if (not isinstance(state, dict) or set(state) != keys
                or type(state["schema_version"]) is not int or state["schema_version"] != 2):
            raise ValueError("Invalid rehearsal checkpoint schema")
        if (state["target_generation_backend"] != TARGET_GENERATION_BACKEND
                or type(state["target_generation_batch_size"]) is not int
                or state["target_generation_batch_size"] != TARGET_GENERATION_BATCH_SIZE):
            raise ValueError("Rehearsal requires cpu_float32 target generation with fixed 32-row batches")
        validate_models(student, teacher, include_backward)
        coefficient, batch = _config(state["coefficient"], state["batch_per_family"])
        source = _json_copy(state["source_metadata"], "source_metadata")
        identity = _json_copy(state["teacher_identity"], "teacher_identity")
        if identity != teacher_identity(teacher, teacher_source_metadata):
            raise ValueError("Rehearsal frozen teacher identity mismatch")
        device = next(student.parameters()).device
        observations = _tensors(state["observations_by_family"], 860, "observations", device)
        targets = _tensors(state["targets_by_family"], 24, "targets", device,
                           {family: len(value) for family, value in observations.items()})
        hashes = {"observations": {k: tensor_sha256(v) for k, v in observations.items()},
                  "targets": {k: tensor_sha256(v) for k, v in targets.items()}}
        if _json_copy(state["tensor_sha256"], "tensor_sha256") != hashes:
            raise ValueError("Rehearsal saved tensor SHA mismatch")
        return cls(observations, targets, coefficient, batch, source, identity)
