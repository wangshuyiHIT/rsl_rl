"""CPU-only secondary-teacher checks, including the actual audited AMP actor."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.modules.legacy_lateral_teacher import (
    LegacyLateralTeacher, pure_lateral_mask, weighted_lateral_teacher_mse,
)
from rsl_rl.networks import EmpiricalNormalization, MLP

TRAIN = Path(__file__).resolve().parents[3]
LEGACY = TRAIN / "logs/rsl_rl/scaler_amp/2026-09-11_13-42-47_amp_humanlike_20260911_stage1/model_7999.pt"
LEGACY_SHA = "9e988185c4eec427beb1dcf09e15d406b197f2d07f3640f2e2c3c353b5850fad"


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    with torch.random.fork_rng():
        torch.manual_seed(12)
        actor = MLP(860, 24, [512, 256, 128], activation="elu")
    normalizer = EmpiricalNormalization(860)
    normalizer._mean[:] = torch.linspace(-.3, .3, 860)
    normalizer._std[:] = torch.linspace(.8, 1.2, 860)
    normalizer._var[:] = normalizer._std.square()
    normalizer.count.fill_(12345)
    state = {"actor." + k: v for k, v in actor.state_dict().items()}
    state.update({"actor_obs_normalizer." + k: v for k, v in normalizer.state_dict().items()})
    state["critic.0.weight"] = torch.tensor([float("nan")])  # must never restore or validate a critic
    state["std"] = torch.ones(24)
    path = tmp_path_factory.mktemp("legacy_teacher") / "explicit.pt"
    torch.save({"model_state_dict": state, "iter": 7999,
                "optimizer_state_dict": {"not_used": "no optimizer is constructed"},
                "disc_state_dict": {"not_used": "no discriminator is constructed"}}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def teacher(source):
    return LegacyLateralTeacher.from_checkpoint(*source)


def observations(commands):
    x = torch.zeros(len(commands), 10, 86)
    x[:, -1, 6:9] = torch.tensor(commands)
    return x.reshape(len(commands), 860)


def test_explicit_sha_and_state_contract(source, teacher):
    path, sha = source
    assert teacher.identity["source_checkpoint_sha256"] == sha
    assert teacher.identity["source_iteration"] == 7999
    assert teacher.normalized and teacher.actor_obs_normalizer.count.item() == 12345
    assert not hasattr(teacher, "critic") and not hasattr(teacher, "optimizer")
    assert all(k.startswith(("actor.", "actor_obs_normalizer.")) for k in teacher.state_dict())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        LegacyLateralTeacher.from_checkpoint(path, "0" * 64)


def test_load_and_restore_do_not_advance_training_rng(source):
    before = torch.random.get_rng_state().clone()
    teacher = LegacyLateralTeacher.from_checkpoint(*source)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    LegacyLateralTeacher.from_checkpoint_state(teacher.checkpoint_state(), source[1])
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_actor_and_normalizer_remain_frozen_and_observations_unchanged(teacher):
    teacher.train()
    assert not teacher.training and not teacher.actor.training and not teacher.actor_obs_normalizer.training
    assert not any(p.requires_grad for p in teacher.parameters())
    x = observations([[0., .35, 0.], [0., -.35, 0.]])
    original = x.clone()
    count = teacher.actor_obs_normalizer.count.clone()
    actions = teacher(x)
    assert not actions.requires_grad and torch.isfinite(actions).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(teacher.actor_obs_normalizer.count, count, rtol=0, atol=0)


def test_latest_pure_lateral_mask_excludes_all_other_commands():
    x = observations([[0., .35, 0.], [0., -.35, 0.], [.5, .35, 0.], [0., .35, .5],
                      [0., 0., 0.], [.5, 0., 0.], [0., 0., .5], [0., .1, 0.]])
    x[:, 7] = .35  # first historical frame is never the gate
    assert pure_lateral_mask(x).tolist() == [True, True, False, False, False, False, False, False]


def test_default_zero_does_not_require_or_evaluate_teacher():
    x = observations([[0., .35, 0.]])
    student = torch.ones(1, 24, requires_grad=True)
    loss = weighted_lateral_teacher_mse(student, x, None)
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_nonlateral_zero_does_not_require_teacher():
    x = observations([[.5, 0., 0.], [0., 0., .5], [0., 0., 0.]])
    student = torch.ones(3, 24, requires_grad=True)
    loss = weighted_lateral_teacher_mse(student, x, None, coefficient=.05)
    loss.backward()
    assert loss.item() == 0. and torch.count_nonzero(student.grad) == 0


def test_weighted_loss_uses_only_lateral_denominator_and_student_gradients(teacher):
    x = observations([[0., .35, 0.], [.5, 0., 0.], [0., -.35, 0.], [0., .35, 0.]])
    student = (teacher(x) + torch.tensor([1., 100., 3., 200.])[:, None]).requires_grad_()
    weights = torch.tensor([1., 999., 3., 0.], requires_grad=True)
    loss = weighted_lateral_teacher_mse(student, x, teacher, coefficient=.05, sample_weights=weights)
    assert loss.item() == pytest.approx(.05 * (1. + 3. * 9.) / 4.)
    loss.backward()
    assert torch.count_nonzero(student.grad[[1, 3]]) == 0
    assert torch.count_nonzero(student.grad[[0, 2]]) > 0
    assert weights.grad is None
    assert all(p.grad is None for p in teacher.parameters())


def test_roundtrip_preserves_actor_and_identity_without_original_file(teacher, source, tmp_path):
    x = observations([[0., .35, 0.]])
    payload = teacher.checkpoint_state()
    payload["identity"]["source_checkpoint_path"] = str(tmp_path / "intentionally_unavailable.pt")
    restored = LegacyLateralTeacher.from_checkpoint_state(payload, source[1],
        expected_actor_state_sha256=teacher.identity["actor_state_sha256"])
    torch.testing.assert_close(restored(x), teacher(x), rtol=0, atol=0)
    assert restored.identity == payload["identity"] and not restored.training
    # Exported tensors/identity do not alias the active frozen module.
    payload["model_state_dict"]["actor.0.bias"].add_(1.)
    assert teacher.checkpoint_state()["identity"] == teacher.identity


def test_restore_rejects_state_coordinate_and_source_drift(teacher, source):
    payload = teacher.checkpoint_state()
    bad = copy.deepcopy(payload)
    bad["model_state_dict"]["actor.0.bias"][0] += .1
    with pytest.raises(ValueError, match="state digest mismatch"):
        LegacyLateralTeacher.from_checkpoint_state(bad, source[1])
    bad = copy.deepcopy(payload)
    bad["contract"]["yaw_indices"] = []
    with pytest.raises(ValueError, match="coordinate contract mismatch"):
        LegacyLateralTeacher.from_checkpoint_state(bad, source[1])
    with pytest.raises(ValueError, match="source identity mismatch"):
        LegacyLateralTeacher.from_checkpoint_state(payload, "0" * 64)
    with pytest.raises(ValueError, match="actor identity mismatch"):
        LegacyLateralTeacher.from_checkpoint_state(payload, source[1], expected_actor_state_sha256="0" * 64)


def test_frozen_model_mutation_is_detected(teacher):
    with torch.no_grad():
        teacher.actor[0].bias[0] += .1
    with pytest.raises(ValueError, match="changed after loading"):
        teacher.checkpoint_state()


def test_wrong_shapes_and_nonfinite_weights_rejected(teacher):
    with pytest.raises(ValueError, match="shape.*860"):
        teacher(torch.zeros(1, 859))
    x = observations([[0., .35, 0.]])
    with pytest.raises(ValueError, match="weights"):
        weighted_lateral_teacher_mse(torch.ones(1, 24), x, teacher, coefficient=.05,
                                     sample_weights=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="coefficient"):
        weighted_lateral_teacher_mse(torch.ones(1, 24), x, teacher, coefficient=-.1)


def test_optional_nominal_conditioning_zeros_every_history_frame_before_normalizer(source, teacher):
    nominal = LegacyLateralTeacher.from_checkpoint(*source, nominal_morphology=True)
    x = observations([[0., .35, 0.], [0., -.35, 0.]])
    frames = x.reshape(2, 10, 86)
    frames[..., 81:86] = torch.arange(100).reshape(2, 10, 5) / 100.
    original = x.clone()
    conditioned = x.clone()
    conditioned.reshape(2, 10, 86)[..., 81:86] = 0.
    assert not teacher.nominal_morphology and nominal.nominal_morphology
    torch.testing.assert_close(nominal(x), teacher(conditioned), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_nominal_conditioning_checkpoint_roundtrip_and_mode_guard(source):
    nominal = LegacyLateralTeacher.from_checkpoint(*source, nominal_morphology=True)
    payload = nominal.checkpoint_state()
    assert payload["nominal_morphology"] is True
    assert payload["contract"]["morphology_conditioning"] == "zero_all_10x5_slots_before_normalizer"
    restored = LegacyLateralTeacher.from_checkpoint_state(payload, source[1], expected_nominal_morphology=True)
    x = observations([[0., .35, 0.]])
    x.reshape(1,10,86)[...,81:86] = .5
    torch.testing.assert_close(restored(x), nominal(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="morphology-conditioning mismatch"):
        LegacyLateralTeacher.from_checkpoint_state(payload, source[1], expected_nominal_morphology=False)


def test_actual_7999_outputs_match_explicit_evaluator_adapter_exactly():
    if not LEGACY.is_file():
        pytest.skip("Audited local AMP checkpoint is not distributed with source tests")
    teacher = LegacyLateralTeacher.from_checkpoint(LEGACY, LEGACY_SHA)
    spec = importlib.util.spec_from_file_location("_amp_eval_teacher_parity", TRAIN / "experiments/evaluate_amp_robustness.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    generator = torch.Generator().manual_seed(20261006)
    current = torch.randn(16, 860, generator=generator) * .2
    signs = torch.ones(24)
    signs[[5, 6, 14, 15, 22, 23]] = -1.
    legacy = current.clone().reshape(16, 10, 86)
    for start in (9, 33, 57):
        legacy[..., start:start + 24] *= signs
    obs = TensorDict({"policy": legacy.reshape(16, 860), "critic": torch.zeros(16, 1490)}, batch_size=[16])
    checkpoint = torch.load(LEGACY, map_location="cpu", weights_only=False)
    reference, normalized = evaluator._actor(checkpoint, obs, "cpu")
    assert normalized
    with torch.no_grad():
        expected = reference.act_inference(obs) * signs
    torch.testing.assert_close(teacher(current), expected, rtol=0, atol=0)
    restored = LegacyLateralTeacher.from_checkpoint_state(teacher.checkpoint_state(), LEGACY_SHA)
    torch.testing.assert_close(restored(current), expected, rtol=0, atol=0)
