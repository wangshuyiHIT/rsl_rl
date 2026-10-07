"""Focused CPU contract and real PPO integration checks for action rehearsal."""
from __future__ import annotations

import copy
from unittest import mock

import pytest
import torch

from rsl_rl.modules.policy_rehearsal import FAMILIES, tensor_sha256, TARGET_GENERATION_BATCH_SIZE
from .test_amp_lateral_reference import runner, zero_teacher


def pools():
    result = {}
    for i, family in enumerate(FAMILIES):
        x = torch.randn((i + 1) * 3, 860) * .02
        x[:, 0] = i + 1
        x.reshape(-1, 10, 86)[:, :, 6:9] = torch.tensor(
            [[0., .35, 0.], [0., -.35, 0.], [-.35, 0., 0.]][i])
        result[family] = x
    return result


def installed():
    r, obs = runner()
    r.set_lateral_reference_policy(zero_teacher(r), include_backward=True,
                                   source_metadata={"source_checkpoint": {"sha256": "a" * 64}})
    return r, obs


def equal(a, b):
    if torch.is_tensor(a):
        assert tensor_sha256(a) == tensor_sha256(b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_balanced_sampling_equal_family_loss_and_no_distribution_mutation():
    r, obs = installed()
    data = pools()
    r.set_lateral_rehearsal(data, coefficient=.25, batch_per_family=7)
    with torch.no_grad():
        for p in r.alg.policy.actor.parameters(): p.zero_()
        # A simple input marker makes the actual per-family action MSE 1,4,9.
        r.alg.policy.actor[0].weight[0, 0] = 1.
        r.alg.policy.actor[-1].weight[:, 0] = 1.
        r.alg.policy.act(obs)
    before = (r.alg.policy.action_mean.clone(), r.alg.policy.action_std.clone())
    seen = []
    hook = r.alg.policy.actor.register_forward_pre_hook(lambda model, args: seen.append(args[0].detach().clone()))
    with mock.patch("torch.randint", wraps=torch.randint) as sample:
        loss = r.alg._lateral_rehearsal_loss()
    hook.remove()
    torch.testing.assert_close(loss, torch.tensor(.25 * (1 + 4 + 9) / 3))
    assert len(seen) == 1 and seen[0].shape == (21, 860)
    assert torch.equal(seen[0][:, 0], torch.tensor([1.] * 7 + [2.] * 7 + [3.] * 7))
    assert [call.args[:2] for call in sample.call_args_list] == [(3, (7,)), (6, (7,)), (9, (7,))]
    equal(before, (r.alg.policy.action_mean, r.alg.policy.action_std))


def test_fixed_targets_no_gradients_to_teacher_or_caller_and_cloned_source():
    r, _ = installed()
    data = {k: x.requires_grad_() for k, x in pools().items()}
    source = {"dataset_manifest_sha256": "b" * 64, "nested": {"label": "original"}}
    teacher = r.alg.lateral_reference_policy
    original = copy.deepcopy(teacher.state_dict())
    r.set_lateral_rehearsal(data, source_metadata=source)
    state = r.alg.lateral_rehearsal.state_dict()
    with torch.no_grad():
        for x in data.values(): x.fill_(123.)
    source["nested"]["label"] = "changed"
    equal(r.alg.lateral_rehearsal.state_dict(), state)
    # Loss consumes stored targets, not another teacher inference call.
    with mock.patch.object(teacher.actor, "forward", side_effect=AssertionError("teacher re-evaluated")):
        loss = r.alg._lateral_rehearsal_loss()
        loss.backward()
    assert sum(p.grad.abs().sum() for p in r.alg.policy.actor.parameters()) > 0
    assert all(p.grad is None for p in r.alg.policy.critic.parameters())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(x.grad is None for x in data.values())
    assert all(not x.requires_grad and x.grad_fn is None for x in r.alg.lateral_rehearsal.targets_by_family.values())
    equal(original, teacher.state_dict())
    for group in r.alg.optimizer.param_groups:
        assert not {id(p) for p in group["params"]} & {id(p) for p in teacher.parameters()}
    assert not teacher.training


@pytest.mark.parametrize("change", ["missing", "extra", "empty", "shape", "nan", "inf", "double",
                                   "coefficient", "bool_coefficient", "batch", "bool_batch", "source"])
def test_invalid_install_has_no_mutation(change):
    r, _ = installed()
    r.set_lateral_rehearsal(pools())
    original = r.alg.lateral_rehearsal
    data, kw = pools(), {}
    if change == "missing": del data["left"]
    elif change == "extra": data["forward"] = data["left"]
    elif change == "empty": data["left"] = torch.empty(0, 860)
    elif change == "shape": data["left"] = torch.zeros(3, 859)
    elif change == "nan": data["left"][0, 0] = float("nan")
    elif change == "inf": data["left"][0, 0] = float("inf")
    elif change == "double": data["left"] = data["left"].double()
    elif change == "coefficient": kw["coefficient"] = 0.
    elif change == "bool_coefficient": kw["coefficient"] = True
    elif change == "batch": kw["batch_per_family"] = 1.5
    elif change == "bool_batch": kw["batch_per_family"] = True
    else: kw["source_metadata"] = {"bad": float("nan")}
    rng = torch.random.get_rng_state()
    with pytest.raises(ValueError): r.set_lateral_rehearsal(data, **kw)
    assert r.alg.lateral_rehearsal is original
    equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("change", ["missing", "side_only", "train_mode", "grad_enabled", "optimizer", "normalized"])
def test_requires_independent_raw_frozen_backward_teacher(change):
    r, _ = installed()
    if change == "missing": r.alg.lateral_reference_policy = None
    elif change == "side_only": r.alg.lateral_reference_policy_include_backward = False
    elif change == "train_mode": r.alg.lateral_reference_policy.train()
    elif change == "grad_enabled": r.alg.lateral_reference_policy.requires_grad_(True)
    elif change == "normalized": r.alg.policy.actor_obs_normalization = True
    else: r.alg.optimizer.add_param_group({"params": list(r.alg.lateral_reference_policy.parameters())})
    with pytest.raises(ValueError): r.set_lateral_rehearsal(pools())
    assert r.alg.lateral_rehearsal is None


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_self_contained_checkpoint_roundtrip_and_playback(load_optimizer, tmp_path):
    r, obs = installed()
    r.alg.reference_policy = zero_teacher(r).eval().requires_grad_(False)
    r.alg.reference_policy_anchor_coef = .02
    r.set_lateral_rehearsal(pools(), coefficient=.37, batch_per_family=5,
                            source_metadata={"dataset": "external path no longer required"})
    path = tmp_path / "rehearsal.pt"
    r.save(str(path))
    expected = r.alg.lateral_rehearsal.state_dict()
    dest, _ = runner()
    dest.load(str(path), map_location="cpu", load_optimizer=load_optimizer)
    equal(dest.alg.lateral_rehearsal.state_dict(), expected)
    equal(dest.alg.reference_policy.state_dict(), r.alg.reference_policy.state_dict())
    equal(dest.alg.lateral_reference_policy.state_dict(), r.alg.lateral_reference_policy.state_dict())
    with torch.no_grad(): equal(dest.alg.policy.act_inference(obs), r.alg.policy.act_inference(obs))
    torch.manual_seed(93); first = r.alg._lateral_rehearsal_loss()
    torch.manual_seed(93); second = dest.alg._lateral_rehearsal_loss()
    equal(first, second)


@pytest.mark.parametrize("change", ["schema", "schema1", "bool_schema", "missing_field", "backend", "generation_batch", "empty", "target_shape", "target_nan",
                                   "observation_nan", "coefficient", "batch", "source", "tensor_sha", "teacher_sha",
                                   "teacher_changed", "teacher_absent", "side_only"])
def test_invalid_checkpoint_rejected_before_any_live_mutation(change, tmp_path):
    r, _ = installed(); r.set_lateral_rehearsal(pools())
    path = tmp_path / "invalid.pt"; r.save(str(path))
    payload = torch.load(path, weights_only=True)
    state = payload["lateral_rehearsal_state"]
    if change == "schema": state["schema_version"] = 3
    elif change == "schema1":
        state["schema_version"] = 1
        del state["target_generation_backend"], state["target_generation_batch_size"]
    elif change == "bool_schema": state["schema_version"] = True
    elif change == "missing_field": del state["source_metadata"]
    elif change == "backend": state["target_generation_backend"] = "cuda_tf32"
    elif change == "generation_batch": state["target_generation_batch_size"] = 1024
    elif change == "empty": state["observations_by_family"]["left"] = torch.zeros(0, 860)
    elif change == "target_shape": state["targets_by_family"]["left"] = torch.zeros(1, 24)
    elif change == "target_nan": state["targets_by_family"]["left"][0, 0] = float("nan")
    elif change == "observation_nan": state["observations_by_family"]["left"][0, 0] = float("nan")
    elif change == "coefficient": state["coefficient"] = float("inf")
    elif change == "batch": state["batch_per_family"] = 0
    elif change == "source": state["source_metadata"] = {"nan": float("nan")}
    elif change == "tensor_sha": state["targets_by_family"]["left"][0, 0] += 1.
    elif change == "teacher_sha": state["teacher_identity"]["state_sha256"] = "0" * 64
    elif change == "teacher_changed": payload["lateral_reference_policy_state_dict"]["actor.0.weight"][0, 0] += 1.
    elif change == "teacher_absent":
        for key in list(payload):
            if key.startswith("lateral_reference_policy_"): del payload[key]
    else: payload["lateral_reference_policy_config"]["include_backward"] = False
    torch.save(payload, path)
    dest, _ = installed(); dest.set_lateral_rehearsal(pools())
    before = {"policy": copy.deepcopy(dest.alg.policy.state_dict()), "optimizer": copy.deepcopy(dest.alg.optimizer.state_dict()),
              "D": copy.deepcopy(dest.alg.amp_discriminator.state_dict()), "Dopt": copy.deepcopy(dest.alg.disc_optimizer.state_dict()),
              "teacher": dest.alg.lateral_reference_policy, "rehearsal": dest.alg.lateral_rehearsal,
              "cursor": dest._next_learning_iteration}
    with pytest.raises(ValueError, match="schema1.*unverified" if change == "schema1" else None):
        dest.load(str(path), map_location="cpu")
    equal(dest.alg.policy.state_dict(), before["policy"])
    equal(dest.alg.optimizer.state_dict(), before["optimizer"])
    equal(dest.alg.amp_discriminator.state_dict(), before["D"])
    equal(dest.alg.disc_optimizer.state_dict(), before["Dopt"])
    assert dest.alg.lateral_reference_policy is before["teacher"]
    assert dest.alg.lateral_rehearsal is before["rehearsal"]
    assert dest._next_learning_iteration == before["cursor"]


def rollout_update(r, obs):
    with torch.no_grad():
        for _ in range(2):
            r.alg.act(obs)
            r.alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        r.alg.compute_returns(obs)
    return r.alg.update()


def test_disabled_path_rng_numeric_update_bitexact_and_legacy_load_clears(tmp_path):
    torch.manual_seed(171); a, obs = runner()
    torch.manual_seed(171); b, _ = runner()
    b.set_lateral_rehearsal(None)
    # Simulate the old object, which had no rehearsal attribute at all.
    del a.alg.lateral_rehearsal
    torch.manual_seed(714); loss_a = rollout_update(a, obs); rng_a = torch.random.get_rng_state()
    torch.manual_seed(714); loss_b = rollout_update(b, obs); rng_b = torch.random.get_rng_state()
    equal(loss_a, loss_b); equal(rng_a, rng_b)
    equal(a.alg.policy.state_dict(), b.alg.policy.state_dict())
    equal(a.alg.optimizer.state_dict(), b.alg.optimizer.state_dict())
    equal(a.alg.amp_discriminator.state_dict(), b.alg.amp_discriminator.state_dict())
    path = tmp_path / "legacy.pt"; a.save(str(path))
    assert "lateral_rehearsal_state" not in torch.load(path, weights_only=True)
    dest, _ = installed(); dest.set_lateral_rehearsal(pools())
    dest.load(str(path), map_location="cpu")
    assert dest.alg.lateral_rehearsal is None


def test_real_ppo_update_samples_every_minibatch_and_preserves_teacher_and_buffers():
    r, obs = installed(); r.set_lateral_rehearsal(pools(), batch_per_family=4)
    r.alg.num_learning_epochs = 2
    teacher = copy.deepcopy(r.alg.lateral_reference_policy.state_dict())
    rehearsal = r.alg.lateral_rehearsal
    with mock.patch.object(rehearsal, "loss", wraps=rehearsal.loss) as loss:
        result = rollout_update(r, obs)
    assert loss.call_count == 2 and result["amp/lateral_rehearsal"] > 0
    assert result["amp/lateral_reference_policy_anchor"] > 0
    equal(teacher, r.alg.lateral_reference_policy.state_dict())
    assert all(p.grad is None for p in r.alg.lateral_reference_policy.parameters())
    # Two rollout appends only; rehearsal never writes either D buffer.
    assert torch.all(r.alg.disc_obs_buffer._num_pushes == 2)
    assert torch.all(r.alg.disc_demo_obs_buffer._num_pushes == 2)
    assert not torch.cuda.is_initialized()


def test_replacing_or_disabling_bound_teacher_requires_explicit_clear():
    r, _ = installed(); r.set_lateral_rehearsal(pools())
    original = r.alg.lateral_reference_policy
    replacement = zero_teacher(r)
    with torch.no_grad(): replacement.actor[-1].bias.fill_(1.)
    with pytest.raises(ValueError, match="identity"):
        r.set_lateral_reference_policy(replacement, include_backward=True)
    with pytest.raises(ValueError, match="Clear lateral rehearsal"):
        r.set_lateral_reference_policy(None)
    assert r.alg.lateral_reference_policy is original
    with pytest.raises(ValueError, match="include_backward"):
        r.set_lateral_reference_policy(zero_teacher(r), include_backward=False)
    with pytest.raises(ValueError, match="identity"):
        r.set_lateral_reference_policy(zero_teacher(r), include_backward=True,
                                       source_metadata={"source_checkpoint": {"sha256": "c" * 64}})
    rehearsal = r.alg.lateral_rehearsal
    r.set_lateral_reference_policy(zero_teacher(r), include_backward=True,
                                   source_metadata={"source_checkpoint": {"sha256": "a" * 64},
                                                    "installation_request": "new receipt, same original teacher"})
    assert r.alg.lateral_rehearsal is rehearsal
    r.set_lateral_rehearsal(None)
    r.set_lateral_reference_policy(None)
    assert r.alg.lateral_rehearsal is None and r.alg.lateral_reference_policy is None


def test_nonzero_fixed_targets_are_actual_raw_teacher_predictions_with_morphology():
    r, _ = runner()
    teacher = copy.deepcopy(r.alg.policy)
    r.set_lateral_reference_policy(teacher, include_backward=True)
    data = pools()
    for value in data.values(): value.reshape(-1, 10, 86)[:, :, -5:] = .2
    with torch.no_grad():
        expected = {}
        for k, value in data.items():
            padded = torch.zeros(32, 860)
            padded[:len(value)] = value
            expected[k] = teacher.actor(padded)[:len(value)].clone()
    r.set_lateral_rehearsal(data)
    equal(expected, r.alg.lateral_rehearsal.targets_by_family)
    assert any(torch.count_nonzero(v) > 0 for v in expected.values())


@pytest.mark.parametrize("allow_tf32", [False, True])
def test_generation_uses_independent_cpu_float32_clone_and_leaves_flags_and_inputs_unchanged(allow_tf32):
    r, _ = runner()
    r.set_lateral_reference_policy(copy.deepcopy(r.alg.policy), include_backward=True)
    teacher = r.alg.lateral_reference_policy
    data = {k: torch.randn(35 + i, 860) * .05 for i, k in enumerate(FAMILIES)}
    before_data = {k: v.clone() for k, v in data.items()}
    before_teacher = copy.deepcopy(teacher.state_dict())
    seen = []
    handle = teacher.actor.register_forward_pre_hook(
        lambda model, args: seen.append((model is teacher.actor, args[0].device.type, args[0].dtype,
                                         len(args[0]), torch.is_grad_enabled())))
    original_precision = torch.get_float32_matmul_precision()
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        precision = torch.get_float32_matmul_precision()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            r.set_lateral_rehearsal(data)
        assert torch.backends.cuda.matmul.allow_tf32 is allow_tf32
        assert torch.backends.cudnn.allow_tf32 is allow_tf32
        assert torch.get_float32_matmul_precision() == precision
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
    handle.remove()
    assert seen == [(False, "cpu", torch.float32, TARGET_GENERATION_BATCH_SIZE, False)] * 6
    equal(data, before_data)
    equal(teacher.state_dict(), before_teacher)
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in data.values())
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad for p in teacher.parameters())
    with torch.no_grad():
        for k, value in data.items():
            expected = []
            for chunk in value.split(32):
                padded = torch.zeros(32, 860)
                padded[:len(chunk)] = chunk
                expected.append(teacher.actor(padded)[:len(chunk)])
            equal(torch.cat(expected), r.alg.lateral_rehearsal.targets_by_family[k])
    state = r.alg.lateral_rehearsal.state_dict()
    assert state["schema_version"] == 2
    assert state["target_generation_backend"] == "cpu_float32"
    assert state["target_generation_batch_size"] == 32
    assert not torch.cuda.is_initialized()
