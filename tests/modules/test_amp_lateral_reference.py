"""CPU tests for the independent raw current-axis lateral AMP teacher."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPOAMP
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import AMPRunner
from rsl_rl.storage import CircularBuffer, RolloutStorage


TRAIN = Path(__file__).resolve().parents[3]
GROUPS = {"policy": ["policy"], "critic": ["critic"],
          "discriminator": ["disc"], "discriminator_demonstration": ["disc_demo"]}


def observations(commands):
    frames = torch.zeros(len(commands), 10, 86)
    frames[:, :, 6:9] = torch.tensor(commands)[:, None]
    return TensorDict({"policy": frames.flatten(1), "critic": torch.zeros(len(commands), 1490),
                       "disc": torch.ones(len(commands), 3, 2),
                       "disc_demo": torch.full((len(commands), 3, 2), 2.)}, batch_size=[len(commands)])


def runner():
    obs = observations([[0., .35, 0.], [0., -.35, 0.], [.5, 0., 0.], [0., 0., .5]])
    policy = ActorCritic(obs, GROUPS, 24, actor_hidden_dims=[8], critic_hidden_dims=[8],
                         actor_obs_normalization=False, critic_obs_normalization=False)
    cfg = {"loss_type": "LSGAN", "disc_obs_dim": 2, "disc_obs_steps": 3,
           "disc_trunk_weight_decay": 0., "disc_linear_weight_decay": 0.,
           "disc_learning_rate": .001, "disc_max_grad_norm": .5,
           "amp_discriminator": {"hidden_dims": [4], "task_style_lerp": .6},
           "step_dt": .02, "grad_penalty_scale": 1.}
    alg = PPOAMP(policy, RolloutStorage("rl", 4, 2, obs, [24], "cpu"),
                 CircularBuffer(4, 4, "cpu"), CircularBuffer(4, 4, "cpu"),
                 num_learning_epochs=1, num_mini_batches=1, amp_cfg=cfg, device="cpu")
    result = object.__new__(AMPRunner)
    result.alg, result.alg_cfg = alg, {"rnd_cfg": None}
    result.logger = NS(save_model=mock.Mock())
    result.current_learning_iteration = 700
    result._last_completed_learning_iteration = 700
    result._next_learning_iteration = 701
    return result, obs


def zero_teacher(r):
    teacher = copy.deepcopy(r.alg.policy)
    with torch.no_grad():
        for value in teacher.parameters():
            value.zero_()
    return teacher


def test_pure_lateral_mask_latest_frame_only_and_zero_gradient_elsewhere():
    r, _ = runner()
    r.set_lateral_reference_policy(zero_teacher(r))
    commands = [[0., .35, 0.], [0., -.35, 0.], [0., 0., 0.], [.5, .35, 0.],
                [0., .35, .5], [0., .1, 0.], [.1, .35, 0.], [0., .35, -.1]]
    obs = observations(commands)
    obs["policy"].reshape(8, 10, 86)[:, :-1, 6:9] = torch.tensor([.5, 0., .5])
    student = torch.ones(8, 24, requires_grad=True)
    loss = r.alg._lateral_reference_policy_anchor_loss(obs, student)
    assert loss.item() == .5
    loss.backward()
    assert torch.count_nonzero(student.grad[:2]) == 48
    assert torch.count_nonzero(student.grad[2:]) == 0
    assert all(p.grad is None for p in r.alg.lateral_reference_policy.parameters())
    inactive = observations(commands[2:])
    means = torch.ones(6, 24, requires_grad=True)
    zero = r.alg._lateral_reference_policy_anchor_loss(inactive, means)
    assert zero.item() == 0.
    zero.backward()
    assert torch.equal(means.grad, torch.zeros_like(means))


@pytest.mark.parametrize("include_backward", [False, True])
def test_backward_is_opt_in_and_forward_stand_mixed_and_boundary_rows_have_zero_gradient(include_backward):
    r, _ = runner()
    r.set_lateral_reference_policy(zero_teacher(r), include_backward=include_backward)
    obs = observations([[0., .35, 0.], [0., -.35, 0.], [-.35, 0., 0.],
                        [.35, 0., 0.], [0., 0., 0.], [-.1, 0., 0.],
                        [-.35, .1, 0.], [-.35, 0., -.1], [-.35, .2, .2]])
    obs["policy"].reshape(9, 10, 86)[:, :-1, 6:9] = torch.tensor([.6, 0., .5])
    means = torch.ones(9, 24, requires_grad=True)
    r.alg._lateral_reference_policy_anchor_loss(obs, means).backward()
    expected = torch.tensor([True, True, include_backward, False, False, False, False, False, False])
    assert torch.equal(means.grad.abs().sum(dim=1) > 0, expected)


def test_actual_morphology_reaches_teacher_and_primary_band_weighting_is_reused():
    r, _ = runner()
    teacher = zero_teacher(r)
    original = copy.deepcopy(teacher.state_dict())
    source = {"checkpoint": "model_250.pt", "sha256": "a" * 64, "axis": "current"}
    rng = torch.random.get_rng_state().clone()
    r.set_lateral_reference_policy(teacher, source_metadata=source)
    assert torch.equal(rng, torch.random.get_rng_state())
    source["axis"] = "mutated outside"
    assert r.alg.lateral_reference_policy_source_metadata["axis"] == "current"
    obs = observations([[0., .35, 0.], [0., -.35, 0.], [.5, 0., 0.]])
    obs["policy"].reshape(3, 10, 86)[:, :, -5:] = torch.tensor([0., .2, .9])[:, None, None]
    before = obs.clone()
    student = torch.tensor([1., 3., 100.])[:, None].repeat(1, 24).requires_grad_()
    seen = []
    handle = r.alg.lateral_reference_policy.actor.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    loss = r.alg._lateral_reference_policy_anchor_loss(obs, student)
    handle.remove()
    weight = .02 + .98 * torch.exp(torch.tensor(-.2**2 / (2*.1**2)))
    torch.testing.assert_close(loss, .5 * (1 + 9 * weight) / (1 + weight))
    torch.testing.assert_close(seen[0], obs["policy"][:2], rtol=0, atol=0)
    for key in obs.keys():
        torch.testing.assert_close(obs[key], before[key], rtol=0, atol=0)
    for key, value in original.items():
        torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)
    assert teacher.training and not r.alg.lateral_reference_policy.training


def actual_mirror():
    old_path = TRAIN / "scaler_lab/scaler_lab/tasks/amp1/mdp/symmetry/scaler.py"
    spec = importlib.util.spec_from_file_location("_amp_mirror_constants", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    path = TRAIN / "scaler_lab/scaler_lab/tasks/amp2/symmetry.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("scaler_lab"))]
    scope = {name: getattr(old, name) for name in ("_MORPH_DIM", "_NUM_JOINTS", "_switch_joints_left_right")}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["compute_symmetric_states"]


@pytest.mark.parametrize("include_backward", [False, True])
def test_real_frame_major_mirror_batch_selects_both_sides_without_nonside_gradients(include_backward):
    r, _ = runner()
    r.set_lateral_reference_policy(zero_teacher(r), include_backward=include_backward)
    obs = observations([[0., .35, 0.], [-.35, 0., 0.], [.5, 0., 0.]])
    env = NS(cfg=NS(observations=NS(policy=NS(history_length=10), critic=NS(history_length=10))))
    mirrored, _ = actual_mirror()(env, obs=obs)
    assert mirrored["policy"][3, -86 + 7] == -.35
    means = torch.ones(6, 24, requires_grad=True)
    r.alg._lateral_reference_policy_anchor_loss(mirrored, means).backward()
    expected = torch.tensor([True, include_backward, False, True, include_backward, False])
    assert torch.equal(means.grad.abs().sum(dim=1) > 0, expected)


@pytest.mark.parametrize("include_backward", [False, True])
def test_two_teachers_checkpoint_roundtrip_preserves_full_state_config_and_source(tmp_path, include_backward):
    r, obs = runner()
    primary = zero_teacher(r)
    with torch.no_grad():
        primary.actor[-1].bias.fill_(.3)
    r.alg.reference_policy = primary.eval().requires_grad_(False)
    r.alg.reference_policy_anchor_coef = .02
    r.alg.reference_policy_anchor_radius = .35
    r.alg.reference_policy_anchor_floor = .05
    r.alg.reference_policy_preserve_morphology = True
    r.alg.reference_policy_anchor_exclude_lateral_threshold = .1
    lateral = zero_teacher(r)
    with torch.no_grad():
        lateral.actor[-1].bias.fill_(.8)
    r.set_lateral_reference_policy(lateral, coefficient=.7, radius=.12, floor=.03, command_threshold=.08,
                                   source_metadata={"checkpoint": "/frozen/model250.pt", "sha256": "b"*64},
                                   include_backward=include_backward)
    path = tmp_path / "checkpoint.pt"
    r.save(str(path), infos={"separate": True})
    payload = torch.load(path, weights_only=True)
    assert payload["lateral_reference_policy_config"]["schema_version"] == 2
    assert payload["lateral_reference_policy_config"]["include_backward"] is include_backward
    for reset_d in (False, True):
        dest, _ = runner()
        old_d = copy.deepcopy(dest.alg.amp_discriminator.state_dict())
        infos = dest.load(str(path), load_discriminator=not reset_d, map_location="cpu")
        assert infos == {"separate": True}
        assert dest.current_learning_iteration == 701 and dest._last_completed_learning_iteration == 700
        assert dest._next_learning_iteration == 701
        for prefix in ("reference", "lateral_reference"):
            teacher = getattr(dest.alg, prefix + "_policy")
            for key, value in payload[prefix + "_policy_state_dict"].items():
                torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)
            assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
        for attr in ("anchor_coef", "anchor_radius", "anchor_floor", "command_threshold", "source_metadata", "include_backward"):
            name = "lateral_reference_policy_" + attr
            assert getattr(dest.alg, name) == getattr(r.alg, name)
        with torch.no_grad():
            torch.testing.assert_close(dest.alg.lateral_reference_policy.act_inference(obs),
                                       r.alg.lateral_reference_policy.act_inference(obs), rtol=0, atol=0)
        teacher_ids = {id(p) for model in (dest.alg.reference_policy, dest.alg.lateral_reference_policy) for p in model.parameters()}
        assert not teacher_ids.intersection(id(p) for group in dest.alg.optimizer.param_groups for p in group["params"])
        if reset_d:
            for key, value in old_d.items():
                torch.testing.assert_close(dest.alg.amp_discriminator.state_dict()[key], value, rtol=0, atol=0)


def test_legacy_load_clears_lateral_and_default_path_adds_no_checkpoint_keys(tmp_path):
    source, obs = runner()
    assert source.alg._lateral_reference_policy_anchor_loss(obs, torch.ones(4, 24)) is None
    path = tmp_path / "legacy.pt"
    source.save(str(path))
    assert not any(key.startswith("lateral_reference") for key in torch.load(path, weights_only=True))
    dest, _ = runner()
    dest.set_lateral_reference_policy(zero_teacher(dest))
    dest.load(str(path), map_location="cpu")
    assert dest.alg.lateral_reference_policy is None
    assert dest.alg.lateral_reference_policy_anchor_coef == 0
    assert dest.alg.lateral_reference_policy_source_metadata is None
    assert dest.alg.lateral_reference_policy_include_backward is False


def test_version1_teacher_restores_side_only_without_inheriting_new_backward_flag(tmp_path):
    source, _ = runner()
    source.set_lateral_reference_policy(zero_teacher(source))
    path = tmp_path / "version1.pt"
    source.save(str(path))
    payload = torch.load(path, weights_only=True)
    payload["lateral_reference_policy_config"]["schema_version"] = 1
    del payload["lateral_reference_policy_config"]["include_backward"]
    torch.save(payload, path)
    dest, _ = runner()
    dest.set_lateral_reference_policy(zero_teacher(dest), include_backward=True)
    dest.load(str(path), map_location="cpu")
    assert dest.alg.lateral_reference_policy_include_backward is False
    for key, value in payload["lateral_reference_policy_state_dict"].items():
        torch.testing.assert_close(dest.alg.lateral_reference_policy.state_dict()[key], value, rtol=0, atol=0)
    obs = observations([[-.35, 0., 0.]])
    means = torch.ones(1, 24, requires_grad=True)
    loss = dest.alg._lateral_reference_policy_anchor_loss(obs, means)
    assert loss.item() == 0.
    loss.backward()
    assert torch.count_nonzero(means.grad) == 0
    migrated = tmp_path / "version2.pt"
    dest.save(str(migrated))
    cfg = torch.load(migrated, weights_only=True)["lateral_reference_policy_config"]
    assert cfg["schema_version"] == 2 and cfg["include_backward"] is False


@pytest.mark.parametrize("value", [0, 1, None, "true", "False"])
def test_include_backward_requires_explicit_bool(value):
    r, _ = runner()
    with pytest.raises(ValueError, match="must be a bool"):
        r.set_lateral_reference_policy(zero_teacher(r), include_backward=value)
    assert r.alg.lateral_reference_policy is None


@pytest.mark.parametrize("change", ["unknown_schema", "bool_schema", "v1_extra_field", "v2_missing_field", "v2_nonbool"])
def test_backward_schema_mismatch_rejected_before_policy_or_teacher_mutation(tmp_path, change):
    source, _ = runner()
    source.set_lateral_reference_policy(zero_teacher(source), include_backward=True)
    path = tmp_path / "invalid.pt"
    source.save(str(path))
    payload = torch.load(path, weights_only=True)
    cfg = payload["lateral_reference_policy_config"]
    if change == "unknown_schema": cfg["schema_version"] = 3
    elif change == "bool_schema": cfg["schema_version"] = True
    elif change == "v1_extra_field": cfg["schema_version"] = 1
    elif change == "v2_missing_field": del cfg["include_backward"]
    else: cfg["include_backward"] = 1
    torch.save(payload, path)
    dest, _ = runner()
    original = copy.deepcopy(dest.alg.policy.state_dict())
    with pytest.raises(ValueError, match="configuration"):
        dest.load(str(path), map_location="cpu")
    assert dest.alg.lateral_reference_policy is None
    for key, value in original.items():
        torch.testing.assert_close(dest.alg.policy.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["normalized", "dtype", "nonfinite", "coefficient", "source"])
def test_setter_rejects_incompatible_teacher_without_replacing_existing_one(change):
    r, _ = runner()
    teacher = zero_teacher(r)
    r.set_lateral_reference_policy(teacher)
    original = r.alg.lateral_reference_policy
    kwargs = {}
    if change == "normalized": teacher.actor_obs_normalization = True
    elif change == "dtype": teacher.double()
    elif change == "nonfinite":
        with torch.no_grad():
            teacher.actor[0].weight[0, 0] = float("nan")
    elif change == "coefficient": kwargs["coefficient"] = 0.
    else: kwargs["source_metadata"] = {"tensor": torch.ones(1)}
    with pytest.raises(ValueError):
        r.set_lateral_reference_policy(teacher, **kwargs)
    assert r.alg.lateral_reference_policy is original


@pytest.mark.parametrize("change", ["missing_state", "missing_config", "missing_source", "nan_coef", "bad_radius",
                                      "bad_floor", "zero_threshold", "morph_false", "nan_source", "shape"])
def test_invalid_checkpoint_teacher_rejected_before_live_state_changes(tmp_path, change):
    source, _ = runner()
    source.set_lateral_reference_policy(zero_teacher(source))
    path = tmp_path / "checkpoint.pt"
    source.save(str(path))
    payload = torch.load(path, weights_only=True)
    cfg = payload["lateral_reference_policy_config"]
    if change.startswith("missing_"):
        key = {"state": "state_dict", "config": "config", "source": "source_metadata"}[change.removeprefix("missing_")]
        del payload["lateral_reference_policy_" + key]
    elif change == "nan_coef": cfg["coefficient"] = float("nan")
    elif change == "bad_radius": cfg["radius"] = -1.
    elif change == "bad_floor": cfg["floor"] = 1.1
    elif change == "zero_threshold": cfg["command_threshold"] = 0.
    elif change == "morph_false": cfg["preserve_morphology"] = False
    elif change == "nan_source": payload["lateral_reference_policy_source_metadata"] = {"bad": float("nan")}
    else: payload["lateral_reference_policy_state_dict"]["actor.0.weight"] = torch.zeros(1, 1)
    torch.save(payload, path)
    dest, _ = runner()
    before = copy.deepcopy(dest.alg.policy.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        dest.load(str(path), map_location="cpu")
    for key, value in before.items():
        torch.testing.assert_close(dest.alg.policy.state_dict()[key], value, rtol=0, atol=0)


def test_real_cpu_update_uses_both_losses_and_never_updates_teachers():
    torch.manual_seed(18)
    r, obs = runner()
    r.alg.reference_policy = zero_teacher(r).eval().requires_grad_(False)
    r.alg.reference_policy_anchor_coef = .02
    r.alg.reference_policy_preserve_morphology = True
    r.alg.reference_policy_anchor_exclude_lateral_threshold = .1
    r.set_lateral_reference_policy(zero_teacher(r))
    before = copy.deepcopy(r.alg.lateral_reference_policy.state_dict())
    with torch.no_grad():
        for _ in range(2):
            r.alg.act(obs)
            r.alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        r.alg.compute_returns(obs)
    losses = r.alg.update()
    assert losses["amp/reference_policy_anchor"] > 0
    assert losses["amp/lateral_reference_policy_anchor"] > 0
    for key, value in before.items():
        torch.testing.assert_close(r.alg.lateral_reference_policy.state_dict()[key], value, rtol=0, atol=0)
    assert all(p.grad is None for p in r.alg.lateral_reference_policy.parameters())
    assert all(p.grad is None for p in r.alg.reference_policy.parameters())
    assert not torch.cuda.is_initialized()
