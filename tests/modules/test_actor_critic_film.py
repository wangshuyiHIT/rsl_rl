"""CPU checks for the Scaler full-information, critic-only FiLM ablation."""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.modules import ActorCritic, ActorCriticFiLMCritic
from rsl_rl.utils import resolve_callable


GROUPS = {"policy": ["policy"], "critic": ["critic"]}
WIDTHS = (32, 24, 16)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations(batch: int = 4) -> TensorDict:
    generator = torch.Generator(device="cpu").manual_seed(2026)
    return TensorDict(
        {
            "policy": torch.randn(batch, 860, generator=generator),
            "critic": torch.randn(batch, 1490, generator=generator),
        },
        batch_size=[batch],
    )


def make_model(obs=None, cls=ActorCriticFiLMCritic, **kwargs):
    if obs is None:
        obs = observations()
    config = dict(actor_hidden_dims=WIDTHS, critic_hidden_dims=WIDTHS, activation="elu")
    config.update(kwargs)
    return cls(obs, GROUPS, 24, **config)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("state_dependent_std", [False, True])
def test_identity_initialization_matches_plain_actor_and_critic(normalized, state_dependent_std):
    obs = observations()
    config = dict(
        actor_obs_normalization=normalized,
        critic_obs_normalization=normalized,
        state_dependent_std=state_dependent_std,
        noise_std_type="log",
    )
    torch.manual_seed(91)
    plain = make_model(obs, cls=ActorCritic, **config)
    torch.manual_seed(91)
    film = make_model(obs, **config)
    # Ordinary layers have the same initialization, not just the same widths.
    for key, value in plain.actor.state_dict().items():
        torch.testing.assert_close(value, film.actor.state_dict()[key], rtol=0, atol=0)
    for key, value in plain.critic.state_dict().items():
        torch.testing.assert_close(value, film.critic.backbone.state_dict()[key], rtol=0, atol=0)
    plain.update_normalization(obs)
    film.update_normalization(obs)
    torch.testing.assert_close(film.evaluate(obs), plain.evaluate(obs), rtol=0, atol=0)
    torch.testing.assert_close(film.act_inference(obs), plain.act_inference(obs), rtol=0, atol=0)
    assert film.act(obs).shape == (4, 24)
    assert film.get_actions_log_prob(film.action_mean).shape == (4,)
    assert film.entropy.shape == (4,)


def test_context_slices_every_frame_in_oldest_to_newest_order():
    model = make_model()
    frames = torch.full((2, 10, 149), -999.0)
    expected = torch.arange(100, dtype=torch.float32).reshape(2, 10, 5)
    frames[:, :, 81:86] = expected
    context = model.critic.extract_morphology_context(frames.flatten(1))
    torch.testing.assert_close(context, expected.flatten(1), rtol=0, atol=0)
    torch.testing.assert_close(context[:, -5:], expected[:, -1], rtol=0, atol=0)


def test_context_is_extracted_before_empirical_normalization():
    obs = observations()
    model = make_model(obs, critic_obs_normalization=True)
    model.update_normalization(obs)
    captured = []
    handle = model.critic.context_encoder.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        model.evaluate(obs)
    finally:
        handle.remove()
    expected = obs["critic"].reshape(4, 10, 149)[:, :, 81:86].flatten(1)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    normalized_context = model.critic.extract_morphology_context(model.critic_obs_normalizer(obs["critic"]))
    assert not torch.allclose(captured[0], normalized_context)


def test_value_gradient_reaches_film_heads_then_context_encoder():
    torch.manual_seed(47)
    obs = observations()
    model = make_model(obs)
    optimizer = torch.optim.SGD(model.critic.parameters(), lr=0.03)
    target = torch.linspace(-0.8, 1.1, 4).unsqueeze(-1)
    loss = (model.evaluate(obs) - target).square().mean()
    loss.backward()
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0 for head in model.critic.film_heads)
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    # Identity heads necessarily block encoder gradients on the very first
    # backward pass; one real update opens the conditioning path.
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in model.critic.context_encoder.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (model.evaluate(obs) - target).square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.critic.context_encoder.parameters())
    assert model.critic.backbone[0].weight.grad.abs().sum() > 0

    # Hold the entire ordinary critic input fixed: the trained FiLM path itself
    # must respond to context, rather than relying on concatenated input alone.
    context = model.critic.extract_morphology_context(obs["critic"]).detach().requires_grad_()
    value = model.critic(obs["critic"], context)
    context_gradient, = torch.autograd.grad(value.sum(), context)
    assert torch.isfinite(context_gradient).all()
    assert context_gradient.abs().sum() > 0
    assert not torch.allclose(value, model.critic(obs["critic"], context + 0.5))


def test_actor_does_not_read_critic_context_or_film_parameters():
    obs = observations()
    model = make_model(obs, actor_obs_normalization=True, critic_obs_normalization=True)
    model.update_normalization(obs)
    model.eval()
    expected = model.act_inference(obs)
    changed = obs.clone()
    changed["critic"] = torch.randn_like(changed["critic"]) * 5
    with torch.no_grad():
        for head in model.critic.film_heads:
            head.weight.normal_()
            head.bias.normal_()
    torch.testing.assert_close(model.act_inference(changed), expected, rtol=0, atol=0)
    assert ActorCriticFiLMCritic.act_inference is ActorCritic.act_inference
    assert ActorCriticFiLMCritic._build_actor is ActorCritic._build_actor
    assert ActorCriticFiLMCritic.export_as_onnx is ActorCritic.export_as_onnx


@pytest.mark.parametrize(
    "group,shape",
    [("policy", (4, 859)), ("policy", (4, 10, 86)), ("critic", (4, 1491)), ("critic", (4, 10, 149))],
)
def test_constructor_rejects_incompatible_flat_schema(group, shape):
    obs = observations()
    obs[group] = torch.zeros(shape)
    with pytest.raises(ValueError, match="group must have shape"):
        make_model(obs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"film_history_length": 0},
        {"film_history_length": True},
        {"film_morphology_start": -1},
        {"film_morphology_start": 147},
        {"film_morphology_start": 80},
        {"film_morphology_dim": 4},
        {"film_critic_frame_dim": 150},
        {"film_context_hidden_dims": []},
        {"film_context_hidden_dims": [0]},
        {"critic_hidden_dims": []},
    ],
)
def test_constructor_rejects_invalid_film_contract(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)


def test_runtime_shape_errors_fail_closed():
    model = make_model()
    bad = observations()
    bad["critic"] = torch.zeros(4, 1489)
    with pytest.raises(ValueError, match="requires"):
        model.evaluate(bad)
    with pytest.raises(ValueError, match="context requires shape"):
        model.critic(torch.zeros(4, 1490), torch.zeros(4, 5))
    with pytest.raises(ValueError, match="context requires shape"):
        model.critic(torch.zeros(4, 1490), torch.zeros(1, 50))
    with pytest.raises(ValueError, match="one flattened observation group"):
        ActorCriticFiLMCritic(observations(), {"policy": ["policy"], "critic": ["critic", "policy"]}, 24)
    with pytest.raises(ValueError, match="24 revolute actions"):
        ActorCriticFiLMCritic(observations(), GROUPS, 25)
    with pytest.raises(TypeError, match="Unknown FiLM-C options"):
        make_model(film_morphlogy_start=81)


def test_checkpoint_roundtrip_preserves_trained_critic_actor_and_normalizers():
    obs = observations()
    config = dict(actor_obs_normalization=True, critic_obs_normalization=True)
    model = make_model(obs, **config)
    model.update_normalization(obs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model.evaluate(obs).square().mean() + model.act_inference(obs).square().mean()).backward()
        optimizer.step()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = make_model(obs, **config)
    assert restored.load_state_dict(torch.load(stream, weights_only=True)) is True
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored.evaluate(obs), model.evaluate(obs), rtol=0, atol=0)
    torch.testing.assert_close(restored.act_inference(obs), model.act_inference(obs), rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("normalized", [False, True])
def test_actor_jit_and_onnx_exports_match_inference(tmp_path, normalized):
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    obs = observations(batch=1)
    model = make_model(obs, actor_obs_normalization=normalized).eval()
    if normalized:
        model.train()
        model.update_normalization(observations())
        model.eval()
    # The standard actor-only export API expects already normalized tensors.
    export_input = model.actor_obs_normalizer(obs["policy"])
    expected = model.act_inference(obs).detach().numpy()
    model.export_as_onnx(export_input, str(tmp_path))
    exported = onnx.load(str(tmp_path / "actor.onnx"))
    onnx.checker.check_model(exported)
    session = onnxruntime.InferenceSession(str(tmp_path / "actor.onnx"), providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1
    assert session.get_inputs()[0].shape == [1, 860]
    actual = session.run(None, {"input": export_input.detach().numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    # A deployment wrapper can include the unchanged actor normalizer, as the
    # existing Isaac Lab exporters do. No critic/context input is needed.
    actor_with_normalizer = nn.Sequential(copy.deepcopy(model.actor_obs_normalizer), copy.deepcopy(model.actor))
    scripted = torch.jit.script(actor_with_normalizer)
    scripted.save(str(tmp_path / "actor.pt"))
    loaded = torch.jit.load(str(tmp_path / "actor.pt"))
    torch.testing.assert_close(loaded(obs["policy"]), model.act_inference(obs), rtol=0, atol=0)


def test_registration_and_actual_parameter_counts():
    assert resolve_callable("rsl_rl.modules:ActorCriticFiLMCritic") is ActorCriticFiLMCritic
    assert resolve_callable("rsl_rl.modules.actor_critic_film:ActorCriticFiLMCritic") is ActorCriticFiLMCritic
    model = make_model(actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128))
    assert model.parameter_counts() == {
        "actor": 608152,
        "critic_backbone": 927745,
        "critic_film": 119744,
        "critic_total": 1047489,
        "other": 24,
        "total": 1655665,
    }
    assert sum(parameter.numel() for parameter in model.parameters()) == 1655665
