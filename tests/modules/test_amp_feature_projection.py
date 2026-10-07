"""CPU contracts for opt-in AMP feature projection and physical std floors."""
import copy
from types import SimpleNamespace as NS

import pytest
import torch
from rsl_rl.modules.amp import AMPDiscriminator
from rsl_rl.runners import AMPRunner
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location('_amp_checkpoint_fixture', Path(__file__).with_name('test_amp_anchor_checkpoint.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
runner = _fixture.runner


def discriminator(**kwargs):
    return AMPDiscriminator(75, 3, {'discriminator': ['policy'], 'discriminator_demonstration': ['demo']},
                            hidden_dims=[8], **kwargs)


def test_legacy_normalization_reward_state_keys_and_statistics_are_identical():
    d = discriminator()
    x = torch.randn(4, 3, 75)
    d.update_normalization(x)
    expected = d.disc_obs_normalizer(x.reshape(-1, 75)).reshape_as(x)
    torch.testing.assert_close(d.normalize_disc_obs(x), expected, rtol=0, atol=0)
    score = d(expected.flatten(1)).squeeze(-1)
    reward, actual = d.predict_style_reward(x, .02)
    torch.testing.assert_close(actual, score, rtol=0, atol=0)
    torch.testing.assert_close(reward, .02 * (1 - .25 * (score-1).square()).clamp_min(0), rtol=0, atol=0)
    assert d.training
    projected = discriminator(excluded_feature_indices=[10, 34], normalization_std_floor=[.1]*75)
    assert set(projected.state_dict()) == set(d.state_dict())
    projected.load_state_dict(d.state_dict(), strict=True)


def test_both_domains_flattening_statistics_and_input_nonmutation():
    d = discriminator(excluded_feature_indices=[10, 34])
    x, y = torch.randn(4, 3, 75), torch.randn(4, 3, 75)
    old_x, old_y = x.clone(), y.clone()
    data = {'policy': x, 'demo': y}
    for getter, original in [(d.get_disc_obs, x), (d.get_disc_demo_obs, y)]:
        actual = getter(data)
        expected = original.clone()
        expected[:, :, [10, 34]] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(getter(data, True), expected.flatten(1), rtol=0, atol=0)
    d.update_normalization(x)
    assert d.disc_obs_normalizer.mean[[10, 34]].tolist() == [0., 0.]
    d.disc_obs_normalizer._mean[:, [10, 34]] = 99  # even old statistics cannot leak the excluded feature
    assert torch.count_nonzero(d.normalize_disc_obs(x)[:, :, [10, 34]]) == 0
    torch.testing.assert_close(x, old_x, rtol=0, atol=0)
    torch.testing.assert_close(y, old_y, rtol=0, atol=0)


def test_constant_channels_physical_floor_epsilon_and_eager_reward_share_path():
    floors = [.5]*3 + [.1]*24 + [1.]*24 + [.05]*18 + [1.]*6
    d = discriminator(excluded_feature_indices=[10, 34], normalization_std_floor=floors)
    d.update_normalization(torch.zeros(4, 3, 75))
    x = torch.ones(2, 3, 75)
    expected = x / (torch.tensor(floors) + d.disc_obs_normalizer.eps)
    expected[:, :, [10, 34]] = 0
    torch.testing.assert_close(d.normalize_disc_obs(x), expected, rtol=0, atol=0)
    _, scores = d.predict_style_reward(x, .02)
    torch.testing.assert_close(scores, d(expected.flatten(1)).squeeze(-1), rtol=0, atol=0)
    d.eval()
    d.predict_style_reward(x, .02)
    assert not d.training
    d.disc_obs_normalizer._std.fill_(2.)
    expected = x / 2.01
    expected[:, :, [10, 34]] = 0
    torch.testing.assert_close(d.normalize_disc_obs(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [
    {'excluded_feature_indices': [75]}, {'excluded_feature_indices': [True]},
    {'excluded_feature_indices': [10, 10]}, {'normalization_std_floor': [1.]},
    {'normalization_std_floor': [0.]*75}, {'normalization_std_floor': [float('nan')]*75},
])
def test_invalid_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        discriminator(**kwargs)


def prepared_runner(projected=False):
    r = runner()
    r.alg.amp_discriminator = discriminator(**({'excluded_feature_indices': [10, 34],
                                               'normalization_std_floor': [.1]*75} if projected else {}))
    r.alg.disc_optimizer = torch.optim.Adam(r.alg.amp_discriminator.parameters(), lr=.002)
    r.alg.amp_cfg = {'normalize_demo_observations': True}
    return r


def test_fresh_discriminator_restore_preserves_policy_optimizer_teacher_and_cursor(tmp_path):
    source = prepared_runner()
    source.alg.reference_policy = copy.deepcopy(source.alg.policy).eval().requires_grad_(False)
    source.alg.reference_policy_anchor_coef = .05
    source._last_completed_learning_iteration = 17
    source._next_learning_iteration = 18
    # Populate both optimizers with real CPU state.
    for module, optimizer in [(source.alg.policy, source.alg.optimizer),
                              (source.alg.amp_discriminator, source.alg.disc_optimizer)]:
        sum(p.sum() for p in module.parameters()).backward()
        optimizer.step()
    source.alg.amp_discriminator.update_normalization(torch.randn(4, 3, 75))
    path = tmp_path/'source.pt'
    source.save(str(path))
    dest = prepared_runner(True)
    original = copy.deepcopy(dest.alg.amp_discriminator.state_dict())
    with pytest.raises(ValueError, match='feature contract mismatch'):
        dest.load(str(path))
    dest.load(str(path), load_discriminator=False)
    for name, value in original.items():
        torch.testing.assert_close(dest.alg.amp_discriminator.state_dict()[name], value, rtol=0, atol=0)
    assert not dest.alg.disc_optimizer.state
    assert dest.alg.optimizer.state
    assert dest.current_learning_iteration == 18
    for name, value in source.alg.policy.state_dict().items():
        torch.testing.assert_close(dest.alg.policy.state_dict()[name], value, rtol=0, atol=0)
    for name, value in source.alg.reference_policy.state_dict().items():
        torch.testing.assert_close(dest.alg.reference_policy.state_dict()[name], value, rtol=0, atol=0)
    assert not dest.alg.reference_policy.training
    assert all(not p.requires_grad for p in dest.alg.reference_policy.parameters())


def test_same_contract_restores_and_new_contract_rejects_historical_absence(tmp_path):
    source = prepared_runner(True)
    path = tmp_path/'source.pt'
    source.save(str(path))
    dest = prepared_runner(True)
    dest.load(str(path))
    for name, value in source.alg.amp_discriminator.state_dict().items():
        torch.testing.assert_close(dest.alg.amp_discriminator.state_dict()[name], value, rtol=0, atol=0)
    data = torch.load(path, weights_only=False)
    data.pop('amp_discriminator_feature_contract')
    torch.save(data, path)
    with pytest.raises(ValueError, match='contract missing'):
        dest.load(str(path))
    dest.load(str(path), load_discriminator=False)
    # Unknown old normalization-demo convention preserves the old API only
    # when no new projection or physical-floor semantics are requested.
    prepared_runner().load(str(path))


def test_saved_normalization_demo_contract_change_rejected(tmp_path):
    source = prepared_runner()
    path = tmp_path/'source.pt'
    source.save(str(path))
    dest = prepared_runner()
    dest.alg.amp_cfg['normalize_demo_observations'] = False
    with pytest.raises(ValueError, match='feature contract mismatch'):
        dest.load(str(path))
