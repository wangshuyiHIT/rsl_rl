"""CPU checkpoint round trips preserve the original frozen AMP teacher."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest import mock

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import AMPRunner


def runner():
    obs = TensorDict({"policy": torch.zeros(2, 4), "critic": torch.zeros(2, 5)}, batch_size=[2])
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 2,
                         actor_hidden_dims=[4], critic_hidden_dims=[4],
                         actor_obs_normalization=True, critic_obs_normalization=True)
    discriminator = torch.nn.Linear(3, 1)
    discriminator.disc_obs_normalizer = torch.nn.Identity()
    result = object.__new__(AMPRunner)
    result.alg = NS(policy=policy, amp_discriminator=discriminator,
                    optimizer=torch.optim.Adam(policy.parameters()),
                    disc_optimizer=torch.optim.Adam(discriminator.parameters()))
    result.alg_cfg = {"rnd_cfg": None}
    result.logger = NS(save_model=mock.Mock())
    result.current_learning_iteration = 17
    return result


class AnchorCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "checkpoint.pt"
        self.source = runner()
        reference = copy.deepcopy(self.source.alg.policy)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(3.)
            reference.actor_obs_normalizer._mean.fill_(7.)
        reference.eval().requires_grad_(False)
        self.source.alg.reference_policy = reference
        self.source.alg.reference_policy_anchor_coef = .05
        self.source.alg.reference_policy_anchor_radius = .35
        self.source.alg.reference_policy_anchor_floor = .05
        self.source.alg.reference_policy_preserve_morphology = True
        self.source.alg.reference_policy_anchor_exclude_lateral_threshold = .1

    def test_resume_restores_original_teacher_flags_normalizer_without_optimizer_ownership(self):
        self.source.save(str(self.path))
        destination = runner()
        destination.load(str(self.path), map_location="cpu")
        for key, value in self.source.alg.reference_policy.state_dict().items():
            torch.testing.assert_close(destination.alg.reference_policy.state_dict()[key], value, atol=0, rtol=0)
        self.assertFalse(torch.equal(destination.alg.reference_policy.actor[0].weight,
                                     destination.alg.policy.actor[0].weight))
        for attr in ("reference_policy_anchor_coef", "reference_policy_anchor_radius",
                     "reference_policy_anchor_floor", "reference_policy_preserve_morphology",
                     "reference_policy_anchor_exclude_lateral_threshold"):
            self.assertEqual(getattr(destination.alg, attr), getattr(self.source.alg, attr))
        self.assertFalse(destination.alg.reference_policy.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in destination.alg.reference_policy.parameters()))
        optimized = {id(p) for optimizer in (destination.alg.optimizer, destination.alg.disc_optimizer)
                     for group in optimizer.param_groups for p in group["params"]}
        self.assertFalse(optimized & {id(p) for p in destination.alg.reference_policy.parameters()})
        self.assertEqual(destination.current_learning_iteration, 17)

    def test_legacy_checkpoint_without_anchor_does_not_create_or_reuse_teacher(self):
        self.source.alg.reference_policy = None
        self.source.save(str(self.path))
        data = torch.load(self.path, weights_only=True)
        self.assertNotIn("reference_policy_state_dict", data)
        destination = runner()
        destination.alg.reference_policy = copy.deepcopy(destination.alg.policy)
        destination.alg.reference_policy_anchor_coef = .9
        destination.alg.reference_policy_preserve_morphology = True
        destination.load(str(self.path), map_location="cpu")
        self.assertIsNone(destination.alg.reference_policy)
        self.assertEqual(destination.alg.reference_policy_anchor_coef, 0.)
        self.assertFalse(destination.alg.reference_policy_preserve_morphology)
        self.assertIsNone(destination.alg.reference_policy_anchor_exclude_lateral_threshold)

    def test_partial_or_invalid_anchor_configuration_is_rejected(self):
        self.source.save(str(self.path))
        original = torch.load(self.path, weights_only=True)
        for case in ("missing_weights", "missing_config", "nan_lateral", "negative_radius", "wrong_shape"):
            with self.subTest(case=case):
                data = copy.deepcopy(original)
                if case == "missing_weights":
                    del data["reference_policy_state_dict"]
                elif case == "missing_config":
                    del data["reference_policy_config"]
                elif case == "nan_lateral":
                    data["reference_policy_config"]["exclude_lateral_threshold"] = float("nan")
                elif case == "negative_radius":
                    data["reference_policy_config"]["radius"] = -1
                else:
                    data["reference_policy_state_dict"]["actor.0.weight"] = torch.zeros(1, 1)
                torch.save(data, self.path)
                with self.assertRaises((ValueError, RuntimeError)):
                    runner().load(str(self.path), map_location="cpu")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
