from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections.abc import Mapping
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent, AMPDiscriminator
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import resolve_callable
from rsl_rl.algorithms import PPO
from rsl_rl.modules.amp import LossType


class PPOAMP(PPO):

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer, 
        disc_demo_obs_buffer: CircularBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy,
            storage,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            normalize_advantage_per_mini_batch,
            device,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )
        
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for PPOAMP algorithm.")
        
        if self.amp_cfg["loss_type"] == "GAN":
            self.loss_type = LossType.GAN
        elif self.amp_cfg["loss_type"] == "LSGAN":
            self.loss_type = LossType.LSGAN
        elif self.amp_cfg["loss_type"] == "WGAN":
            self.loss_type = LossType.WGAN
        else:
            raise ValueError(f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'GAN', 'LSGAN', or 'WGAN'")
        
        self.amp_discriminator: AMPDiscriminator = AMPDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.policy.obs_groups,
            loss_type=self.loss_type,
            device=device,
            **self.amp_cfg.get("amp_discriminator", {})
        ).to(self.device)
        
        # optimizer for policy and discriminator
        params = [
            {
                "name": "disc_trunk", 
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],  # L2 regularization for the discriminator trunk
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],  # L2 regularization for the discriminator linear layer
            }
        ]
        # use a separate optimizer for the AMP discriminator
        self.disc_optimizer = optim.Adam(
            params,
            lr=self.amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)
        if not math.isfinite(self.disc_max_grad_norm) or self.disc_max_grad_norm <= 0:
            raise ValueError("disc_max_grad_norm must be finite and positive")
        
        # Storage for AMP discriminator observations
        self.disc_obs_buffer: CircularBuffer = disc_obs_buffer
        self.disc_demo_obs_buffer: CircularBuffer = disc_demo_obs_buffer
        self.lateral_reference_policy = None
        self.lateral_rehearsal = None

    def _lateral_rehearsal_loss(self) -> torch.Tensor | None:
        rehearsal = getattr(self, "lateral_rehearsal", None)
        return None if rehearsal is None else rehearsal.loss(self.policy)

    def _reference_policy_anchor_loss(
        self,
        obs: TensorDict,
        student_mean: torch.Tensor,
    ) -> torch.Tensor | None:
        """Keep AMP-2 gait near nominal while allowing morphology adaptation.

        AMP-3's actor observation is ten frame-major 86-D frames, with the
        five morphology features at the end of every frame.  The frozen AMP-2
        teacher sees those slots as nominal zero by default; a morphology-aware
        Flat teacher opts in to preserving them. A Gaussian morphology
        weight makes the constraint strongest at nominal geometry and weaker
        toward compact/extended robots, where PPO needs freedom to compensate
        for changed dynamics. Canonical Flat-to-AMP tuning may additionally
        exclude lateral commands using an explicit threshold; older tasks
        retain all command families when that option is absent.
        """

        reference = getattr(self, "reference_policy", None)
        coefficient = float(
            getattr(self, "reference_policy_anchor_coef", 0.0)
        )
        if reference is None or coefficient <= 0.0:
            return None

        actor_obs = self.policy.get_actor_obs(obs)
        if actor_obs.shape[-1] % 10 != 0:
            raise ValueError(
                "AMP-3 reference anchor requires ten frame-major observations, "
                f"got width {actor_obs.shape[-1]}"
            )
        frame_width = actor_obs.shape[-1] // 10
        if frame_width < 5:
            raise ValueError(
                f"AMP-3 reference anchor frame is too small: {frame_width}"
            )
        morphology_indices = torch.cat(
            [
                torch.arange(
                    frame * frame_width + frame_width - 5,
                    frame * frame_width + frame_width,
                    device=actor_obs.device,
                )
                for frame in range(10)
            ]
        )

        teacher_obs = actor_obs.detach().clone()
        if not getattr(self, "reference_policy_preserve_morphology", False):
            teacher_obs[..., morphology_indices] = 0.0
        with torch.no_grad():
            normalized_teacher_obs = reference.actor_obs_normalizer(
                teacher_obs
            )
            teacher_mean = reference.actor(normalized_teacher_obs)

        # Morphology is constant across the history, so the latest five slots
        # are sufficient for proximity weighting.
        morphology = actor_obs[..., -5:].detach()
        radius = float(
            getattr(self, "reference_policy_anchor_radius", 0.35)
        )
        floor = float(
            getattr(self, "reference_policy_anchor_floor", 0.05)
        )
        if radius <= 0.0 or not 0.0 <= floor <= 1.0:
            raise ValueError(
                "reference anchor requires radius > 0 and floor in [0, 1]"
            )
        proximity = torch.exp(
            -torch.mean(torch.square(morphology), dim=-1)
            / (2.0 * radius * radius)
        )
        weight = floor + (1.0 - floor) * proximity
        # Canonical Flat-to-AMP tuning may explicitly release a command family
        # the frozen teacher cannot execute.  Default None leaves historical
        # AMP anchors unchanged. Use the latest RAW frame, before the teacher's
        # normalizer; older frames can contain a preceding command.
        lateral_threshold = getattr(
            self, "reference_policy_anchor_exclude_lateral_threshold", None
        )
        if lateral_threshold is not None:
            lateral_threshold = float(lateral_threshold)
            if not math.isfinite(lateral_threshold) or lateral_threshold < 0.0:
                raise ValueError("reference anchor lateral threshold must be finite and non-negative")
            if frame_width < 9:
                raise ValueError("reference anchor lateral exclusion requires a 3-D command at frame slots 6:9")
            latest_command_y = actor_obs[..., -frame_width + 7].detach()
            weight = weight * (latest_command_y.abs() <= lateral_threshold)
        action_error = torch.mean(
            torch.square(student_mean - teacher_mean), dim=-1
        )
        return coefficient * torch.sum(weight * action_error) / torch.clamp(
            torch.sum(weight), min=1.0
        )

    def _lateral_reference_policy_anchor_loss(
        self, obs: TensorDict, student_mean: torch.Tensor,
    ) -> torch.Tensor | None:
        """Retain a frozen current-axis gait on pure lateral and optional backward commands.

        Use the primary anchor's morphology band weighting and action-mean MSE,
        preserving all actual morphology inputs. Only the latest raw command
        selects rows, including when a minibatch contains mirrored observations.
        """
        reference = getattr(self, "lateral_reference_policy", None)
        if reference is None:
            return None
        coefficient = float(self.lateral_reference_policy_anchor_coef)
        radius = float(self.lateral_reference_policy_anchor_radius)
        floor = float(self.lateral_reference_policy_anchor_floor)
        threshold = float(self.lateral_reference_policy_command_threshold)
        include_backward = getattr(self, "lateral_reference_policy_include_backward", False)
        if type(include_backward) is not bool:
            raise ValueError("Lateral reference include_backward must be a bool")
        if (not all(math.isfinite(value) for value in (coefficient, radius, floor, threshold))
                or coefficient <= 0 or radius <= 0 or not 0 <= floor <= 1 or threshold <= 0):
            raise ValueError("Invalid lateral reference policy anchor configuration")
        actor_obs = self.policy.get_actor_obs(obs)
        if actor_obs.ndim != 2 or actor_obs.shape[1] != 860 or student_mean.shape != (actor_obs.shape[0], 24):
            raise ValueError("Lateral reference policy requires raw 10x86 observations and 24 action means")
        command = actor_obs[:, -86 + 6:-86 + 9].detach()
        active = ((command[:, 1].abs() > threshold)
                  & (command[:, 0].abs() < threshold)
                  & (command[:, 2].abs() < threshold))
        if include_backward:
            active = active | ((command[:, 0] < -threshold)
                               & (command[:, 1].abs() < threshold)
                               & (command[:, 2].abs() < threshold))
        # Slicing before the loss also guarantees zero gradient on unselected
        # rows; an inactive batch still returns a differentiable exact zero.
        if not active.any():
            return student_mean[active].sum() * 0.0
        teacher_obs = actor_obs[active].detach().clone()
        with torch.no_grad():
            teacher_mean = reference.actor(reference.actor_obs_normalizer(teacher_obs))
        morphology = teacher_obs[:, -5:]
        proximity = torch.exp(-torch.mean(torch.square(morphology), dim=-1) / (2.0 * radius * radius))
        weight = floor + (1.0 - floor) * proximity
        error = torch.mean(torch.square(student_mean[active] - teacher_mean), dim=-1)
        return coefficient * torch.sum(weight * error) / torch.clamp(torch.sum(weight), min=1.0)

    def _reward_time_amp_obs(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> TensorDict:
        """Validate the canonical full-batch snapshot without falling back to next-state inputs."""
        key = "amp_reward_observations"
        if "task_style_gate" not in obs:
            if key in extras:
                raise ValueError("AMP reward snapshot requires the task_style_gate observation group")
            return obs
        snapshot = extras.get(key)
        if not isinstance(snapshot, Mapping) or set(snapshot) != {"observations", "done_env_ids"}:
            raise ValueError("Canonical AMP requires a complete reward-time observation snapshot")
        count = rewards.numel()
        if obs.batch_size != torch.Size([count]) or dones.numel() != count:
            raise ValueError("AMP reward snapshot batch must match rewards, dones, and observations")
        if not torch.isfinite(dones).all() or not ((dones == 0) | (dones == 1)).all():
            raise ValueError("AMP reward snapshot dones must be finite binary values")
        groups = snapshot["observations"]
        required = {"disc", "disc_demo", "style_gate", "task_style_gate"}
        if not isinstance(groups, Mapping) or set(groups) != required:
            raise ValueError("AMP reward snapshot requires disc, disc_demo, style_gate, and task_style_gate")
        disc_shape = (count, self.amp_cfg["disc_obs_steps"], self.amp_cfg["disc_obs_dim"])
        for name in required:
            value = groups[name]
            shape = disc_shape if name in ("disc", "disc_demo") else (count, 1)
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"AMP reward snapshot {name} must be finite floating point with shape {shape}")
        done_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        ids = snapshot["done_env_ids"]
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long
                or not torch.equal(ids, done_ids.to(ids.device))):
            raise ValueError("AMP reward snapshot done_env_ids must match done environments exactly")
        terminal = extras.get("amp_terminal_observations")
        if ids.numel():
            if (not isinstance(terminal, Mapping) or set(terminal) != {"env_ids", "observations"}
                    or not isinstance(terminal["env_ids"], torch.Tensor)):
                raise ValueError("AMP terminal observations must be the reward snapshot's done subset")
            terminal_ids = terminal["env_ids"]
            if (terminal_ids.dtype != torch.long or terminal_ids.ndim != 1
                    or not torch.equal(terminal_ids, ids.to(terminal_ids.device))):
                raise ValueError("AMP terminal env_ids must equal reward snapshot done_env_ids")
            terminal_groups = terminal["observations"]
            if not isinstance(terminal_groups, Mapping) or set(terminal_groups) != required:
                raise ValueError("AMP terminal observations require all four reward snapshot groups")
            for name in required:
                value = terminal_groups[name]
                expected = groups[name][ids.to(groups[name].device)]
                if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                        or not torch.equal(value.to(expected.device), expected)):
                    raise ValueError(f"AMP terminal {name} must equal the reward snapshot's done subset")
        elif terminal is not None:
            raise ValueError("AMP terminal observations were supplied without done environments")
        return TensorDict(dict(groups), batch_size=[count])

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        reward_obs = self._reward_time_amp_obs(obs, rewards, dones, extras)
        disc_obs = self.amp_discriminator.get_disc_obs(reward_obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(reward_obs, flatten_history_dim=False)
        gate = reward_obs.get("style_gate")
        task_gate = reward_obs.get("task_style_gate")
        if "task_style_gate" in reward_obs:
            def strict_gate(value, name, count):
                if (value is None or value.shape != (count, 1)
                        or not torch.isfinite(value).all()
                        or (value < 0).any() or (value > 1).any()):
                    raise ValueError(f"AMP {name} must be finite in [0, 1] with shape ({count}, 1)")
                return value[:, 0].to(rewards.device).clone()

            gate = strict_gate(gate, "style_gate", rewards.numel())
            task_gate = strict_gate(task_gate, "task_style_gate", rewards.numel())
            if (gate > task_gate).any():
                raise ValueError("AMP requires style_gate <= task_style_gate")
        elif gate is None:
            gate = torch.ones_like(rewards)
        else:
            if gate.shape != (rewards.numel(), 1) or not torch.isfinite(gate).all():
                raise ValueError("AMP style_gate must be finite with shape (num_envs, 1)")
            gate = gate[:, 0].to(rewards.device).clamp(0.0, 1.0).clone()
        terminal = extras.get("amp_terminal_observations")
        done_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel():
            if terminal is None:
                raise ValueError("AMP terminal transitions require pre-reset discriminator observations")
            ids = terminal["env_ids"]
            if ids.ndim != 1 or ids.dtype != torch.long:
                raise ValueError("AMP terminal env_ids must be a 1D long tensor")
            ids = ids.to(disc_obs.device)
            if not torch.equal(torch.sort(ids).values, done_ids.to(ids.device)):
                raise ValueError("AMP terminal env_ids must match done environments exactly")
            terminal_obs = TensorDict(terminal["observations"], batch_size=[ids.numel()])
            terminal_disc = self.amp_discriminator.get_disc_obs(terminal_obs, flatten_history_dim=False)
            terminal_demo = self.amp_discriminator.get_disc_demo_obs(terminal_obs, flatten_history_dim=False)
            if (terminal_disc.shape != disc_obs[ids].shape
                    or terminal_demo.shape != disc_demo_obs[ids].shape):
                raise ValueError("AMP terminal discriminator shape mismatch")
            disc_obs = disc_obs.clone()
            disc_demo_obs = disc_demo_obs.clone()
            disc_obs[ids] = terminal_disc.to(disc_obs.device)
            disc_demo_obs[ids] = terminal_demo.to(disc_demo_obs.device)
            if "style_gate" in obs:
                terminal_gate = terminal_obs.get("style_gate")
                if task_gate is not None:
                    terminal_gate = strict_gate(terminal_gate, "terminal style_gate", ids.numel())
                    terminal_task_gate = strict_gate(terminal_obs.get("task_style_gate"),
                                                     "terminal task_style_gate", ids.numel())
                    if (terminal_gate > terminal_task_gate).any():
                        raise ValueError("AMP terminal requires style_gate <= task_style_gate")
                    gate[ids.to(gate.device)] = terminal_gate
                    task_gate[ids.to(task_gate.device)] = terminal_task_gate
                else:
                    if (terminal_gate is None or terminal_gate.shape != (ids.numel(), 1)
                            or not torch.isfinite(terminal_gate).all()):
                        raise ValueError("AMP terminal style_gate must be finite with shape (num_done, 1)")
                    gate[ids.to(gate.device)] = terminal_gate[:, 0].to(gate.device).clamp(0.0, 1.0)
        elif terminal is not None:
            raise ValueError("AMP terminal observations were supplied without done environments")
        # Compute the Style Reward. Progress gates only the style bonus.
        raw_style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc_obs, dt=self.amp_cfg["step_dt"])
        if task_gate is None:
            # Preserve the legacy arithmetic exactly when the new field is absent.
            mixed_rewards = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=raw_style_rewards)
            self.rewards_lerp = gate * mixed_rewards + (1.0 - gate) * rewards
        else:
            # Supported commands keep the same task coefficient even at zero
            # progress; slowing down must never restore a larger task coefficient.
            task_component = self.amp_discriminator.lerp_reward(
                task_reward=rewards, style_reward=torch.zeros_like(raw_style_rewards))
            style_component = self.amp_discriminator.lerp_reward(
                task_reward=torch.zeros_like(rewards), style_reward=raw_style_rewards)
            self.rewards_lerp = (task_gate * task_component + (1.0 - task_gate) * rewards
                                 + gate * style_component)
        self.style_rewards = gate * raw_style_rewards
        # Store the un-normalized disc obs and disc demo obs into buffers
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        # Call the parent class method with the new rewards
        super().process_env_step(obs, self.rewards_lerp, dones, extras)
        
    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # AMP discriminator loss and other info
        mean_disc_loss = 0
        mean_disc_grad_penalty = 0
        mean_disc_score = 0
        mean_disc_demo_score = 0
        mean_reference_anchor_loss = 0
        mean_lateral_reference_anchor_loss = 0
        mean_lateral_rehearsal_loss = 0
        lateral_rehearsal_enabled = getattr(self, "lateral_rehearsal", None) is not None
        lateral_reference_anchor_enabled = getattr(self, "lateral_reference_policy", None) is not None
        reference_anchor_enabled = (
            getattr(self, "reference_policy", None) is not None
            and float(getattr(self, "reference_policy_anchor_coef", 0.0))
            > 0.0
        )

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        # Iterate over batches
        for samples, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = samples
            
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            reference_anchor_loss = self._reference_policy_anchor_loss(
                obs_batch[:original_batch_size],
                mu_batch,
            )
            if reference_anchor_loss is not None:
                loss += reference_anchor_loss
            lateral_reference_anchor_loss = self._lateral_reference_policy_anchor_loss(
                obs_batch[:original_batch_size], mu_batch,
            )
            if lateral_reference_anchor_loss is not None:
                loss += lateral_reference_anchor_loss
            lateral_rehearsal_loss = self._lateral_rehearsal_loss()
            if lateral_rehearsal_loss is not None:
                loss += lateral_rehearsal_loss

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # AMP discriminator loss
            with torch.no_grad():
                if self.amp_cfg.get("normalize_demo_observations", False):
                    # Both classes and the gradient penalty use one fixed set of
                    # statistics for this minibatch, including demonstration-only
                    # variation that may be absent from an early policy rollout.
                    self.amp_discriminator.update_normalization(
                        torch.cat((disc_obs_batch, disc_demo_obs_batch), dim=0)
                    )
                disc_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch) # [mini_batch_size, disc_obs_steps, disc_obs_dim]
                disc_demo_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)
            
            mini_batch_size = disc_obs_batch_normed.shape[0]
            disc_score = self.amp_discriminator(disc_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            disc_demo_score = self.amp_discriminator(disc_demo_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            
            if self.loss_type == LossType.GAN:
                bce = torch.nn.BCEWithLogitsLoss()
                policy_loss = bce(
                    disc_score, torch.zeros_like(disc_score, device=self.device)
                )
                demo_loss = bce(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.LSGAN:
                policy_loss = torch.nn.MSELoss()(
                    disc_score, -1 * torch.ones_like(disc_score, device=self.device)
                )
                demo_loss = torch.nn.MSELoss()(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.WGAN:
                disc_loss = - torch.mean(disc_demo_score) + torch.mean(disc_score)
            else: 
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'")

            disc_grad_penalty = self.amp_discriminator.compute_grad_penalty(
                demo_data=disc_demo_obs_batch_normed.reshape(mini_batch_size, -1),
                scale=self.amp_cfg["grad_penalty_scale"]
            )
            disc_total_loss = disc_loss + disc_grad_penalty

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            # Compute the gradients for AMP discriminator
            self.disc_optimizer.zero_grad()
            disc_total_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # Apply the gradients for AMP discriminator
            nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.disc_max_grad_norm)
            self.disc_optimizer.step()
            # Preserve the historical policy-only, post-step update unless the
            # shared policy/demo statistics were already updated above.
            if not self.amp_cfg.get("normalize_demo_observations", False):
                self.amp_discriminator.update_normalization(disc_obs_batch)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # AMP discriminator loss and other info
            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += disc_grad_penalty.item()
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()
            if reference_anchor_loss is not None:
                mean_reference_anchor_loss += reference_anchor_loss.item()
            if lateral_reference_anchor_loss is not None:
                mean_lateral_reference_anchor_loss += lateral_reference_anchor_loss.item()
            if lateral_rehearsal_loss is not None:
                mean_lateral_rehearsal_loss += lateral_rehearsal_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_disc_grad_penalty /= num_updates
        mean_disc_score /= num_updates
        mean_disc_demo_score /= num_updates
        if reference_anchor_enabled:
            mean_reference_anchor_loss /= num_updates
        if lateral_reference_anchor_enabled:
            mean_lateral_reference_anchor_loss /= num_updates
        if lateral_rehearsal_enabled:
            mean_lateral_rehearsal_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["amp/disc_loss"] = mean_disc_loss
        loss_dict["amp/disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["amp/disc_score"] = mean_disc_score
        loss_dict["amp/disc_demo_score"] = mean_disc_demo_score
        if reference_anchor_enabled:
            loss_dict["amp/reference_policy_anchor"] = (
                mean_reference_anchor_loss
            )
        if lateral_reference_anchor_enabled:
            loss_dict["amp/lateral_reference_policy_anchor"] = mean_lateral_reference_anchor_loss
        if lateral_rehearsal_enabled:
            loss_dict["amp/lateral_rehearsal"] = mean_lateral_rehearsal_loss

        return loss_dict
