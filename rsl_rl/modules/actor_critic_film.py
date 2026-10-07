# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A critic-only FiLM ablation for the Scaler U5 observation contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.networks import MLP
from rsl_rl.utils import resolve_nn_activation

from .actor_critic import ActorCritic


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class _FiLMCritic(nn.Module):
    """Keep the full critic input and modulate hidden activations only.

    Context is the unnormalized, already encoded morphology token from every
    frame, in oldest-to-newest order. It is passed separately from the ordinary
    (optionally empirically normalized) full critic input. For each hidden layer,
    h = (1 + delta_gamma(context)) * activation(linear(h)) + beta(context).
    Zero-initialized heads make this exactly the ordinary critic initially.
    The value output layer is not modulated.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        frame_dim: int,
        history_length: int,
        morphology_start: int,
        morphology_dim: int,
        context_hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.frame_dim = frame_dim
        self.history_length = history_length
        self.morphology_start = morphology_start
        self.morphology_dim = morphology_dim
        self.context_dim = history_length * morphology_dim

        # Build the ordinary critic first, preserving its architecture and RNG
        # initialization under the same seed as ActorCritic.
        self.backbone = MLP(input_dim, 1, list(hidden_dims), activation)
        context_layers: list[nn.Module] = []
        context_width = self.context_dim
        for width in context_hidden_dims:
            context_layers.extend((nn.Linear(context_width, width), resolve_nn_activation(activation)))
            context_width = width
        self.context_encoder = nn.Sequential(*context_layers)
        self.film_heads = nn.ModuleList(nn.Linear(context_width, 2 * width) for width in hidden_dims)
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _check_observation_shape(self, observations: torch.Tensor) -> None:
        if observations.ndim != 2 or observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"FiLM-C critic requires [batch, {self.input_dim}] observations "
                f"({self.history_length} frames of {self.frame_dim}), got {tuple(observations.shape)}"
            )

    def extract_morphology_context(self, raw_observations: torch.Tensor) -> torch.Tensor:
        self._check_observation_shape(raw_observations)
        frames = raw_observations.reshape(-1, self.history_length, self.frame_dim)
        return frames[
            :, :, self.morphology_start : self.morphology_start + self.morphology_dim
        ].reshape(raw_observations.shape[0], self.context_dim)

    def forward(self, observations: torch.Tensor, morphology_context: torch.Tensor) -> torch.Tensor:
        self._check_observation_shape(observations)
        expected_context_shape = (observations.shape[0], self.context_dim)
        if morphology_context.ndim != 2 or tuple(morphology_context.shape) != expected_context_shape:
            raise ValueError(
                f"FiLM-C context requires shape {expected_context_shape}, got {tuple(morphology_context.shape)}"
            )
        encoded_context = self.context_encoder(morphology_context)
        hidden = observations
        for index, head in enumerate(self.film_heads):
            hidden = self.backbone[2 * index + 1](self.backbone[2 * index](hidden))
            delta_gamma, beta = head(encoded_context).chunk(2, dim=-1)
            hidden = (1.0 + delta_gamma) * hidden + beta
        return self.backbone[-1](hidden)


class ActorCriticFiLMCritic(ActorCritic):
    """Scaler FiLM-C: unchanged U5 actor and a full-information FiLM critic.

    The default contract is a single flattened 86 x 10 actor group, a single
    flattened 149 x 10 critic group, and 24 revolute actions. The critic retains
    all concatenated true-state inputs. Its extra conditioning context is the
    five features at offsets 81:86 in each frame before empirical normalization.
    Scaler BaseEnv flattens Isaac Lab CircularBuffer.buffer, whose frames are
    oldest first and newest last; that order is retained in the 50-D context.

    Actor construction, normalization, action distribution, inference, and actor
    export are inherited unchanged from ActorCritic. This is an additive
    architecture ablation inspired by FiLM conditioning, not a reproduction of
    MorFiC. Match the actor/critic widths and normalization flags to U5 in the
    training configuration; the extra parameter count is explicitly available
    through parameter_counts().
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        *,
        film_critic_frame_dim: int = 149,
        film_history_length: int = 10,
        film_morphology_start: int = 81,
        film_morphology_dim: int = 5,
        film_context_hidden_dims: tuple[int, ...] | list[int] = (64,),
        **kwargs: dict[str, Any],
    ) -> None:
        # Validate the flat-frame layout before ActorCritic builds any network.
        self.film_critic_frame_dim = _positive_int("film_critic_frame_dim", film_critic_frame_dim)
        self.film_history_length = _positive_int("film_history_length", film_history_length)
        self.film_morphology_dim = _positive_int("film_morphology_dim", film_morphology_dim)
        if (
            isinstance(film_morphology_start, bool)
            or not isinstance(film_morphology_start, int)
            or film_morphology_start < 0
            or film_morphology_start + film_morphology_dim > film_critic_frame_dim
        ):
            raise ValueError("The FiLM morphology slice must lie entirely within one critic frame")
        if (film_critic_frame_dim, film_morphology_start, film_morphology_dim) != (149, 81, 5):
            raise ValueError("Scaler FiLM-C requires 149-D critic frames with true morphology at 81:86")
        self.film_morphology_start = film_morphology_start
        self.film_context_hidden_dims = tuple(
            _positive_int("film_context_hidden_dims entry", width) for width in film_context_hidden_dims
        )
        if not self.film_context_hidden_dims:
            raise ValueError("film_context_hidden_dims must contain at least one hidden layer")
        if not critic_hidden_dims:
            raise ValueError("critic_hidden_dims must contain at least one hidden layer")
        for width in critic_hidden_dims:
            _positive_int("critic_hidden_dims entry", width)
        if num_actions != 24:
            raise ValueError(f"Scaler FiLM-C requires 24 revolute actions, got {num_actions}")
        for role, frame_dim in (("policy", 86), ("critic", film_critic_frame_dim)):
            groups = obs_groups.get(role, [])
            if len(groups) != 1:
                raise ValueError(f"FiLM-C requires one flattened observation group for {role}, got {groups}")
            group = obs[groups[0]]
            expected_dim = frame_dim * film_history_length
            if group.ndim != 2 or group.shape[-1] != expected_dim:
                raise ValueError(
                    f"FiLM-C {role} group must have shape [batch, {expected_dim}], got {tuple(group.shape)}"
                )
        if any(key.startswith("film_") for key in kwargs):
            raise TypeError(f"Unknown FiLM-C options: {[key for key in kwargs if key.startswith('film_')]}")
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )

    def _build_critic(
        self, num_critic_obs: int, critic_hidden_dims: tuple[int, ...] | list[int], activation: str
    ) -> nn.Module:
        return _FiLMCritic(
            num_critic_obs,
            critic_hidden_dims,
            activation,
            self.film_critic_frame_dim,
            self.film_history_length,
            self.film_morphology_start,
            self.film_morphology_dim,
            self.film_context_hidden_dims,
        )

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        raw_observations = self.get_critic_obs(obs)
        context = self.critic.extract_morphology_context(raw_observations)
        normalized_observations = self.critic_obs_normalizer(raw_observations)
        return self.critic(normalized_observations, context)

    def parameter_counts(self) -> dict[str, int]:
        """Report actual trainable parameter counts, not observation dimensions."""
        count = lambda module: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        actor = count(self.actor)
        backbone = count(self.critic.backbone)
        film = count(self.critic.context_encoder) + count(self.critic.film_heads)
        total = count(self)
        return {
            "actor": actor,
            "critic_backbone": backbone,
            "critic_film": film,
            "critic_total": backbone + film,
            "other": total - actor - backbone - film,
            "total": total,
        }
