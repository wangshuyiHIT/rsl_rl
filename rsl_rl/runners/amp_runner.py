from __future__ import annotations

import os
import copy
import math
import json
import statistics
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import PPO, PPOAMP
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticCNN,
    EncoderActorCritic,
    EncoderMoEActorCritic,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
    resolve_amp_config,
)
from rsl_rl.modules.moe import collect_actor_moe_gate_log
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.utils.logger import Logger
from rsl_rl.utils.amp_logger import LoggerAMP
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules.policy_rehearsal import LateralPolicyRehearsal


class AMPRunner(OnPolicyRunner):
    
    alg: PPOAMP

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)
        # Cursor is separate from the last completed checkpoint label.
        self._next_learning_iteration = self.current_learning_iteration
        self._last_completed_learning_iteration = None

    def _construct_logger(self, log_dir: str | None) -> Logger:
        # OnPolicyRunner builds its logger through this hook, so returning
        # LoggerAMP here avoids a second W&B initialization after super().
        return LoggerAMP(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            max_episode_length_s=(self.env.max_episode_length*self.env.unwrapped.step_dt),
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = getattr(self, "_next_learning_iteration", self.current_learning_iteration)
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Extract AMP rewards (only for logging)
                    style_rewards = self.alg.style_rewards
                    total_rewards = self.alg.rewards_lerp
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards, style_rewards, total_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()
            loss_dict.update(collect_actor_moe_gate_log(self.alg.policy, obs))

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self._last_completed_learning_iteration = it
            self._next_learning_iteration = it + 1
            # train.py installs curriculum callbacks for the next rollout.
            # Invoke exactly once after a successfully completed optimizer update.
            callback = getattr(self, "iteration_callback", None)
            if callback is not None:
                callback(self, it)

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )
            
            # Save model (rank 0 only in distributed mode)
            if it % self.cfg["save_interval"] == 0 and not self.logger.disable_logs:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _discriminator_feature_contract(self) -> dict:
        discriminator = self.alg.amp_discriminator
        features = (discriminator.feature_contract() if hasattr(discriminator, "feature_contract")
                    else {"excluded_feature_indices": [], "normalization_std_floor": None})
        return {**features, "normalize_demo_observations": bool(
            getattr(self.alg, "amp_cfg", {}).get("normalize_demo_observations", False))}

    def _prepare_lateral_reference_policy(self, policy, config: dict, source_metadata: dict):
        """Validate and independently freeze the optional current-axis raw teacher."""
        expected_keys = {"schema_version", "coefficient", "radius", "floor", "command_threshold", "preserve_morphology"}
        if (not isinstance(config, dict) or type(config.get("schema_version")) is not int
                or config["schema_version"] not in (1, 2)):
            raise ValueError("Invalid lateral reference policy configuration schema")
        if config["schema_version"] == 2:
            expected_keys.add("include_backward")
        if (set(config) != expected_keys or config["preserve_morphology"] is not True
                or (config["schema_version"] == 2 and type(config["include_backward"]) is not bool)):
            raise ValueError("Invalid lateral reference policy configuration")
        include_backward = config["include_backward"] if config["schema_version"] == 2 else False
        try:
            coefficient, radius, floor, threshold = (
                float(config[name]) for name in ("coefficient", "radius", "floor", "command_threshold"))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid lateral reference policy configuration") from error
        if (not all(math.isfinite(value) for value in (coefficient, radius, floor, threshold))
                or coefficient <= 0 or radius <= 0 or not 0 <= floor <= 1 or threshold <= 0):
            raise ValueError("Invalid lateral reference policy configuration")
        if not isinstance(source_metadata, dict):
            raise ValueError("Lateral reference source_metadata must be a JSON dictionary")
        try:
            source = json.loads(json.dumps(source_metadata, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Lateral reference source_metadata must be finite JSON data") from error
        student = self.alg.policy
        if type(policy) is not ActorCritic or type(student) is not ActorCritic:
            raise ValueError("Lateral reference policy requires an unadapted ActorCritic")
        for model in (student, policy):
            actor_layers = [layer for layer in model.actor.modules() if isinstance(layer, torch.nn.Linear)]
            critic_layers = [layer for layer in model.critic.modules() if isinstance(layer, torch.nn.Linear)]
            if (model.actor_obs_normalization or model.critic_obs_normalization or model.state_dependent_std
                    or not isinstance(model.actor_obs_normalizer, torch.nn.Identity)
                    or not isinstance(model.critic_obs_normalizer, torch.nn.Identity)
                    or actor_layers[0].in_features != 860 or actor_layers[-1].out_features != 24
                    or critic_layers[0].in_features != 1490):
                raise ValueError("Lateral reference policy requires raw 860/1490 observations and 24 actions")
        state, student_state = policy.state_dict(), student.state_dict()
        if (state.keys() != student_state.keys()
                or any(value.shape != student_state[name].shape or value.dtype != student_state[name].dtype
                       or not torch.isfinite(value).all()
                       for name, value in state.items())
                or policy.obs_groups["policy"] != student.obs_groups["policy"]
                or policy.obs_groups["critic"] != student.obs_groups["critic"]):
            raise ValueError("Lateral reference policy must match the student's complete state contract")
        reference = copy.deepcopy(policy).to(next(student.parameters()).device)
        reference.eval().requires_grad_(False)
        return reference, {
            "schema_version": 2, "coefficient": coefficient, "radius": radius, "floor": floor,
            "command_threshold": threshold, "preserve_morphology": True, "include_backward": include_backward,
        }, source

    def _assign_lateral_reference_policy(self, prepared) -> None:
        if prepared is None:
            self.alg.lateral_reference_policy = None
            self.alg.lateral_reference_policy_anchor_coef = 0.0
            self.alg.lateral_reference_policy_anchor_radius = .10
            self.alg.lateral_reference_policy_anchor_floor = .02
            self.alg.lateral_reference_policy_command_threshold = .1
            self.alg.lateral_reference_policy_include_backward = False
            self.alg.lateral_reference_policy_source_metadata = None
            return
        reference, config, source = prepared
        self.alg.lateral_reference_policy = reference
        self.alg.lateral_reference_policy_anchor_coef = config["coefficient"]
        self.alg.lateral_reference_policy_anchor_radius = config["radius"]
        self.alg.lateral_reference_policy_anchor_floor = config["floor"]
        self.alg.lateral_reference_policy_command_threshold = config["command_threshold"]
        self.alg.lateral_reference_policy_include_backward = config["include_backward"]
        self.alg.lateral_reference_policy_source_metadata = source

    def set_lateral_reference_policy(
        self, policy: ActorCritic | None, coefficient: float = .5, radius: float = .10,
        floor: float = .02, command_threshold: float = .1, *, source_metadata: dict | None = None,
        include_backward: bool = False,
    ) -> None:
        """Install a separate frozen raw teacher; None disables only this anchor.

        Source metadata is archived with the complete teacher state. Callers
        bind the source checkpoint/robot identity before providing this model.
        No optimizer takes ownership of the teacher and the primary is intact.
        Backward commands are opt-in; old version-1 checkpoints remain side-only.
        """
        if type(include_backward) is not bool:
            raise ValueError("Lateral reference include_backward must be a bool")
        if policy is None:
            if getattr(self.alg, "lateral_rehearsal", None) is not None:
                raise ValueError("Clear lateral rehearsal before disabling its frozen teacher")
            self._assign_lateral_reference_policy(None)
            return
        config = {"schema_version": 2, "coefficient": coefficient, "radius": radius, "floor": floor,
                  "command_threshold": command_threshold, "preserve_morphology": True,
                  "include_backward": include_backward}
        prepared = self._prepare_lateral_reference_policy(
            policy, config, {} if source_metadata is None else source_metadata)
        rehearsal = getattr(self.alg, "lateral_rehearsal", None)
        if rehearsal is not None:
            rehearsal.validate_binding(self.alg.policy, prepared[0], prepared[1]["include_backward"], prepared[2])
        self._assign_lateral_reference_policy(prepared)

    def set_lateral_rehearsal(
        self, observations_by_family, *, coefficient: float = .25, batch_per_family: int = 64,
        source_metadata: dict | None = None,
    ) -> None:
        """Install balanced left/right/backward fixed-target action rehearsal.

        Raw float32 [N,860] pools are copied, and the already-installed frozen
        backward-enabled lateral teacher generates detached 24-action targets.
        None explicitly disables this training-only loss; inference is unchanged.
        """
        if observations_by_family is None:
            self.alg.lateral_rehearsal = None
            return
        teacher = getattr(self.alg, "lateral_reference_policy", None)
        if teacher is not None:
            teacher_ids = {id(p) for p in teacher.parameters()}
            for optimizer in (self.alg.optimizer, self.alg.disc_optimizer):
                if teacher_ids.intersection(id(p) for group in optimizer.param_groups for p in group["params"]):
                    raise ValueError("Rehearsal teacher must not belong to any optimizer")
        prepared = LateralPolicyRehearsal.prepare(
            self.alg.policy, teacher, getattr(self.alg, "lateral_reference_policy_include_backward", False),
            observations_by_family, coefficient=coefficient, batch_per_family=batch_per_family,
            source_metadata=source_metadata,
            teacher_source_metadata=getattr(self.alg, "lateral_reference_policy_source_metadata", None),
        )
        self.alg.lateral_rehearsal = prepared

    def save(self, path: str, infos: dict | None = None) -> None:
        # Keep the historical `iter` label, but archive an unambiguous next
        # label for continuation. A save immediately after load preserves both.
        completed = getattr(self, "_last_completed_learning_iteration", None)
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration if completed is None else completed,
            "next_learning_iteration": getattr(self, "_next_learning_iteration", self.current_learning_iteration),
            "infos": infos,
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Save AMP model
        saved_dict["amp_discriminator_state_dict"] = self.alg.amp_discriminator.state_dict()
        saved_dict["amp_discriminator_normalizer_state_dict"] = self.alg.amp_discriminator.disc_obs_normalizer.state_dict()
        saved_dict["amp_discriminator_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        saved_dict["amp_discriminator_feature_contract"] = self._discriminator_feature_contract()
        reference = getattr(self.alg, "reference_policy", None)
        if reference is not None and getattr(self.alg, "reference_policy_anchor_coef", 0.0) > 0.0:
            saved_dict["reference_policy_state_dict"] = reference.state_dict()
            saved_dict["reference_policy_config"] = {
                "schema_version": 1,
                "coefficient": float(self.alg.reference_policy_anchor_coef),
                "radius": float(getattr(self.alg, "reference_policy_anchor_radius", 0.35)),
                "floor": float(getattr(self.alg, "reference_policy_anchor_floor", 0.05)),
                "preserve_morphology": bool(getattr(self.alg, "reference_policy_preserve_morphology", False)),
                "exclude_lateral_threshold": getattr(self.alg, "reference_policy_anchor_exclude_lateral_threshold", None),
            }
        lateral_reference = getattr(self.alg, "lateral_reference_policy", None)
        if lateral_reference is not None:
            saved_dict["lateral_reference_policy_state_dict"] = lateral_reference.state_dict()
            saved_dict["lateral_reference_policy_config"] = {
                "schema_version": 2,
                "coefficient": float(self.alg.lateral_reference_policy_anchor_coef),
                "radius": float(self.alg.lateral_reference_policy_anchor_radius),
                "floor": float(self.alg.lateral_reference_policy_anchor_floor),
                "command_threshold": float(self.alg.lateral_reference_policy_command_threshold),
                "preserve_morphology": True,
                "include_backward": self.alg.lateral_reference_policy_include_backward,
            }
            saved_dict["lateral_reference_policy_source_metadata"] = copy.deepcopy(
                self.alg.lateral_reference_policy_source_metadata)
        rehearsal = getattr(self.alg, "lateral_rehearsal", None)
        if rehearsal is not None:
            rehearsal.validate_binding(self.alg.policy, lateral_reference,
                                       self.alg.lateral_reference_policy_include_backward,
                                       self.alg.lateral_reference_policy_source_metadata)
            saved_dict["lateral_rehearsal_state"] = rehearsal.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None,
             load_discriminator: bool = True) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if type(load_discriminator) is not bool:
            raise ValueError("load_discriminator must be an explicit bool")
        lateral_keys = {"lateral_reference_policy_state_dict", "lateral_reference_policy_config",
                        "lateral_reference_policy_source_metadata"}
        prepared_lateral = None
        if lateral_keys.intersection(loaded_dict):
            if not lateral_keys.issubset(loaded_dict):
                raise ValueError("AMP checkpoint requires lateral teacher weights, configuration, and source metadata")
            lateral_reference = copy.deepcopy(self.alg.policy)
            torch.nn.Module.load_state_dict(lateral_reference, loaded_dict["lateral_reference_policy_state_dict"], strict=True)
            prepared_lateral = self._prepare_lateral_reference_policy(
                lateral_reference, loaded_dict["lateral_reference_policy_config"],
                loaded_dict["lateral_reference_policy_source_metadata"])
        prepared_rehearsal = None
        if "lateral_rehearsal_state" in loaded_dict:
            if prepared_lateral is None:
                raise ValueError("Rehearsal checkpoint requires its frozen lateral teacher")
            prepared_rehearsal = LateralPolicyRehearsal.from_state_dict(
                loaded_dict["lateral_rehearsal_state"], student=self.alg.policy,
                teacher=prepared_lateral[0], include_backward=prepared_lateral[1]["include_backward"],
                teacher_source_metadata=prepared_lateral[2],
            )
        if load_discriminator:
            # Historical checkpoints had unprojected features and no physical
            # std floor. Keep their legacy path, but reject projected restores
            # before changing any policy, teacher or optimizer state.
            saved_contract = loaded_dict.get("amp_discriminator_feature_contract")
            current_contract = self._discriminator_feature_contract()
            if saved_contract is None:
                if (current_contract["excluded_feature_indices"]
                        or current_contract["normalization_std_floor"] is not None):
                    raise ValueError("AMP discriminator feature contract missing; use load_discriminator=False for new semantics")
            elif saved_contract != current_contract:
                raise ValueError("AMP discriminator feature contract mismatch; use load_discriminator=False for new semantics")
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used.  Checkpoints written before RND was
        # enabled carry no RND weights; keep the fresh module so old
        # models stay loadable (e.g. for play/eval comparisons).
        if self.alg_cfg["rnd_cfg"]:
            if "rnd_state_dict" in loaded_dict:
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            else:
                warnings.warn(
                    f"checkpoint {path} has no 'rnd_state_dict'; keeping the freshly "
                    "initialized RND module."
                )
        # Load AMP model
        if load_discriminator:
            self.alg.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])
            self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(loaded_dict["amp_discriminator_normalizer_state_dict"])
        reference_state = loaded_dict.get("reference_policy_state_dict")
        reference_cfg = loaded_dict.get("reference_policy_config")
        if (reference_state is None) != (reference_cfg is None):
            raise ValueError("AMP checkpoint requires both reference policy weights and configuration")
        if reference_state is None:
            # Older checkpoints did not archive a teacher. Never silently use
            # the resumed student (or a previously loaded teacher) as one.
            self.alg.reference_policy = None
            self.alg.reference_policy_anchor_coef = 0.0
            self.alg.reference_policy_preserve_morphology = False
            self.alg.reference_policy_anchor_exclude_lateral_threshold = None
        else:
            coefficient = float(reference_cfg["coefficient"])
            radius = float(reference_cfg["radius"])
            floor = float(reference_cfg["floor"])
            preserve = reference_cfg["preserve_morphology"]
            lateral = reference_cfg.get("exclude_lateral_threshold")
            if (reference_cfg.get("schema_version") != 1
                    or not all(math.isfinite(value) for value in (coefficient, radius, floor))
                    or coefficient <= 0 or radius <= 0 or not 0 <= floor <= 1
                    or not isinstance(preserve, bool)
                    or (lateral is not None and (not math.isfinite(float(lateral)) or float(lateral) < 0))):
                raise ValueError("Invalid AMP reference policy configuration in checkpoint")
            reference = copy.deepcopy(self.alg.policy)
            torch.nn.Module.load_state_dict(reference, reference_state, strict=True)
            reference.eval()
            reference.requires_grad_(False)
            self.alg.reference_policy = reference
            self.alg.reference_policy_anchor_coef = coefficient
            self.alg.reference_policy_anchor_radius = radius
            self.alg.reference_policy_anchor_floor = floor
            self.alg.reference_policy_preserve_morphology = preserve
            self.alg.reference_policy_anchor_exclude_lateral_threshold = None if lateral is None else float(lateral)
        # A historical checkpoint must never inherit a previously installed
        # lateral teacher. New checkpoints restore it independently of D reset.
        self._assign_lateral_reference_policy(prepared_lateral)
        # An older checkpoint must not inherit a previously installed dataset.
        self.alg.lateral_rehearsal = prepared_rehearsal
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.alg_cfg["rnd_cfg"] and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            # AMP discriminator optimizer
            if load_discriminator:
                self.alg.disc_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
        # Load current learning iteration
        if resumed_training:
            completed = loaded_dict["iter"]
            next_iteration = loaded_dict.get("next_learning_iteration", completed)
            if (type(completed) is not int or type(next_iteration) is not int
                    or completed < 0 or next_iteration < completed
                    or next_iteration > completed + 1):
                raise ValueError("Invalid AMP checkpoint iteration cursor")
            self.current_learning_iteration = next_iteration
            self._next_learning_iteration = next_iteration
            self._last_completed_learning_iteration = completed
        return loaded_dict["infos"]

    def train_mode(self):
        super().train_mode()
        self.alg.amp_discriminator.train()
        self.alg.amp_discriminator.disc_obs_normalizer.train()
        
    def eval_mode(self):
        super().eval_mode()
        self.alg.amp_discriminator.eval()
        self.alg.amp_discriminator.disc_obs_normalizer.eval()
    
    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""
        # Resolve RND config if used
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config if used
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        
        # Resolve AMP config
        self.alg_cfg = resolve_amp_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticCNN | EncoderActorCritic | EncoderMoEActorCritic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )
        
        # Initialize AMP discriminator observation buffers
        disc_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs, 
            device=self.device
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=self.alg_cfg["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=self.env.num_envs, 
            device=self.device
        )

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPOAMP = alg_class(
            actor_critic, storage, disc_obs_buffer, disc_demo_obs_buffer, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        return alg
