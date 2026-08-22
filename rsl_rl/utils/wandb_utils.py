# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name)
        self._allow_config_change = bool(
            cfg.get("resume", False) or os.environ.get("WANDB_RESUME")
        )
        wandb.config.update(
            {"log_dir": log_dir},
            allow_val_change=self._allow_config_change,
        )
        self._wandb_step: int | None = None
        self._wandb_scalars: dict[str, float] = {}

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        update_kwargs = {
            "allow_val_change": self._allow_config_change,
        }
        wandb.config.update({"runner_cfg": train_cfg}, **update_kwargs)
        wandb.config.update(
            {"policy_cfg": train_cfg["policy"]}, **update_kwargs
        )
        wandb.config.update(
            {"alg_cfg": train_cfg["algorithm"]}, **update_kwargs
        )
        try:
            wandb.config.update(
                {"env_cfg": env_cfg.to_dict()}, **update_kwargs
            )
        except Exception:
            wandb.config.update(
                {"env_cfg": asdict(env_cfg)}, **update_kwargs
            )

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        if self._wandb_step is not None and global_step != self._wandb_step:
            self._flush_wandb_scalars()
        self._wandb_step = global_step
        self._wandb_scalars[tag] = scalar_value

    def _flush_wandb_scalars(self) -> None:
        if not self._wandb_scalars:
            return
        wandb.log(self._wandb_scalars, step=self._wandb_step)
        self._wandb_scalars = {}

    def flush(self) -> None:
        super().flush()
        self._flush_wandb_scalars()

    def stop(self) -> None:
        self._flush_wandb_scalars()
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))
