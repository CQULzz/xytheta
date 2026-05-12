# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views import XformPrimView

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def root_xy_out_of_bounds(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    """Terminate when an object's ``x, y`` position leaves ``(xmin, xmax, ymin, ymax)``."""
    asset = env.scene[asset_cfg.name]
    if isinstance(asset, RigidObject):
        root_pos_w = asset.data.root_pos_w
    elif isinstance(asset, XformPrimView):
        root_pos_w = asset.get_world_poses()[0]
    else:
        raise TypeError(f"Unsupported asset type for bounds termination: {type(asset)}")
    pos_xy = root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    xmin, xmax, ymin, ymax = bounds
    return (pos_xy[:, 0] < xmin) | (pos_xy[:, 0] > xmax) | (pos_xy[:, 1] < ymin) | (pos_xy[:, 1] > ymax)
