# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _root_pose_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    if isinstance(asset, RigidObject):
        return asset.data.root_pos_w, asset.data.root_quat_w
    if isinstance(asset, XformPrimView):
        return asset.get_world_poses()
    raise TypeError(f"Unsupported asset type for planar pose observation: {type(asset)}")


def planar_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return planar pose as ``x, y, cos(yaw), sin(yaw)`` in the environment frame."""
    root_pos_w, root_quat_w = _root_pose_w(env, asset_cfg)
    pos_xy = root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    yaw = euler_xyz_from_quat(root_quat_w)[2]
    return torch.cat((pos_xy, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)), dim=-1)


def planar_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return planar body velocity as ``forward_velocity, lateral_velocity, yaw_rate``."""
    asset = env.scene[asset_cfg.name]
    if isinstance(asset, RigidObject):
        lin_vel_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)
        return torch.cat((lin_vel_b[:, :2], asset.data.root_ang_vel_w[:, 2].unsqueeze(-1)), dim=-1)
    return torch.zeros(env.num_envs, 3, device=env.device)


def lidar_ranges(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Return ray-cast LiDAR ranges, replacing misses with ``max_distance``."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w
    ranges = torch.linalg.norm(ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    ranges = torch.nan_to_num(ranges, nan=max_distance, posinf=max_distance, neginf=max_distance)
    ranges = torch.clamp(ranges, max=max_distance)
    return ranges / max_distance if normalize else ranges
