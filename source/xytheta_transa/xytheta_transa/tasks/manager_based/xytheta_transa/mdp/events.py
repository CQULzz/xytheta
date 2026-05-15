# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views import XformPrimView

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_planar_root_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    default_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
):
    """Reset a planar car pose and clear its velocity."""
    asset = env.scene[asset_cfg.name]
    if isinstance(asset, XformPrimView):
        asset._sync_usd_on_fabric_write = True
    elif not isinstance(asset, (Articulation, RigidObject)):
        raise TypeError(f"reset_planar_root_pose_uniform received unsupported asset type: {type(asset)}.")

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    default_pos_tensor = torch.tensor(default_pos, device=env.device).unsqueeze(0)
    positions = default_pos_tensor + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    if isinstance(asset, (Articulation, RigidObject)):
        root_pose = torch.cat((positions, orientations_delta), dim=-1)
        root_velocity = torch.zeros(len(env_ids), 6, device=env.device)
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        if isinstance(asset, Articulation):
            joint_pos = asset.data.default_joint_pos[env_ids].clone()
            joint_vel = asset.data.default_joint_vel[env_ids].clone()
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    else:
        asset.set_world_poses(positions=positions, orientations=orientations_delta, indices=env_ids)


def reset_xform_root_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    default_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
):
    """Backward-compatible name for the previous Xform-only reset helper."""
    reset_planar_root_pose_uniform(env, env_ids, pose_range, default_pos, asset_cfg)
