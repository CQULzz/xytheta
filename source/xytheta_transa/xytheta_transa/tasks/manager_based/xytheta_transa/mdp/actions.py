# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv


class PlanarVelocityAction(ActionTerm):
    """Apply planar forward/yaw velocity commands to a planar car body."""

    cfg: "PlanarVelocityActionCfg"
    _asset: RigidObject | XformPrimView

    def __init__(self, cfg: "PlanarVelocityActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if isinstance(self._asset, XformPrimView):
            self._asset._sync_usd_on_fabric_write = True
        elif not isinstance(self._asset, RigidObject):
            raise TypeError(
                "PlanarVelocityAction expects a RigidObject or XformPrimView asset, "
                f"got {type(self._asset)} for '{self.cfg.asset_name}'."
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        forward_vel = self._processed_actions[:, 0]
        yaw_rate = self._processed_actions[:, 1]

        if isinstance(self._asset, RigidObject):
            yaw_w = euler_xyz_from_quat(self._asset.data.root_quat_w)[2]
            zeros = torch.zeros_like(yaw_w)
            root_pose_w = torch.cat(
                (
                    self._asset.data.root_pos_w[:, :2],
                    self._asset.data.default_root_state[:, 2:3] + self._env.scene.env_origins[:, 2:3],
                    quat_from_euler_xyz(zeros, zeros, yaw_w),
                ),
                dim=-1,
            )
            self._asset.write_root_pose_to_sim(root_pose_w)

            root_vel_w = torch.zeros_like(self._asset.data.root_vel_w)
            root_vel_w[:, 0] = torch.cos(yaw_w) * forward_vel
            root_vel_w[:, 1] = torch.sin(yaw_w) * forward_vel
            root_vel_w[:, 5] = yaw_rate
            self._asset.write_root_velocity_to_sim(root_vel_w)
            return

        root_pos_w, root_quat_w = self._asset.get_world_poses()
        yaw_w = euler_xyz_from_quat(root_quat_w)[2]

        dt = self._env.physics_dt
        root_pos_w[:, 0] += torch.cos(yaw_w) * forward_vel * dt
        root_pos_w[:, 1] += torch.sin(yaw_w) * forward_vel * dt
        yaw_w = yaw_w + yaw_rate * dt

        zeros = torch.zeros_like(yaw_w)
        root_quat_w = quat_from_euler_xyz(zeros, zeros, yaw_w)
        self._asset.set_world_poses(root_pos_w, root_quat_w)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class PlanarVelocityActionCfg(ActionTermCfg):
    """Configuration for planar velocity control of a car body."""

    class_type: type[ActionTerm] = PlanarVelocityAction

    scale: tuple[float, float] = (1.0, 1.0)
    """Scale for normalized ``[forward_velocity, yaw_rate]`` actions."""

    offset: tuple[float, float] = (0.0, 0.0)
    """Offset for normalized ``[forward_velocity, yaw_rate]`` actions."""


class DifferentialWheelVelocityAction(ActionTerm):
    """Map ``[forward_velocity, yaw_rate]`` commands to differential wheel velocity targets."""

    cfg: "DifferentialWheelVelocityActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "DifferentialWheelVelocityActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not isinstance(self._asset, Articulation):
            raise TypeError(
                "DifferentialWheelVelocityAction expects an Articulation asset, "
                f"got {type(self._asset)} for '{self.cfg.asset_name}'."
            )

        self._left_joint_ids, self._left_joint_names = self._asset.find_joints(
            self.cfg.left_joint_names, preserve_order=True
        )
        self._right_joint_ids, self._right_joint_names = self._asset.find_joints(
            self.cfg.right_joint_names, preserve_order=True
        )
        if len(self._left_joint_ids) == 0 or len(self._right_joint_ids) == 0:
            raise ValueError(
                "DifferentialWheelVelocityAction needs at least one left and one right wheel joint. "
                f"Resolved left={self._left_joint_names}, right={self._right_joint_names}."
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)
        self._left_targets = torch.zeros(self.num_envs, len(self._left_joint_ids), device=self.device)
        self._right_targets = torch.zeros(self.num_envs, len(self._right_joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        forward_vel = self._processed_actions[:, 0]
        yaw_rate = self._processed_actions[:, 1]

        half_track = 0.5 * self.cfg.wheel_track
        left_wheel_vel = (forward_vel - yaw_rate * half_track) / self.cfg.wheel_radius
        right_wheel_vel = (forward_vel + yaw_rate * half_track) / self.cfg.wheel_radius

        self._left_targets[:] = left_wheel_vel.unsqueeze(-1)
        self._right_targets[:] = right_wheel_vel.unsqueeze(-1)
        self._asset.set_joint_velocity_target(self._left_targets, joint_ids=self._left_joint_ids)
        self._asset.set_joint_velocity_target(self._right_targets, joint_ids=self._right_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class DifferentialWheelVelocityActionCfg(ActionTermCfg):
    """Configuration for differential wheel velocity control."""

    class_type: type[ActionTerm] = DifferentialWheelVelocityAction

    left_joint_names: tuple[str, ...] = (".*left.*wheel_joint",)
    """Regex patterns for left wheel joints."""

    right_joint_names: tuple[str, ...] = (".*right.*wheel_joint",)
    """Regex patterns for right wheel joints."""

    wheel_radius: float = 0.18
    """Wheel radius in meters."""

    wheel_track: float = 0.62
    """Distance between left and right wheel contact centers in meters."""

    scale: tuple[float, float] = (1.0, 1.0)
    """Scale for normalized ``[forward_velocity, yaw_rate]`` actions."""

    offset: tuple[float, float] = (0.0, 0.0)
    """Offset for normalized ``[forward_velocity, yaw_rate]`` actions."""
