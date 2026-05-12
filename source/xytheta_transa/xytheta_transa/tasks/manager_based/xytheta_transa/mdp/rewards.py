# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


class lidar_new_area_reward(ManagerTermBase):
    """Reward newly explored LiDAR-visible area for each car independently."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.car_names = tuple(cfg.params["car_names"])
        self.lidar_names = tuple(cfg.params["lidar_names"])
        self.xmin, self.xmax, self.ymin, self.ymax = cfg.params["bounds"]
        self.grid_resolution = cfg.params["grid_resolution"]
        self.robot_radius = cfg.params["robot_radius"]
        self.lidar_max_distance = cfg.params["lidar_max_distance"]
        self.num_cars = len(self.car_names)

        self.grid_width = int(torch.ceil(torch.tensor((self.xmax - self.xmin) / self.grid_resolution)).item())
        self.grid_height = int(torch.ceil(torch.tensor((self.ymax - self.ymin) / self.grid_resolution)).item())
        self.explored = torch.zeros(
            env.num_envs, self.num_cars, self.grid_height, self.grid_width, dtype=torch.bool, device=env.device
        )
        self._baseline_pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.explored[env_ids] = False
        self._baseline_pending[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        car_names: tuple[str, ...],
        lidar_names: tuple[str, ...],
        bounds: tuple[float, float, float, float],
        grid_resolution: float,
        robot_radius: float,
        lidar_max_distance: float,
    ) -> torch.Tensor:
        visible = self._compute_visible_grid(env)
        new_cells = visible & ~self.explored
        new_area = new_cells.reshape(env.num_envs, self.num_cars, -1).sum(dim=(1, 2)).to(torch.float32)
        new_area = new_area * (self.grid_resolution**2)

        if torch.any(self._baseline_pending):
            new_area[self._baseline_pending] = 0.0
            self._baseline_pending[:] = False

        self.explored |= visible
        # RewardManager multiplies by dt; divide here so the final reward is measured in square meters per step.
        return new_area / env.step_dt

    def _compute_visible_grid(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        visible = torch.zeros(
            env.num_envs, self.num_cars, self.grid_height, self.grid_width, dtype=torch.bool, device=env.device
        )
        sample_count = int(torch.ceil(torch.tensor(self.robot_radius / self.grid_resolution)).item()) + 1
        ray_samples = torch.linspace(0.0, self.robot_radius, sample_count, device=env.device)
        env_ids_template = torch.arange(env.num_envs, device=env.device)

        for car_id, lidar_name in enumerate(self.lidar_names):
            sensor = env.scene.sensors[lidar_name]
            _ = sensor.data
            ray_starts_w = sensor._ray_starts_w
            ray_directions_w = sensor._ray_directions_w
            ray_hits_w = sensor.data.ray_hits_w

            hit_distances = torch.linalg.norm(ray_hits_w - ray_starts_w, dim=-1)
            hit_is_valid = torch.isfinite(hit_distances) & (hit_distances > 0.0)
            hit_distances = torch.where(
                hit_is_valid,
                torch.clamp(hit_distances, max=self.lidar_max_distance),
                torch.full_like(hit_distances, self.lidar_max_distance),
            )

            ray_starts_xy = ray_starts_w[:, :, :2] - env.scene.env_origins[:, None, :2]
            ray_directions_xy = ray_directions_w[:, :, :2]
            xy_direction_norm = torch.linalg.norm(ray_directions_xy, dim=-1).clamp_min(1.0e-6)
            max_samples_inside_radius = self.robot_radius / xy_direction_norm
            max_visible_samples = torch.minimum(hit_distances, max_samples_inside_radius)
            valid_samples = ray_samples.view(1, 1, -1) <= max_visible_samples.unsqueeze(-1)
            points_xy = ray_starts_xy.unsqueeze(2) + ray_directions_xy.unsqueeze(2) * ray_samples.view(1, 1, -1, 1)

            ix = torch.floor((points_xy[..., 0] - self.xmin) / self.grid_resolution).to(torch.long)
            iy = torch.floor((points_xy[..., 1] - self.ymin) / self.grid_resolution).to(torch.long)
            inside = (ix >= 0) & (ix < self.grid_width) & (iy >= 0) & (iy < self.grid_height) & valid_samples
            if not torch.any(inside):
                continue

            env_ids = env_ids_template.view(-1, 1, 1).expand_as(ix)
            visible[env_ids[inside], car_id, iy[inside], ix[inside]] = True

        return visible


def wall_proximity_penalty(
    env: ManagerBasedRLEnv,
    car_names: tuple[str, ...],
    bounds: tuple[float, float, float, float],
    margin: float,
) -> torch.Tensor:
    """Penalize cars as their root positions approach the arena boundary."""
    xmin, xmax, ymin, ymax = bounds
    penalty = torch.zeros(env.num_envs, device=env.device)
    for car_name in car_names:
        asset = env.scene[car_name]
        if not isinstance(asset, RigidObject):
            raise TypeError(f"wall_proximity_penalty expected a RigidObject for '{car_name}', got {type(asset)}.")
        pos_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        distance_to_wall = torch.minimum(
            torch.minimum(pos_xy[:, 0] - xmin, xmax - pos_xy[:, 0]),
            torch.minimum(pos_xy[:, 1] - ymin, ymax - pos_xy[:, 1]),
        )
        penalty += torch.clamp((margin - distance_to_wall) / margin, min=0.0) ** 2
    return penalty
