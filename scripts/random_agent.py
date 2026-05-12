# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Use -1 to sample a new seed.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many environment steps.")
parser.add_argument("--action_hold_time", type=float, default=1.0, help="Seconds to keep each random action.")
parser.add_argument("--demo_motion", action="store_true", default=False, help="Use fixed visible car motions.")
parser.add_argument("--reward_guided", action="store_true", default=False, help="Use a reward-inspired exploration controller.")
parser.add_argument("--print_car_poses", action="store_true", default=False, help="Print car poses once per second.")
parser.add_argument("--exploration_csv", type=str, default=None, help="Path for exploration area CSV output.")
parser.add_argument(
    "--disable_exploration_csv", action="store_true", default=False, help="Disable exploration area CSV output."
)
parser.add_argument(
    "--split_exploration_csv_by_env",
    action="store_true",
    default=False,
    help="Write one exploration CSV per environment instead of one combined CSV.",
)
parser.add_argument("--exploration_grid_resolution", type=float, default=0.05, help="Exploration grid size in meters.")
parser.add_argument("--exploration_log_interval", type=float, default=0.5, help="Seconds between CSV rows.")
parser.add_argument(
    "--disable_visual_sync",
    action="store_true",
    default=False,
    help="Disable copying rigid-body poses back to USD for GUI visualization.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg

import xytheta_transa.tasks  # noqa: F401
from xytheta_transa.tasks.manager_based.xytheta_transa.xytheta_transa_env_cfg import ARENA_BOUNDS, LIDAR_MAX_DISTANCE


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class ExplorationCsvLogger:
    """Track per-car explored 2D area and write area-vs-time rows to CSV."""

    def __init__(
        self,
        env,
        csv_path: str | None,
        bounds: tuple[float, float, float, float],
        grid_resolution: float,
        robot_radius: float,
        lidar_max_distance: float,
        log_interval_s: float,
        split_by_env: bool,
    ):
        self.env = env.unwrapped
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.num_cars = 3
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.grid_resolution = grid_resolution
        self.robot_radius = robot_radius
        self.lidar_max_distance = lidar_max_distance
        self.split_by_env = split_by_env
        self.log_every_steps = max(1, round(log_interval_s / self.env.step_dt))
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.grid_width = math.ceil((self.xmax - self.xmin) / grid_resolution)
        self.grid_height = math.ceil((self.ymax - self.ymin) / grid_resolution)
        xs = self.xmin + (torch.arange(self.grid_width, device=self.device) + 0.5) * grid_resolution
        ys = self.ymin + (torch.arange(self.grid_height, device=self.device) + 0.5) * grid_resolution
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.grid_xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        self.explored = torch.zeros(
            self.num_envs, self.num_cars, self.grid_height, self.grid_width, dtype=torch.bool, device=self.device
        )

        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"logs/exploration/exploration_{timestamp}.csv"
        self.csv_paths = self._make_csv_paths(Path(csv_path))
        self._files = []
        self._writers = []
        header = ["time_s", "step", "env_id", "episode", "car_id", "explored_area_m2", "explored_ratio"]
        for path in self.csv_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = path.open("w", newline="", buffering=1)
            writer = csv.writer(csv_file)
            writer.writerow(header)
            self._files.append(csv_file)
            self._writers.append(writer)
        print(f"[INFO]: Writing exploration CSV to {self.csv_path_text}")

    def close(self):
        for csv_file in self._files:
            csv_file.close()

    @property
    def csv_path_text(self) -> str:
        if len(self.csv_paths) == 1:
            return str(self.csv_paths[0])
        return ", ".join(str(path) for path in self.csv_paths)

    def _make_csv_paths(self, csv_path: Path) -> list[Path]:
        if csv_path.suffix == "":
            csv_path = csv_path.with_suffix(".csv")
        if not self.split_by_env:
            return [csv_path]
        return [csv_path.with_name(f"{csv_path.stem}_env{env_id}{csv_path.suffix}") for env_id in range(self.num_envs)]

    def mark_and_maybe_write(self, step_count: int, done: torch.Tensor | None = None):
        if done is not None and torch.any(done):
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            self.explored[done_ids] = False
            self.episode_ids[done_ids] += 1
        self._mark_lidar_visible_neighborhoods()
        if step_count % self.log_every_steps == 0:
            self._write_rows(step_count)

    def _write_rows(self, step_count: int):
        explored_cells = self.explored.reshape(self.num_envs, self.num_cars, -1).sum(dim=2)
        explored_area = explored_cells.to(torch.float32) * (self.grid_resolution**2)
        total_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        explored_ratio = explored_area / total_area
        time_s = step_count * self.env.step_dt
        for env_id in range(self.num_envs):
            writer = self._writers[env_id] if self.split_by_env else self._writers[0]
            for car_id in range(self.num_cars):
                writer.writerow(
                    [
                        f"{time_s:.4f}",
                        step_count,
                        env_id,
                        int(self.episode_ids[env_id].item()),
                        car_id,
                        f"{explored_area[env_id, car_id].item():.6f}",
                        f"{explored_ratio[env_id, car_id].item():.6f}",
                    ]
                )

    def _mark_lidar_visible_neighborhoods(self):
        sample_count = math.ceil(self.lidar_max_distance / self.grid_resolution) + 1
        ray_samples = torch.linspace(0.0, self.lidar_max_distance, sample_count, device=self.device)
        env_ids_template = torch.arange(self.num_envs, device=self.device)

        for car_id in range(self.num_cars):
            sensor = self.env.scene.sensors[f"lidar_{car_id}"]
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

            ray_starts_xy = ray_starts_w[:, :, :2] - self.env.scene.env_origins[:, None, :2]
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
            self.explored[env_ids[inside], car_id, iy[inside], ix[inside]] = True


class RewardGuidedExplorer:
    """Greedy controller that steers cars toward nearby unexplored area."""

    def __init__(
        self,
        env,
        bounds: tuple[float, float, float, float],
        grid_resolution: float,
        robot_radius: float,
        lidar_max_distance: float,
        lookahead_distance: float = 2.5,
        sector_count: int = 24,
    ):
        self.env = env.unwrapped
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.num_cars = 3
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.grid_resolution = grid_resolution
        self.robot_radius = robot_radius
        self.lidar_max_distance = lidar_max_distance
        self.lookahead_distance = lookahead_distance

        self.grid_width = math.ceil((self.xmax - self.xmin) / grid_resolution)
        self.grid_height = math.ceil((self.ymax - self.ymin) / grid_resolution)
        xs = self.xmin + (torch.arange(self.grid_width, device=self.device) + 0.5) * grid_resolution
        ys = self.ymin + (torch.arange(self.grid_height, device=self.device) + 0.5) * grid_resolution
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.grid_xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        self.explored = torch.zeros(
            self.num_envs, self.num_cars, self.grid_height, self.grid_width, dtype=torch.bool, device=self.device
        )
        self.heading_angles = torch.linspace(
            -math.pi, math.pi, sector_count + 1, device=self.device, dtype=torch.float32
        )[:-1]
        self.heading_dirs = torch.stack((torch.cos(self.heading_angles), torch.sin(self.heading_angles)), dim=-1)
        self.blocked_steps = torch.zeros(self.num_envs, self.num_cars, dtype=torch.long, device=self.device)
        self.recovery_steps = torch.zeros_like(self.blocked_steps)
        self.recovery_turn = torch.ones(self.num_envs, self.num_cars, dtype=torch.float32, device=self.device)

    def compute_actions(self, done: torch.Tensor | None = None) -> torch.Tensor:
        if done is not None and torch.any(done):
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            self.explored[done_ids] = False
            self.blocked_steps[done_ids] = 0
            self.recovery_steps[done_ids] = 0

        self.explored |= self._current_visible_grid()
        actions = torch.zeros(self.env.num_envs, self.num_cars * 2, device=self.device)
        for car_id in range(self.num_cars):
            car = self.env.scene[f"car_{car_id}"]
            car_xy = car.data.root_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
            yaw = euler_xyz_from_quat(car.data.root_quat_w)[2]
            for env_id in range(self.num_envs):
                heading_error = self._best_heading_error(env_id, car_id, car_xy[env_id], yaw[env_id])
                front_clearance, turn_bias = self._front_clearance_and_turn_bias(env_id, car_id, yaw[env_id])
                yaw_action = torch.clamp(heading_error / 1.0, -1.0, 1.0)
                forward_action = torch.where(torch.abs(heading_error) < 0.75, 0.85, 0.25)

                if self.recovery_steps[env_id, car_id] > 0:
                    self.recovery_steps[env_id, car_id] -= 1
                    actions[env_id, 2 * car_id] = -0.45
                    actions[env_id, 2 * car_id + 1] = self.recovery_turn[env_id, car_id]
                    continue

                distance_to_wall = self._distance_to_wall(car_xy[env_id])
                if distance_to_wall < 0.8:
                    yaw_action = self._turn_toward_arena_center(car_xy[env_id], yaw[env_id])
                    forward_action = torch.tensor(0.25, device=self.device)
                if front_clearance < 0.65:
                    self.blocked_steps[env_id, car_id] += 1
                    yaw_action = turn_bias
                    forward_action = torch.tensor(0.0, device=self.device)
                    if front_clearance < 0.4 or self.blocked_steps[env_id, car_id] > 10:
                        self.recovery_steps[env_id, car_id] = 18
                        self.recovery_turn[env_id, car_id] = turn_bias
                        forward_action = torch.tensor(-0.45, device=self.device)
                else:
                    self.blocked_steps[env_id, car_id] = 0

                actions[env_id, 2 * car_id] = forward_action
                actions[env_id, 2 * car_id + 1] = yaw_action
        return actions

    def _best_heading_error(self, env_id: int, car_id: int, car_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        unknown = ~self.explored[env_id, car_id].reshape(-1)
        delta = self.grid_xy - car_xy.unsqueeze(0)
        distances = torch.linalg.norm(delta, dim=-1)
        nearby_unknown = unknown & (distances > self.robot_radius * 0.8) & (distances < self.lookahead_distance)
        if not torch.any(nearby_unknown):
            return torch.tensor(0.65, device=self.device)

        unit_delta = delta[nearby_unknown] / distances[nearby_unknown].unsqueeze(-1).clamp_min(1.0e-6)
        sector_scores = torch.matmul(unit_delta, self.heading_dirs.T).clamp_min(0.0).pow(2).sum(dim=0)

        candidate_points = car_xy.unsqueeze(0) + self.heading_dirs * self.lookahead_distance
        candidate_wall_distance = torch.minimum(
            torch.minimum(candidate_points[:, 0] - self.xmin, self.xmax - candidate_points[:, 0]),
            torch.minimum(candidate_points[:, 1] - self.ymin, self.ymax - candidate_points[:, 1]),
        )
        sector_scores -= torch.clamp(0.75 - candidate_wall_distance, min=0.0) * 200.0
        sector_scores += 0.01 * torch.cos(_wrap_to_pi(self.heading_angles - yaw))

        best_angle = self.heading_angles[torch.argmax(sector_scores)]
        return _wrap_to_pi(best_angle - yaw)

    def _front_clearance_and_turn_bias(self, env_id: int, car_id: int, yaw: torch.Tensor) -> tuple[float, torch.Tensor]:
        sensor = self.env.scene.sensors[f"lidar_{car_id}"]
        _ = sensor.data
        ray_starts_w = sensor._ray_starts_w[env_id]
        ray_directions_w = sensor._ray_directions_w[env_id]
        ray_hits_w = sensor.data.ray_hits_w[env_id]
        hit_distances = torch.linalg.norm(ray_hits_w - ray_starts_w, dim=-1)
        hit_distances = torch.nan_to_num(
            hit_distances, nan=self.lidar_max_distance, posinf=self.lidar_max_distance, neginf=self.lidar_max_distance
        ).clamp(max=self.lidar_max_distance)

        directions_xy = ray_directions_w[:, :2]
        directions_xy = directions_xy / torch.linalg.norm(directions_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)))
        right = torch.stack((-torch.sin(yaw), torch.cos(yaw)))
        forward_dot = torch.matmul(directions_xy, forward)
        right_dot = torch.matmul(directions_xy, right)

        front_mask = forward_dot > math.cos(math.radians(35.0))
        front_clearance = hit_distances[front_mask].min().item() if torch.any(front_mask) else self.lidar_max_distance
        left_mask = (forward_dot > 0.0) & (right_dot < -0.15)
        right_mask = (forward_dot > 0.0) & (right_dot > 0.15)
        left_clearance = hit_distances[left_mask].min() if torch.any(left_mask) else torch.tensor(self.lidar_max_distance, device=self.device)
        right_clearance = hit_distances[right_mask].min() if torch.any(right_mask) else torch.tensor(self.lidar_max_distance, device=self.device)
        turn_bias = torch.where(left_clearance > right_clearance, 0.85, -0.85)
        return front_clearance, turn_bias

    def _distance_to_wall(self, car_xy: torch.Tensor) -> float:
        distance = torch.minimum(
            torch.minimum(car_xy[0] - self.xmin, self.xmax - car_xy[0]),
            torch.minimum(car_xy[1] - self.ymin, self.ymax - car_xy[1]),
        )
        return distance.item()

    def _turn_toward_arena_center(self, car_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        target_angle = torch.atan2(-car_xy[1], -car_xy[0])
        return torch.clamp(_wrap_to_pi(target_angle - yaw), -1.0, 1.0)

    def _current_visible_grid(self) -> torch.Tensor:
        visible = torch.zeros(
            self.num_envs, self.num_cars, self.grid_height, self.grid_width, dtype=torch.bool, device=self.device
        )
        sample_count = math.ceil(self.robot_radius / self.grid_resolution) + 1
        ray_samples = torch.linspace(0.0, self.robot_radius, sample_count, device=self.device)
        env_ids_template = torch.arange(self.num_envs, device=self.device)

        for car_id in range(self.num_cars):
            sensor = self.env.scene.sensors[f"lidar_{car_id}"]
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

            ray_starts_xy = ray_starts_w[:, :, :2] - self.env.scene.env_origins[:, None, :2]
            ray_directions_xy = ray_directions_w[:, :, :2]
            xy_direction_norm = torch.linalg.norm(ray_directions_xy, dim=-1).clamp_min(1.0e-6)
            max_samples_inside_radius = self.robot_radius / xy_direction_norm
            max_visible_samples = torch.minimum(hit_distances, max_samples_inside_radius)
            valid_samples = ray_samples.view(1, 1, -1) <= max_visible_samples.unsqueeze(-1)
            points_xy = ray_starts_xy.unsqueeze(2) + ray_directions_xy.unsqueeze(2) * ray_samples.view(1, 1, -1, 1)

            ix = torch.floor((points_xy[..., 0] - self.xmin) / self.grid_resolution).to(torch.long)
            iy = torch.floor((points_xy[..., 1] - self.ymin) / self.grid_resolution).to(torch.long)
            inside = (ix >= 0) & (ix < self.grid_width) & (iy >= 0) & (iy < self.grid_height) & valid_samples
            if torch.any(inside):
                env_ids = env_ids_template.view(-1, 1, 1).expand_as(ix)
                visible[env_ids[inside], car_id, iy[inside], ix[inside]] = True

        return visible


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 2**31 - 1)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        torch.manual_seed(args_cli.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args_cli.seed)
        print(f"[INFO]: Using random seed: {args_cli.seed}")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # In some Isaac Sim/Fabric configurations, RigidObject tensors move while the GUI USD prims stay at their
    # authored poses. Mirroring the rigid-body pose to USD keeps the viewport faithful to the physics state.
    visual_sync_views = []
    if env.unwrapped.sim.has_gui() and not args_cli.disable_visual_sync:
        visual_sync_views = [
            XformPrimView(
                f"/World/envs/env_.*/Car_{car_id}",
                device=env.unwrapped.device,
                stage=env.unwrapped.scene.stage,
                sync_usd_on_fabric_write=True,
            )
            for car_id in range(3)
        ]
        print("[INFO]: Syncing car rigid-body poses back to USD for GUI visualization.")

    def sync_car_visuals_to_usd():
        if not visual_sync_views:
            return
        for car_id, view in enumerate(visual_sync_views):
            car = env.unwrapped.scene[f"car_{car_id}"]
            view.set_world_poses(
                positions=car.data.root_pos_w.detach(),
                orientations=car.data.root_quat_w.detach(),
            )
        env.unwrapped.sim.render()

    def project_cars_to_planar_pose():
        for car_id in range(3):
            car = env.unwrapped.scene[f"car_{car_id}"]
            yaw = euler_xyz_from_quat(car.data.root_quat_w)[2]
            zeros = torch.zeros_like(yaw)
            root_pose_w = torch.cat(
                (
                    car.data.root_pos_w[:, :2],
                    car.data.default_root_state[:, 2:3] + env.unwrapped.scene.env_origins[:, 2:3],
                    quat_from_euler_xyz(zeros, zeros, yaw),
                ),
                dim=-1,
            )
            root_vel_w = car.data.root_vel_w.clone()
            root_vel_w[:, 2] = 0.0
            root_vel_w[:, 3:5] = 0.0
            car.write_root_pose_to_sim(root_pose_w)
            car.write_root_velocity_to_sim(root_vel_w)

    project_cars_to_planar_pose()
    sync_car_visuals_to_usd()
    reward_guided_agent = None
    if args_cli.reward_guided:
        reward_guided_agent = RewardGuidedExplorer(
            env=env,
            bounds=ARENA_BOUNDS,
            grid_resolution=args_cli.exploration_grid_resolution,
            robot_radius=1.0,
            lidar_max_distance=LIDAR_MAX_DISTANCE,
        )
        print("[INFO]: Using reward-guided exploration controller instead of random actions.")
    exploration_logger = None
    if not args_cli.disable_exploration_csv:
        exploration_logger = ExplorationCsvLogger(
            env=env,
            csv_path=args_cli.exploration_csv,
            bounds=ARENA_BOUNDS,
            grid_resolution=args_cli.exploration_grid_resolution,
            robot_radius=1.0,
            lidar_max_distance=LIDAR_MAX_DISTANCE,
            log_interval_s=args_cli.exploration_log_interval,
            split_by_env=args_cli.split_exploration_csv_by_env,
        )
        exploration_logger.mark_and_maybe_write(step_count=0)
    # hold each sampled random action for a fixed amount of simulated time
    hold_steps = max(1, round(args_cli.action_hold_time / env.unwrapped.step_dt))
    if reward_guided_agent is None:
        print(f"[INFO]: Holding each random action for {hold_steps} env steps ({args_cli.action_hold_time:.3f} s).")
    else:
        print("[INFO]: Recomputing reward-guided actions every environment step.")
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    demo_actions = torch.tensor([[1.0, 0.35, 1.0, -0.35, 0.8, 0.6]], device=env.unwrapped.device)
    step_count = 0
    done = None
    try:
        # simulate environment
        while simulation_app.is_running() and (args_cli.max_steps is None or step_count < args_cli.max_steps):
            # run everything in inference mode
            with torch.inference_mode():
                if reward_guided_agent is not None:
                    actions = reward_guided_agent.compute_actions(done)
                elif args_cli.demo_motion:
                    actions[:] = demo_actions
                # sample actions from -1 to 1 and hold them for a while
                elif step_count % hold_steps == 0:
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions
                _, _, terminated, truncated, _ = env.step(actions)
                project_cars_to_planar_pose()
                sync_car_visuals_to_usd()
                next_step_count = step_count + 1
                if exploration_logger is not None:
                    done = terminated | truncated
                    exploration_logger.mark_and_maybe_write(next_step_count, done=done)
                else:
                    done = terminated | truncated
                if args_cli.print_car_poses and step_count % max(1, round(1.0 / env.unwrapped.step_dt)) == 0:
                    pose_parts = []
                    for car_id in range(3):
                        car = env.unwrapped.scene[f"car_{car_id}"]
                        pos = car.data.root_pos_w[0, :2].detach().cpu().numpy()
                        pose_parts.append(f"car_{car_id}=({pos[0]:+.2f}, {pos[1]:+.2f})")
                    print(f"[POSE step={step_count:06d}] " + " ".join(pose_parts), flush=True)
                step_count = next_step_count
    except KeyboardInterrupt:
        print("[INFO]: Interrupted by user.")
    finally:
        if exploration_logger is not None:
            exploration_logger.close()
            print(f"[INFO]: Exploration CSV saved to {exploration_logger.csv_path_text}")
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
