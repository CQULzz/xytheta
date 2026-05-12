# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from . import mdp

ARENA_BOUNDS = (-4.5, 4.5, -4.5, 4.5)
LIDAR_MAX_DISTANCE = 8.0
OBSTACLE_NAMES = (
    "Obstacle_Wall_North",
    "Obstacle_Wall_South",
    "Obstacle_Wall_East",
    "Obstacle_Wall_West",
    "Obstacle_Box_0",
    "Obstacle_Box_1",
    "Obstacle_Column_0",
    "Obstacle_Column_1",
)


def _car_cfg(name: str, pos: tuple[float, float, float], color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.72, 0.42, 0.24),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                linear_damping=0.8,
                angular_damping=0.8,
                max_linear_velocity=2.5,
                max_angular_velocity=360.0,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.03,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _wheel_cfg(car_name: str, wheel_name: str, pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{car_name}/{wheel_name}",
        spawn=sim_utils.CylinderCfg(
            radius=0.12,
            height=0.08,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02), roughness=0.65),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _front_marker_cfg(car_name: str) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{car_name}/Front_Marker",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.30, 0.035),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.92, 0.12), roughness=0.45),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.32, 0.0, 0.16)),
    )


def _box_obstacle(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float] = (0.35, 0.35, 0.35),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.03,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _cylinder_obstacle(
    name: str,
    radius: float,
    height: float,
    pos: tuple[float, float, float],
    color: tuple[float, float, float] = (0.28, 0.42, 0.46),
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.03,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _lidar_cfg(car_name: str) -> MultiMeshRayCasterCfg:
    return MultiMeshRayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{car_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=4,
            vertical_fov_range=(-4.0, 4.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        max_distance=LIDAR_MAX_DISTANCE,
        debug_vis=True,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=f"{{ENV_REGEX_NS}}/{obstacle_name}",
                is_shared=True,
                track_mesh_transforms=False,
            )
            for obstacle_name in OBSTACLE_NAMES
        ],
    )


@configclass
class XythetaTransaSceneCfg(InteractiveSceneCfg):
    """Arena with multiple planar cars and simple LiDAR-visible obstacles."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0)),
    )

    car_0 = _car_cfg("Car_0", pos=(-2.8, -2.2, 0.12), color=(0.78, 0.18, 0.12))
    car_1 = _car_cfg("Car_1", pos=(-2.6, 2.0, 0.12), color=(0.12, 0.38, 0.82))
    car_2 = _car_cfg("Car_2", pos=(2.7, -1.8, 0.12), color=(0.22, 0.62, 0.30))

    car_0_front_marker = _front_marker_cfg("Car_0")
    car_0_wheel_front_left = _wheel_cfg("Car_0", "Wheel_FL", (0.24, 0.25, -0.06))
    car_0_wheel_front_right = _wheel_cfg("Car_0", "Wheel_FR", (0.24, -0.25, -0.06))
    car_0_wheel_rear_left = _wheel_cfg("Car_0", "Wheel_RL", (-0.24, 0.25, -0.06))
    car_0_wheel_rear_right = _wheel_cfg("Car_0", "Wheel_RR", (-0.24, -0.25, -0.06))

    car_1_front_marker = _front_marker_cfg("Car_1")
    car_1_wheel_front_left = _wheel_cfg("Car_1", "Wheel_FL", (0.24, 0.25, -0.06))
    car_1_wheel_front_right = _wheel_cfg("Car_1", "Wheel_FR", (0.24, -0.25, -0.06))
    car_1_wheel_rear_left = _wheel_cfg("Car_1", "Wheel_RL", (-0.24, 0.25, -0.06))
    car_1_wheel_rear_right = _wheel_cfg("Car_1", "Wheel_RR", (-0.24, -0.25, -0.06))

    car_2_front_marker = _front_marker_cfg("Car_2")
    car_2_wheel_front_left = _wheel_cfg("Car_2", "Wheel_FL", (0.24, 0.25, -0.06))
    car_2_wheel_front_right = _wheel_cfg("Car_2", "Wheel_FR", (0.24, -0.25, -0.06))
    car_2_wheel_rear_left = _wheel_cfg("Car_2", "Wheel_RL", (-0.24, 0.25, -0.06))
    car_2_wheel_rear_right = _wheel_cfg("Car_2", "Wheel_RR", (-0.24, -0.25, -0.06))

    obstacle_wall_north = _box_obstacle("Obstacle_Wall_North", (9.5, 0.18, 0.7), (0.0, 4.6, 0.35))
    obstacle_wall_south = _box_obstacle("Obstacle_Wall_South", (9.5, 0.18, 0.7), (0.0, -4.6, 0.35))
    obstacle_wall_east = _box_obstacle("Obstacle_Wall_East", (0.18, 9.5, 0.7), (4.6, 0.0, 0.35))
    obstacle_wall_west = _box_obstacle("Obstacle_Wall_West", (0.18, 9.5, 0.7), (-4.6, 0.0, 0.35))
    obstacle_box_0 = _box_obstacle("Obstacle_Box_0", (1.3, 0.55, 0.8), (-0.9, -0.2, 0.4), (0.48, 0.36, 0.22))
    obstacle_box_1 = _box_obstacle("Obstacle_Box_1", (0.55, 1.5, 0.8), (1.2, 1.35, 0.4), (0.42, 0.32, 0.52))
    obstacle_column_0 = _cylinder_obstacle("Obstacle_Column_0", 0.36, 0.9, (2.3, -0.05, 0.45))
    obstacle_column_1 = _cylinder_obstacle("Obstacle_Column_1", 0.28, 0.9, (-1.9, 1.55, 0.45))

    lidar_0 = _lidar_cfg("Car_0")
    lidar_1 = _lidar_cfg("Car_1")
    lidar_2 = _lidar_cfg("Car_2")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=650.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the three cars."""

    car_0_drive = mdp.PlanarVelocityActionCfg(asset_name="car_0", scale=(1.2, 1.8))
    car_1_drive = mdp.PlanarVelocityActionCfg(asset_name="car_1", scale=(1.2, 1.8))
    car_2_drive = mdp.PlanarVelocityActionCfg(asset_name="car_2", scale=(1.2, 1.8))


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        car_0_pose = ObsTerm(func=mdp.planar_pose, params={"asset_cfg": SceneEntityCfg("car_0")})
        car_0_lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_0"), "max_distance": LIDAR_MAX_DISTANCE},
        )

        car_1_pose = ObsTerm(func=mdp.planar_pose, params={"asset_cfg": SceneEntityCfg("car_1")})
        car_1_lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_1"), "max_distance": LIDAR_MAX_DISTANCE},
        )

        car_2_pose = ObsTerm(func=mdp.planar_pose, params={"asset_cfg": SceneEntityCfg("car_2")})
        car_2_lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_2"), "max_distance": LIDAR_MAX_DISTANCE},
        )

        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset events."""

    reset_car_0 = EventTerm(
        func=mdp.reset_planar_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("car_0"),
            "default_pos": (-2.8, -2.2, 0.12),
            "pose_range": {"yaw": (-math.pi, math.pi)},
        },
    )
    reset_car_1 = EventTerm(
        func=mdp.reset_planar_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("car_1"),
            "default_pos": (-2.6, 2.0, 0.12),
            "pose_range": {"yaw": (-math.pi, math.pi)},
        },
    )
    reset_car_2 = EventTerm(
        func=mdp.reset_planar_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("car_2"),
            "default_pos": (2.7, -1.8, 0.12),
            "pose_range": {"yaw": (-math.pi, math.pi)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for simple free-driving experiments."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class RewardsCfgV1:
    """Reward terms for exploration-driven LiDAR coverage."""

    exploration = RewTerm(
        func=mdp.lidar_new_area_reward,
        weight=1.0,
        params={
            "car_names": ("car_0", "car_1", "car_2"),
            "lidar_names": ("lidar_0", "lidar_1", "lidar_2"),
            "bounds": ARENA_BOUNDS,
            "grid_resolution": 0.05,
            "robot_radius": 1.0,
            "lidar_max_distance": LIDAR_MAX_DISTANCE,
        },
    )
    wall_proximity = RewTerm(
        func=mdp.wall_proximity_penalty,
        weight=-2.0,
        params={
            "car_names": ("car_0", "car_1", "car_2"),
            "bounds": ARENA_BOUNDS,
            "margin": 0.75,
        },
    )
    terminating = RewTerm(func=mdp.is_terminated, weight=-30.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    car_0_out_of_bounds = DoneTerm(
        func=mdp.root_xy_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("car_0"), "bounds": ARENA_BOUNDS},
    )
    car_1_out_of_bounds = DoneTerm(
        func=mdp.root_xy_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("car_1"), "bounds": ARENA_BOUNDS},
    )
    car_2_out_of_bounds = DoneTerm(
        func=mdp.root_xy_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("car_2"), "bounds": ARENA_BOUNDS},
    )


@configclass
class XythetaTransaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment with three planar LiDAR cars."""

    scene: XythetaTransaSceneCfg = XythetaTransaSceneCfg(num_envs=1, env_spacing=12.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 45.0
        self.viewer.eye = (7.5, -7.5, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


@configclass
class XythetaTransaEnvCfgV1(XythetaTransaEnvCfg):
    """Exploration-reward variant of the base three-car LiDAR environment."""

    rewards: RewardsCfgV1 = RewardsCfgV1()
