# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp

OFFROAD_ARENA_BOUNDS = (-7.5, 7.5, -7.5, 7.5)
OFFROAD_LIDAR_MAX_DISTANCE = 10.0
OFFROAD_CAR_NAMES = ("car_0",)
OFFROAD_LIDAR_NAMES = ("lidar_0",)
OFFROAD_CAR_URDF_PATH = Path(__file__).resolve().parents[3] / "assets" / "offroad_diff_drive_car.urdf"


OFFROAD_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=20260515,
    curriculum=False,
    size=(5.0, 5.0),
    border_width=0.4,
    num_rows=3,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.45, 0.9),
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(0.03, 0.18),
            noise_step=0.02,
            border_width=0.2,
        ),
        "grid_rocks": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.05, 0.28),
            platform_width=1.0,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.04, 0.25),
            num_obstacles=28,
            platform_width=1.0,
        ),
        "scattered_cones": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.15,
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=8,
                height=0.08,
                radius=0.16,
                max_yx_angle=12.0,
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=18,
                height=0.30,
                radius=0.34,
                max_yx_angle=28.0,
            ),
            platform_width=1.0,
        ),
        "low_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.04, 0.14),
            step_width=0.35,
            platform_width=1.2,
            border_width=0.3,
            holes=False,
        ),
    },
)


def _offroad_car_cfg(name: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(OFFROAD_CAR_URDF_PATH),
            usd_dir="/tmp/xytheta_transa/offroad_assets",
            usd_file_name="offroad_diff_drive_car.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            root_link_name="base_link",
            merge_fixed_joints=True,
            self_collision=False,
            replace_cylinders_with_capsules=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_linear_velocity=4.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=2.5,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="velocity",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=35.0, damping=0.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "wheel_drive": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                effort_limit_sim=90.0,
                velocity_limit_sim=45.0,
                stiffness=0.0,
                damping=8.0,
            )
        },
    )


def _rock_cfg(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.03),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=0.9, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _cone_rock_cfg(
    name: str,
    radius: float,
    height: float,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.ConeCfg(
            radius=radius,
            height=height,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.03),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.1, dynamic_friction=0.9, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _offroad_lidar_cfg(car_name: str) -> MultiMeshRayCasterCfg:
    return MultiMeshRayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{car_name}/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        max_distance=OFFROAD_LIDAR_MAX_DISTANCE,
        debug_vis=True,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                is_shared=True,
                track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Offroad_.*",
                track_mesh_transforms=False,
            ),
        ],
    )


@configclass
class XythetaTransaOffroadSceneCfg(InteractiveSceneCfg):
    """Fixed 15 m x 15 m off-road scene for single-car exploration data collection."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OFFROAD_TERRAIN_GENERATOR_CFG,
        use_terrain_origins=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.3,
            dynamic_friction=1.1,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.23), roughness=0.95),
        debug_vis=False,
    )

    car_0 = _offroad_car_cfg("Car_0", pos=(0.0, 0.0, 0.65))

    offroad_rock_0 = _rock_cfg("Offroad_Rock_0", (0.8, 0.45, 0.35), (-3.0, -1.8, 0.18), (0.35, 0.32, 0.26))
    offroad_rock_1 = _rock_cfg("Offroad_Rock_1", (0.5, 1.0, 0.28), (2.2, 1.2, 0.14), (0.30, 0.28, 0.24))
    offroad_rock_2 = _rock_cfg("Offroad_Rock_2", (1.0, 0.55, 0.42), (-0.6, 3.1, 0.21), (0.38, 0.34, 0.27))
    offroad_cone_0 = _cone_rock_cfg("Offroad_Cone_0", 0.35, 0.55, (3.5, -2.6, 0.28), (0.28, 0.31, 0.27))
    offroad_cone_1 = _cone_rock_cfg("Offroad_Cone_1", 0.28, 0.45, (-3.6, 2.5, 0.22), (0.32, 0.30, 0.25))

    lidar_0 = _offroad_lidar_cfg("Car_0")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.87, 0.9), intensity=850.0),
    )


@configclass
class OffroadActionsCfg:
    """Action specifications for the off-road differential-drive car."""

    car_0_drive = mdp.DifferentialWheelVelocityActionCfg(
        asset_name="car_0",
        scale=(1.0, 1.4),
        wheel_radius=0.18,
        wheel_track=0.62,
    )


@configclass
class OffroadObservationsCfg:
    """Observation specifications for the off-road MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        car_0_pose = ObsTerm(func=mdp.planar_pose, params={"asset_cfg": SceneEntityCfg("car_0")})
        car_0_lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_0"), "max_distance": OFFROAD_LIDAR_MAX_DISTANCE},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class OffroadEventCfg:
    """Reset events for the off-road car."""

    reset_car_0 = EventTerm(
        func=mdp.reset_planar_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("car_0"),
            "default_pos": (0.0, 0.0, 0.65),
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-math.pi, math.pi)},
        },
    )


@configclass
class OffroadRewardsCfg:
    """Exploration-first reward for off-road data collection."""

    exploration = RewTerm(
        func=mdp.lidar_new_area_reward,
        weight=1.0,
        params={
            "car_names": OFFROAD_CAR_NAMES,
            "lidar_names": OFFROAD_LIDAR_NAMES,
            "bounds": OFFROAD_ARENA_BOUNDS,
            "grid_resolution": 0.05,
            "robot_radius": 1.0,
            "lidar_max_distance": OFFROAD_LIDAR_MAX_DISTANCE,
        },
    )
    terminating = RewTerm(func=mdp.is_terminated, weight=-20.0)


@configclass
class OffroadTerminationsCfg:
    """Safety/reset conditions for the off-road environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    car_0_out_of_bounds = DoneTerm(
        func=mdp.root_xy_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("car_0"), "bounds": OFFROAD_ARENA_BOUNDS},
    )
    car_0_tilt = DoneTerm(
        func=mdp.root_tilt_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("car_0"), "max_tilt": math.radians(70.0)},
    )


@configclass
class XythetaTransaOffroadEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment with one real wheel-contact car on fixed rough terrain."""

    scene: XythetaTransaOffroadSceneCfg = XythetaTransaOffroadSceneCfg(num_envs=1, env_spacing=18.0)
    observations: OffroadObservationsCfg = OffroadObservationsCfg()
    actions: OffroadActionsCfg = OffroadActionsCfg()
    events: OffroadEventCfg = OffroadEventCfg()
    rewards: OffroadRewardsCfg = OffroadRewardsCfg()
    terminations: OffroadTerminationsCfg = OffroadTerminationsCfg()

    arena_bounds: tuple[float, float, float, float] = OFFROAD_ARENA_BOUNDS
    lidar_max_distance: float = OFFROAD_LIDAR_MAX_DISTANCE
    car_names: tuple[str, ...] = OFFROAD_CAR_NAMES
    lidar_names: tuple[str, ...] = OFFROAD_LIDAR_NAMES

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 60.0
        self.viewer.eye = (9.5, -9.5, 7.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation
