#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import magnum as mn
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.registry import registry
from habitat.robots.franka_robot import FrankaRobot
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSim,
    HabitatSimSemanticSensor,
)
from habitat_extensions import ASSET_DIR
from habitat_sim.physics import (
    JointMotorSettings,
    ManagedBulletArticulatedObject,
    ManagedBulletRigidObject,
    MotionType,
)
from habitat_sim.utils.common import orthonormalize_rotation_shear

from habitat_extensions.utils import coll_utils


def make_render_only(obj):
    obj.motion_type = MotionType.KINEMATIC
    obj.collidable = False


@registry.register_simulator(name="PickCubeSim-v0")
class PickCubeSim(HabitatSim):
    # ROBOT_URDF_PATH = "data/robots/franka_panda/panda_arm_hand.urdf"

    def __init__(self, config: Config):
        super().__init__(config)

        # NOTE(jigu): The first episode is used to initialized the simulator
        # When `habitat.Env` is initialized.
        # NOTE(jigu): DO NOT set `_current_scene` to None.
        self._prev_scene_id = None
        # self._prev_scene_dataset = config.SCENE_DATASET
        self._prev_scene_dataset = ""

        self._initialize_templates()

        # objects
        self.rigid_objs: Dict[str, ManagedBulletRigidObject] = OrderedDict()
        self.art_objs: Dict[
            str, ManagedBulletArticulatedObject
        ] = OrderedDict()
        self.viz_objs: Dict[str, ManagedBulletRigidObject] = OrderedDict()

        # robot
        # self.robot = FrankaRobot(urdf_path=self.ROBOT_URDF_PATH, sim=self)
        self.robot = FrankaRobot(
            urdf_path=osp.join(ASSET_DIR, "hab_panda.urdf"), sim=self
        )
        self.verbose = False

    def _initialize_templates(self):
        obj_attr_mgr = self.get_object_template_manager()
        obj_attr_mgr.load_configs(ASSET_DIR)
        # obj_attr_mgr.load_configs("/home/jiayuan/projects/github/habitat-lab/data/objects/ycb/configs")
        # print(obj_attr_mgr.get_template_handles())

    @property
    def timestep(self):
        return self.habitat_config.CONTROL_FREQ / self.habitat_config.SIM_FREQ

    def reconfigure(self, habitat_config: Config):
        """Called before sim.reset() in `habitat.Env`."""

        # NOTE(jigu): DO NOT use self._current_scene to judge
        is_same_scene = habitat_config.SCENE == self._prev_scene_id
        if self.verbose:
            print("is_same_scene", is_same_scene)

        is_same_scene_dataset = (
            habitat_config.SCENE_DATASET == self._prev_scene_dataset
        )

        # The simulator backend will be reconfigured.
        # Assets are invalid after a new scene is configured.
        # Note that ReplicaCAD articulated objects are managed by the backend.
        super().reconfigure(habitat_config)
        self._prev_scene_id = habitat_config.SCENE
        self._prev_scene_dataset = habitat_config.SCENE_DATASET

        if not is_same_scene:
            self.art_objs = OrderedDict()
            self.rigid_objs = OrderedDict()

        if not is_same_scene_dataset:
            self._initialize_templates()

        # Called before new assets are added
        if not is_same_scene:
            self.robot.reconfigure()
            for node in self.robot.sim_obj.visual_scene_nodes:
                node.semantic_id = 200

            # from habitat_extensions.utils import art_utils
            # # print(art_utils.get_joint_motors_info(self.robot.sim_obj))
            # print(art_utils.get_links_info(self.robot.sim_obj))

        if not is_same_scene:
            self._add_articulated_objects()
            # self._initialize_articulated_objects()

        if not is_same_scene:
            self._remove_rigid_objects()
            self._add_rigid_objects()

        assert len(self.viz_objs) == 0, self.viz_objs
        self.viz_objs = OrderedDict()

        if self.habitat_config.get("AUTO_SLEEP", False):
            self.sleep_all_objects()

    def _add_rigid_object(
        self,
        name: str,
        template_name: str,
        position: mn.Vector3,
        rotation: Optional[mn.Quaternion] = None,
        scale=(1, 1, 1),
        static=False,
    ):
        obj_attr_mgr = self.get_object_template_manager()
        rigid_obj_mgr = self.get_rigid_object_manager()

        # Register a new template
        template = obj_attr_mgr.get_template_by_handle(
            obj_attr_mgr.get_template_handles(template_name)[0]
        )
        template.scale = mn.Vector3(scale)
        template_id = obj_attr_mgr.register_template(template, name)

        obj = rigid_obj_mgr.add_object_by_template_id(template_id)
        if static:
            obj.motion_type = MotionType.STATIC
        else:
            obj.motion_type = MotionType.DYNAMIC
        obj.translation = mn.Vector3(position)
        if rotation is None:
            rotation = mn.Quaternion.identity_init()
        obj.rotation = rotation
        return obj

    def _add_rigid_objects(self):
        # obj_templates_mgr = self.get_object_template_manager()
        # rigid_obj_mgr = self.get_rigid_object_manager()

        self.ground = self._add_rigid_object(
            "ground",
            "cubeSolid",
            [0, -0.01, 0],
            scale=[10, 0.01, 10],
            static=True,
        )
        self.ground.semantic_id=0

        self.cube = self._add_rigid_object(
            "cube", "transform_box", [0, 0.02, 0], scale=[0.01] * 3
        )
        self.cube.semantic_id=100

        # self.obj1 = self._add_rigid_object("haha", "002_master_chef_can", [0, 0.5, 0])
        # self.obj2 = self._add_rigid_object("haha", "072-a_toy_airplane", [0, 0.5, 0])

        self.rigid_objs["ground"] = self.ground
        self.rigid_objs["cube"] = self.cube

    def _remove_rigid_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for handle, obj in self.rigid_objs.items():
            assert obj.is_alive, handle
            if self.verbose:
                print(
                    "Remove a rigid object",
                    obj.handle,
                    obj.object_id,
                    obj.is_alive,
                )
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
        self.rigid_objs = OrderedDict()

    def _add_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            if handle == self.robot.sim_obj.handle:  # ignore robot
                continue
            self.art_objs[handle] = art_obj_mgr.get_object_by_handle(handle)

    def _remove_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for art_obj in self.art_objs.values():
            assert art_obj.is_alive
            if self.verbose:
                print(
                    "Remove an articulated object",
                    art_obj.handle,
                    art_obj.object_id,
                    art_obj.is_alive,
                )
            art_obj_mgr.remove_object_by_id(art_obj.object_id)
        self.art_objs = OrderedDict()

    def print_articulated_objects(self):
        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            print(handle, art_obj, art_obj.object_id)

    def sleep_all_objects(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for handle in rigid_obj_mgr.get_object_handles():
            obj = rigid_obj_mgr.get_object_by_handle(handle)
            obj.awake = False

        art_obj_mgr = self.get_articulated_object_manager()
        for handle in art_obj_mgr.get_object_handles():
            art_obj = art_obj_mgr.get_object_by_handle(handle)
            art_obj.awake = False

    def reset(self):
        # The agent and sensors are reset.
        super().reset()

        # Reset the robot
        self.robot.reset()

        # Place the robot
        self.robot.sim_obj.translation = mn.Vector3(-0.615, 0, 0)
        self.robot.sim_obj.rotation = mn.Quaternion.identity_init()

        # Place the cube
        x, z = np.random.uniform(-0.1, 0.1, [2])
        ori = np.random.uniform(0, 2 * np.pi)
        self.cube.translation = mn.Vector3(x, 0.02, z)
        self.cube.rotation = mn.Quaternion(mn.Vector3(0, 1, 0), mn.Rad(ori))

        # Sample a goal position far enough from the object
        obj_pos = np.float32(self.cube.translation)
        for i in range(100):
            x, z = np.random.uniform(-0.1, 0.1, [2])
            y = np.random.uniform(0, 0.5) + obj_pos[1]
            goal_pos = np.hstack([x, y, z])
            if np.linalg.norm(goal_pos - obj_pos) > 0.05:
                break
        self.goal_pos = np.float32(goal_pos)

        return self.get_observations()

    def get_observations(self):
        self.robot.update()
        self._prev_sim_obs = self.get_sensor_observations()
        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def internal_step(self, dt=None):
        """Internal simulation step."""
        if dt is None:
            dt = 1.0 / self.habitat_config.SIM_FREQ
        self.step_world(dt)
        # self.robot.sim_obj.awake = True
        # self.robot.arm_motor_forces = np.random.normal(self.robot.arm_motor_forces, 0.001)

    def internal_step_by_time(self, seconds):
        steps = int(seconds * self.habitat_config.SIM_FREQ)
        for _ in range(steps):
            self.internal_step()

    def step(self, action: Optional[int] = None):
        # virtual agent's action, only for compatibility.
        if action is not None:
            self._default_agent.act(action)

        # step physics
        for _ in range(self.habitat_config.CONTROL_FREQ):
            self.internal_step()

        return self.get_observations()

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    def update_camera(self, sensor_name, cam2world: mn.Matrix4):
        agent_inv_T = self._default_agent.scene_node.transformation.inverted()
        sensor = self._sensors[sensor_name]._sensor_object
        sensor.node.transformation = orthonormalize_rotation_shear(
            agent_inv_T @ cam2world
        )

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    def _remove_viz_objs(self):
        rigid_obj_mgr = self.get_rigid_object_manager()
        for name, obj in self.viz_objs.items():
            assert obj.is_alive, name
            if self.verbose:
                print(
                    "Remove a vis object",
                    name,
                    obj.handle,
                    obj.object_id,
                    obj.is_alive,
                )
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
        self.viz_objs = OrderedDict()

    def add_viz_obj(
        self,
        position: mn.Vector3,
        rotation: Optional[mn.Quaternion] = None,
        scale=mn.Vector3(1, 1, 1),
        template_name="coord_frame",
    ):
        obj_attr_mgr = self.get_object_template_manager()
        rigid_obj_mgr = self.get_rigid_object_manager()

        # register a new template for visualization
        template = obj_attr_mgr.get_template_by_handle(
            obj_attr_mgr.get_template_handles(template_name)[0]
        )
        template.scale = scale
        template_id = obj_attr_mgr.register_template(
            template, f"viz_{template_name}"
        )

        viz_obj = rigid_obj_mgr.add_object_by_template_id(template_id)
        make_render_only(viz_obj)
        viz_obj.translation = position
        if rotation is not None:
            viz_obj.rotation = rotation
        return viz_obj

    def visualize_frame(self, name, T: mn.Matrix4, scale=1.0):
        assert name not in self.viz_objs, name
        self.viz_objs[name] = self.add_viz_obj(
            position=T.translation,
            rotation=mn.Quaternion.from_matrix(T.rotation()),
            scale=mn.Vector3(scale),
            template_name="coord_frame",
        )

    def render(self, mode: str):
        """Render with additional debug info.
        Users can add more visualization to viz_objs before calling sim.render().
        """
        # self.visualize_frame("ee_frame", self.robot.ee_transform, scale=0.15)
        rendered_frame = super().render(mode=mode)
        # Remove visualization in case polluate observations
        self._remove_viz_objs()
        return rendered_frame

    def check_grasp(self):
        robot_id = self.robot.sim_obj.object_id
        contact_points = self.get_physics_contact_points()

        contact_infos = coll_utils.get_contact_infos(
            contact_points, robot_id, link_ids=self.robot.params.gripper_joints
        )
        contact_infos = [
            c for c in contact_infos if c["object_id"] == self.cube.object_id
        ]

        # contact_infos_L = [c for c in contact_infos if c["link_id"] == self.robot.params.gripper_joints[0]]
        # contact_infos_R = [c for c in contact_infos if c["link_id"] == self.robot.params.gripper_joints[1]]

        if len(contact_infos) > 0:
            max_force = max(x["normal_force"] for x in contact_infos)
        else:
            max_force = 0.0

        return max_force > 1e-3


def main():
    import argparse

    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.utils.visualizations.utils import observations_to_image
    from habitat_extensions.utils.viewer import OpenCVViewer

    parser = argparse.ArgumentParser()
    parser.add_argument("--render_uuid", type=str, default="robot_third_rgb")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    # Configuration
    # ---------------------------------------------------------------------------- #
    config = get_config()
    config.defrost()

    config.SIMULATOR.TYPE = "PickCubeSim-v0"
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True

    config.SIMULATOR.AGENT_0.SENSORS = [
        # "RGB_SENSOR",
        "HEAD_RGB_SENSOR",
        "HEAD_DEPTH_SENSOR",
        # "ARM_RGB_SENSOR",
        # "ARM_DEPTH_SENSOR",
        "THIRD_RGB_SENSOR",
    ]
    config.SIMULATOR.RGB_SENSOR.POSITION = [1.5, 0.5, 0.0]
    config.SIMULATOR.RGB_SENSOR.ORIENTATION = [0.0, 1.57, 0.0]
    # config.SIMULATOR.HEAD_DEPTH_SENSOR.NORMALIZE_DEPTH = False
    # config.SIMULATOR.ARM_DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = 512
    config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = 512

    # specific keys
    config.SIMULATOR.VERBOSE = True
    config.SIMULATOR.SIM_FREQ = 120
    config.SIMULATOR.CONTROL_FREQ = 4

    # episode
    config.SIMULATOR["EPISODE"] = {}
    config.SIMULATOR.SCENE = "none"
    # config.SIMULATOR.SCENE_DATASET = ""

    config.merge_from_list(args.opts)
    config.freeze()
    # print(config)

    # ---------------------------------------------------------------------------- #
    # Main
    # ---------------------------------------------------------------------------- #
    with make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR) as sim:
        sim: PickCubeSim
        sim.reconfigure(sim.habitat_config)
        obs = sim.reset()
        viewer = OpenCVViewer(config.SIMULATOR.TYPE)

        # sanity check
        print("obs keys", obs.keys())

        step = 0
        while True:
            img_to_display = observations_to_image(obs, {})
            key = viewer.imshow(img_to_display)
            obs = sim.step(None)
            step += 1
            print(step)


if __name__ == "__main__":
    main()
