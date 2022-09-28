import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces
import magnum as mn

from habitat import logger
from habitat.config import Config
from habitat.config.default import get_config
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.sims import make_sim
from habitat_extensions.pick_cube.sim import PickCubeSim
from habitat_extensions.utils.registration import register_gym_env


def Quaternion2list(x: mn.Quaternion):
    return list(x.vector) + [x.scalar]


def vectorize_pose(T: mn.Matrix4):
    p = np.float32(T.translation)
    q = mn.Quaternion.from_matrix(T.rotation())
    return np.hstack([p, q.scalar, np.float32(q.vector)])


def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issctype(dtype, bool):
        return 0, 1
    else:
        raise TypeError(dtype)


def convert_observation_to_space(observation, prefix=""):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        space = spaces.Dict(
            {
                k: convert_observation_to_space(v, prefix + "/" + k)
                for k, v in observation.items()
            }
        )
    elif isinstance(observation, np.ndarray):
        shape = observation.shape
        dtype = observation.dtype
        dtype_min, dtype_max = get_dtype_bounds(dtype)
        low = np.full(shape, dtype_min)
        high = np.full(shape, dtype_max)
        space = spaces.Box(low, high, dtype=dtype)
    elif isinstance(observation, float):
        logger.warning(f"The observation ({prefix}) is a float")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


@register_gym_env("HabitatPickCube-v0", 200)
class PickCubeEnv(gym.Env):
    _sim: PickCubeSim

    def __init__(self, obs_mode="rgbd"):
        self.obs_mode = obs_mode
        self._config = self.get_config(obs_mode=obs_mode)
        self._viewer = None

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )

        obs = self.reset()
        self.observation_space = convert_observation_to_space(obs)
        self.action_space = spaces.Box(-1, 1, [8], dtype=np.float32)

    @classmethod
    def get_config(cls, obs_mode="rgbd"):
        config = get_config()
        config.defrost()

        config.SIMULATOR.TYPE = "PickCubeSim-v0"
        config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
        config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True

        # Scene
        config.SIMULATOR.SCENE = "none"

        # Sensors
        config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = 512
        config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = 512
        config.SIMULATOR.THIRD_RGB_SENSOR.HFOV = 60
        config.SIMULATOR.HEAD_RGB_SENSOR.WIDTH = 128
        config.SIMULATOR.HEAD_RGB_SENSOR.HEIGHT = 128
        config.SIMULATOR.HEAD_DEPTH_SENSOR.WIDTH = 128
        config.SIMULATOR.HEAD_DEPTH_SENSOR.HEIGHT = 128
        config.SIMULATOR.HEAD_SEMANTIC_SENSOR.WIDTH = 128
        config.SIMULATOR.HEAD_SEMANTIC_SENSOR.HEIGHT = 128
        config.SIMULATOR.ARM_RGB_SENSOR.WIDTH = 128
        config.SIMULATOR.ARM_RGB_SENSOR.HEIGHT = 128
        config.SIMULATOR.ARM_DEPTH_SENSOR.WIDTH = 128
        config.SIMULATOR.ARM_DEPTH_SENSOR.HEIGHT = 128
        config.SIMULATOR.ARM_SEMANTIC_SENSOR.WIDTH = 128
        config.SIMULATOR.ARM_SEMANTIC_SENSOR.HEIGHT = 128
        if obs_mode == "state":
            config.SIMULATOR.AGENT_0.SENSORS = []
        else:
            config.SIMULATOR.AGENT_0.SENSORS = [
                "HEAD_RGB_SENSOR",
                "HEAD_DEPTH_SENSOR",
                "ARM_RGB_SENSOR",
                "ARM_DEPTH_SENSOR",
                "THIRD_RGB_SENSOR",
                "HEAD_SEMANTIC_SENSOR",
                "ARM_SEMANTIC_SENSOR",
            ]

        # Simulator custom
        # It does not affect underlying timestep!
        config.SIMULATOR.SIM_FREQ = 120
        config.SIMULATOR.CONTROL_FREQ = 4

        config.freeze()
        return config

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._sim.seed(seed)

    def reset(self):
        self._sim.reconfigure(self._config.SIMULATOR)
        observations = self._sim.reset()
        observations.update(self.get_obs_state())
        return observations

    def get_obs_state(self):
        return dict(agent=self._get_obs_agent(), extra=self._get_obs_extra())

    def _get_obs_agent(self):
        qpos = np.float32(self._sim.robot.sim_obj.joint_positions)
        qvel = np.float32(self._sim.robot.sim_obj.joint_velocities)
        return dict(qpos=qpos, qvel=qvel)

    def _get_obs_extra(self):
        tcp_pose = vectorize_pose(self._sim.robot.ee_transform)
        return dict(tcp_pose=tcp_pose, goal_pos=self._sim.goal_pos)

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        observations = self.step_action(*args, **kwargs)
        info = self.get_info(obs=observations)
        reward = self.get_reward(obs=observations, info=info)
        done = self.get_done(obs=observations, info=info)

        return observations, reward, done, info

    def step_action(
        self, action: Union[int, str, Dict[str, Any]]
    ) -> Observations:
        # NOTE(jigu): hardcode
        arm_action, gripper_action = action[0:7], action[7]
        arm_action = np.clip(arm_action, -1, 1) * 0.1
        gripper_action = np.clip(gripper_action, -1, 1) * 0.04
        self._sim.robot.arm_motor_pos = (
            self._sim.robot.arm_joint_pos + arm_action
        )
        self._sim.robot.gripper_motor_pos = [gripper_action] * 2

        observations = self._sim.step()
        observations.update(self.get_obs_state())
        return observations

    def get_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_pose = self._sim.robot.ee_transform
        tcp_pos = np.float32(tcp_pose.translation)
        obj_pos = np.float32(self._sim.cube.translation)
        goal_pos = self._sim.goal_pos

        tcp_to_obj_pos = obj_pos - tcp_pos
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        is_grasped = self._sim.check_grasp()
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(goal_pos - obj_pos)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

        return reward

    def get_done(self, info, **kwargs):
        return info["success"]

    def check_obj_placed(self):
        obj_pos = np.float32(self._sim.cube.translation)
        goal_pos = self._sim.goal_pos
        return np.linalg.norm(goal_pos - obj_pos) <= 0.025

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self._sim.robot.sim_obj.joint_positions[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )

    def get_info(self, **kwargs):
        return self.evaluate()

    def render(self, mode="human") -> np.ndarray:
        if mode == "human":
            from habitat.utils.visualizations.utils import (
                observations_to_image,
            )
            from habitat_extensions.utils.viewer import OpenCVViewer

            obs = self._sim.get_observations()
            obs["robot_third_rgb"] = self._sim.render("robot_third_rgb")
            img = observations_to_image(obs, {})

            if self._viewer is None:
                self._viewer = OpenCVViewer("PickCube-v0")
            return self._viewer.imshow(img)
        else:
            return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()
        if self._viewer is not None:
            self._viewer.close()


def main():
    # env = PickCubeEnv()
    env = gym.make("HabitatPickCube-v0")

    env.seed(0)  # specify a seed for randomness
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() * 0
        obs, reward, done, info = env.step(action)
        print(obs["agent"])
        print(obs["extra"])
        # print(reward, info)
        key = env.render()
        if key == "r":
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
