
from .env import PickCubeEnv

"""
from gym.core import Wrapper, ObservationWrapper
import gym, numpy as np


class MyWrapper(ObservationWrapper):
    def observation(self, observation):
        return observation
    
    def render(self, *args, **kwargs):
        return np.zeros([128, 128, 3], dtype=np.uint8)
    

def build_habitat():
    env = gym.make("HabitatPickCube-v0", obs_mode="rgb_mode")
    env = MyWrapper(env)
    return env


def register_habtitat():
    gym.envs.register(
        id='HabitatPickCube-v1',
        entry_point='habitat_extensions.pick_cube:build_habitat',
        max_episode_steps=1000,
    )
register_habtitat()
"""