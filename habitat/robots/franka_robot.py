# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat.robots.static_manipulator import (
    StaticManipulator,
    StaticManipulatorParams,
    RobotCameraParams,
)


class FrankaRobot(StaticManipulator):
    def _get_franka_params(self) -> StaticManipulatorParams:
        return StaticManipulatorParams(
            arm_joints=[0, 1, 2, 3, 4, 5, 6],
            gripper_joints=[9, 10],
            # fmt: off
            arm_init_params=[0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4],
            # fmt: on
            gripper_init_params=[0.04, 0.04],
            ee_offset=mn.Vector3(),  # zeroed
            # ee_link=8,
            ee_link=11,
            ee_constraint=np.array([[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]),
            gripper_closed_state=np.array(
                [
                    0.0,
                    0.0,
                ]
            ),
            gripper_open_state=np.array(
                [
                    0.04,
                    0.04,
                ]
            ),
            gripper_state_eps=0.001,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            cameras={
                "robot_arm": RobotCameraParams(
                    cam_offset_pos=mn.Vector3([0.045, 0, 0.03]),
                    cam_look_at_pos=mn.Vector3([0.045, 0, 0.03 + 1]),
                    cam_up=mn.Vector3(1, 0, 0),
                    attached_link_id=8,
                ),
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0.2 + 0.615, 0.4, 0.0),
                    cam_look_at_pos=mn.Vector3(0, 0, 0),
                    attached_link_id=-1,
                ),
                "robot_third": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(1.0 + 0.615, 0.8, 1.0),
                    cam_look_at_pos=mn.Vector3(0, 0.5, 0.0),
                    attached_link_id=-1,
                ),
            },
        )

    def __init__(self, *args, **kwargs):
        kwargs["params"] = self._get_franka_params()
        super().__init__(*args, **kwargs)
