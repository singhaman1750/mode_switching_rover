# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import ModeSwitchingRoverRoughEnvCfg


@configclass
class ModeSwitchingRoverFlatEnvCfg(ModeSwitchingRoverRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Keep flat training stable first (disable aggressive perturbations).
        # Re-enable later for robustness after base locomotion is stable.
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None
        self.events.randomize_reset_joints = None

        # Conservative reset velocities to reduce startup instability.
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        if self.__class__.__name__ == "ModeSwitchingRoverFlatEnvCfg":
            self.disable_zero_weight_rewards()
