# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from robot_lab.assets.mode_switching_rover import MODE_SWITCHING_ROVER_CFG  # isort: skip


@configclass
class ModeSwitchingRoverActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class ModeSwitchingRoverRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class ModeSwitchingRoverRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ModeSwitchingRoverActionsCfg = ModeSwitchingRoverActionsCfg()
    rewards: ModeSwitchingRoverRewardsCfg = ModeSwitchingRoverRewardsCfg()

    base_link_name = "base"
    foot_link_name = "Wheel.*"

    # fmt: off
    leg_joint_names = [
        "FRA", "FLA", "BRA", "BLA",
        "FRB", "FLB", "BRB", "BLB",
    ]
    wheel_joint_names = ["FRP", "FLP", "BRP", "BLP"]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = MODE_SWITCHING_ROVER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Rover"
        )

        # Base-link dependent scanners are disabled until body names are verified.
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.contact_forces = None
        self.terminations.illegal_contact = None

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # NOTE:
        # joint_pos_rel_without_wheel() masks by wheel joint IDs in articulation
        # index space. Therefore, asset_cfg must cover full articulation joints.
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = ".*"
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = ".*"
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {
            ".*A$": 0.25,
            ".*B$": 0.25,
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Rewards------------------------------
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.base_height_l2.weight = 0
        self.rewards.body_lin_acc_l2.weight = 0

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -1.0e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR(A|B)$", "BL(A|B)$"],
            ["FL(A|B)$", "BR(A|B)$"],
        ]

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Disable foot/contact-dependent terms until foot body names are
        # verified.
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_gait.weight = 0

        if self.__class__.__name__ == "ModeSwitchingRoverRoughEnvCfg":
            self.disable_zero_weight_rewards()
