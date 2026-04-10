# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for mode-switching rover assets."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

MODE_SWITCHING_ROVER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/"
            "mode_switching_rover/mode_switching_rover3.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        rot=(0.0, 1.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "actuatorA": DCMotorCfg(
            joint_names_expr=["FRA", "FLA", "BRA", "BLA"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "actuatorB": DCMotorCfg(
            joint_names_expr=[
                "FRB",
                "FLB",
                "BRB",
                "BLB",
            ],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["FRP", "FLP", "BRP", "BLP"],
            effort_limit_sim=20.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "FLD",
                "FRD",
                "BLD",
                "BRD",
                "FLC",  # "FLF",
                "FRC",  # "FRF",
                "BLC",  # "BLF",
                "BRC",  # "BRF",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
"""Configuration for mode-switching rover loaded from USD.

This is an initial, conservative setup. For best performance, split the
actuator group by joint type (wheel/steering/suspension) once joint names are
confirmed from the USD articulation.
"""
