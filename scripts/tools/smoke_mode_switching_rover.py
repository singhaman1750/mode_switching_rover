# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ModeSwitchingRover with Isaac Sim UI.

Run this file directly from VS Code (Run Python File) or terminal.
By default it launches Isaac Sim with UI and runs 300 random-action steps.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Smoke test ModeSwitchingRover task."
)
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Rough-ModeSwitchingRover-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--spawn_only",
    action="store_true",
    default=False,
    help="Only spawn and keep UI open, without stepping random actions.",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.2,
    help="Scale factor for random actions in [-1, 1].",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

# AppLauncher adds --headless and device args.
# Keep UI by not passing --headless.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    import time

    import gymnasium as gym
    import torch

    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    print(f"[INFO] Running smoke test: {args_cli.task}")
    num_actions = env.action_space.shape[-1]
    print(
        f"[INFO] num_envs={env.unwrapped.num_envs}, "
        f"num_actions={num_actions}"
    )
    print(
        "[INFO] Spawned rover at /World/envs/env_0/Robot. "
        "Use Stage panel to verify articulation and bodies."
    )

    if args_cli.spawn_only:
        print(
            "[INFO] Spawn-only mode: keeping UI alive. "
            "Close window to exit."
        )
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(
                    torch.zeros(
                        env.action_space.shape,
                        device=env.unwrapped.device,
                    )
                )
            time.sleep(0.01)
        env.close()
        return

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.steps:
        with torch.inference_mode():
            actions = args_cli.action_scale * (2 * torch.rand(
                env.action_space.shape,
                device=env.unwrapped.device,
            ) - 1)
            env.step(actions)
        if step_count % 50 == 0:
            print(f"[INFO] step={step_count}/{args_cli.steps}")
        step_count += 1

    print("[INFO] Smoke test completed.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
