# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatible wrapper for the renamed exploration rollout script."""

from pathlib import Path
import runpy
import sys

print(
    "[WARN]: scripts/random_agent.py has been renamed to scripts/exploration_agent.py. "
    "Please use the new script name in future commands.",
    file=sys.stderr,
)

runpy.run_path(str(Path(__file__).with_name("exploration_agent.py")), run_name="__main__")
