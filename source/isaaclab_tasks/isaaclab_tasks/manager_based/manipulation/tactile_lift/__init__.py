# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile box-lifting manipulation task.

Two variants are registered:

* ``Isaac-TactileLift-Franka-NoTactile-v0`` — baseline without tactile sensor observations.
* ``Isaac-TactileLift-Franka-Tactile-v0`` — with per-hexagon contact force observations and
  a force-efficiency reward term.

Comparison of these two variants demonstrates whether tactile feedback helps the agent find
a more force-efficient grasp.
"""
