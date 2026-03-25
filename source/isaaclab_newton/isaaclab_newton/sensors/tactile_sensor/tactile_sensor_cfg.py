# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration class for the hexagonal tactile sensor."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.utils import configclass


@configclass
class TactileSensorCfg(SensorBaseCfg):
    """Configuration for the hexagonal tactile sensor.

    The tactile sensor consists of a grid of hexagonal collision shapes (sensing units)
    arranged on a rigid body that is attached to a robot link via a fixed joint.  Each
    hexagon is an independent contact shape, and the Newton physics engine reports the
    net contact force on each shape individually.

    The sensor body is a separate :class:`~pxr.UsdPhysics.RigidBody` connected to the
    robot link specified by :attr:`attach_prim_path` through a fixed joint.  This keeps
    the sensor module self-contained and easy to attach or detach.

    .. note::
        Because Newton currently exposes a single global contact sensor instance
        (:attr:`~isaaclab.sim._impl.newton_manager.NewtonManager._newton_contact_sensor`),
        only **one** :class:`TactileSensor` (or
        :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensor`) may be active in a
        scene at a time.  If you need both, a future extension to
        :class:`~isaaclab.sim._impl.newton_manager.NewtonManager` would be required.

    Example:
        .. code-block:: python

            tactile_cfg = TactileSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/TactileSensor",
                attach_prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                num_rings=2,          # 19 hexagons
                hex_radius=0.006,     # 6 mm circumradius
                hex_height=0.002,     # 2 mm thickness
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
            )
    """

    # Import here to avoid circular imports at module load time.
    from .tactile_sensor import TactileSensor

    class_type: type = TactileSensor

    # ------------------------------------------------------------------
    # Hex-grid geometry
    # ------------------------------------------------------------------

    num_rings: int = 2
    """Number of rings around the central hexagon.

    * 0 → 1 hex (single unit)
    * 1 → 7 hexes
    * 2 → 19 hexes (default)
    * N → ``1 + 3 * N * (N + 1)`` hexes
    """

    hex_radius: float = 0.006
    """Circumradius of each hexagon (distance from centre to vertex) in metres."""

    hex_height: float = 0.002
    """Thickness (height along the sensor normal axis) of each hexagonal prism in metres."""

    pointy_top: bool = False
    """Hexagon orientation.  ``False`` (default) = flat-top edge; ``True`` = pointy vertex on top."""

    # ------------------------------------------------------------------
    # Attachment to robot link
    # ------------------------------------------------------------------

    attach_prim_path: str = MISSING
    """USD prim path of the robot link to which the sensor body is fixed-jointed.

    Supports the ``{ENV_REGEX_NS}`` substitution, e.g.
    ``"{ENV_REGEX_NS}/Robot/panda_hand"``.
    """

    sensor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation offset of the sensor body relative to :attr:`attach_prim_path` (metres)."""

    sensor_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation of the sensor body relative to :attr:`attach_prim_path` as a
    quaternion ``(w, x, y, z)``."""

    # ------------------------------------------------------------------
    # Contact filtering
    # ------------------------------------------------------------------

    filter_prim_paths_expr: list[str] | None = None
    """Regex expressions for bodies that should act as contact partners.

    When set, :attr:`~TactileSensorData.force_matrix_w` contains per-partner forces in
    addition to the net force in :attr:`~TactileSensorData.net_forces_w`.

    Supports ``{ENV_REGEX_NS}`` substitution, e.g. ``["{ENV_REGEX_NS}/Box"]``.
    """

    filter_shape_paths_expr: list[str] | None = None
    """Regex expressions for collision *shapes* (instead of bodies) to use as contact
    partners.  Mutually exclusive with :attr:`filter_prim_paths_expr`."""

    force_threshold: float = 1.0
    """Minimum contact-force norm (N) to consider a hexagon as in contact.

    Only used for air/contact time tracking (:attr:`track_air_time`).
    """

    track_air_time: bool = False
    """Whether to track per-hexagon contact and air durations."""

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(
        prim_path="/Visuals/TactileSensor"
    )
    """Marker configuration used for debug visualisation.

    Each marker is placed at a hexagon centre when that hexagon detects contact.
    """
