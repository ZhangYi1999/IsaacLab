# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportOptionalSubscript=false

"""Hexagonal tactile sensor backed by Newton physics."""

from __future__ import annotations

import logging
import re
import torch
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from newton.sensors import ContactSensor as NewtonContactSensor
from newton.sensors import MatchKind
from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sim._impl.newton_manager import NewtonManager

from .hex_grid_utils import compute_hex_grid_positions, create_hex_prism_vertices
from .tactile_sensor_data import TactileSensorData

if TYPE_CHECKING:
    from .tactile_sensor_cfg import TactileSensorCfg

logger = logging.getLogger(__name__)


class TactileSensor(SensorBase):
    """A hexagonal tactile sensor array backed by the Newton physics engine.

    Each sensing unit is a hexagonal collision shape (a thin prism). Multiple units are
    arranged in a hex-grid and placed directly under a robot link prim so that they share
    that link's rigid body — no separate rigid body or fixed joint is needed.

    Because Newton's :class:`~newton.sensors.ContactSensor` can match individual collision
    *shapes* (not just rigid bodies), each hexagon reports its own net contact force.
    The resulting data tensor has shape ``(num_envs, num_hexagons, 3)``.

    **USD hierarchy created at scene setup time** (inside the designated prim_path):

    .. code-block:: text

        {prim_path}/           ← Xform prim (no rigid body API — inherits parent body)
            hex_0              ← UsdGeom.Mesh + CollisionAPI
            hex_1
            ...
            hex_N

    **Newton contact registration** (done in :meth:`_initialize_impl`, after Newton starts):

    .. code-block:: python

        NewtonManager.add_contact_sensor(
            shape_names_expr="{prim_path}/hex_.*",
            contact_partners_body_expr=...,
        )

    .. warning::
        Newton currently supports only one global contact sensor instance.  A scene that
        also contains a :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensor` will
        have only the **last-initialized** sensor active.  This limitation will be lifted in a
        future release that extends ``NewtonManager`` to handle multiple contact sensor views.

    .. note::
        The ``prim_path`` must be nested beneath the robot link that should act as the host
        rigid body (e.g. ``{ENV_REGEX_NS}/Robot/panda_hand/TactileSensor``).  The hex
        collision shapes will then be automatically associated with that link's physics body.
    """

    cfg: TactileSensorCfg

    def __init__(self, cfg: TactileSensorCfg):
        """Initialise the sensor and create hex geometry in the USD stage.

        USD prims are created here (before Newton starts) so the cloner can replicate them
        to every environment, and Newton can pick them up when it builds its model.

        Args:
            cfg: Sensor configuration.
        """
        super().__init__(cfg)
        self._data = TactileSensorData()
        # Pre-compute hex local positions (stored for data queries / visualisation)
        self._hex_positions_local = torch.from_numpy(
            compute_hex_grid_positions(cfg.num_rings, cfg.hex_radius, cfg.pointy_top)
        ).float()  # (H, 3)
        # Spawn hex collision prims into the stage at the template path
        self._spawn_hex_geometry()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> TactileSensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_hexagons(self) -> int:
        """Number of hexagonal sensing units per environment."""
        return self._hex_positions_local.shape[0]

    @property
    def hex_names(self) -> list[str]:
        """Ordered names of the hexagonal sensing units (e.g. ``["hex_0", "hex_1", ...]``)."""
        return self._hex_names

    @property
    def contact_view(self) -> NewtonContactSensor:
        """The underlying Newton contact sensor view."""
        return NewtonManager._newton_contact_sensor

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # Resolve sensor ids
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids=env_ids, env_mask=env_mask)
        # Zero out data buffers for the reset environments
        self._data._net_forces_w[env_ids] = 0.0
        if self._data._net_forces_w_history is not None:
            self._data._net_forces_w_history[env_ids] = 0.0
        if self._data._force_matrix_w is not None:
            self._data._force_matrix_w[env_ids] = 0.0
        if self._data._current_air_time is not None:
            self._data._current_air_time[env_ids] = 0.0
            self._data._current_contact_time[env_ids] = 0.0
            self._data._last_air_time[env_ids] = 0.0
            self._data._last_contact_time[env_ids] = 0.0

    def find_hexagons(self, name_regex: str) -> tuple[torch.Tensor, list[str], list[int]]:
        """Find hexagon indices whose names match a regex pattern.

        Args:
            name_regex: Regular expression to match against hex names (e.g. ``"hex_[0-5]"``).

        Returns:
            A 3-tuple ``(mask, names, indices)`` where

            * ``mask``: bool tensor of shape ``(H,)`` — True where name matched.
            * ``names``: list of matching hex names.
            * ``indices``: list of matching indices into the ``(N, H, 3)`` force tensor.
        """
        import re as _re

        pattern = _re.compile(name_regex)
        mask = torch.zeros(self.num_hexagons, dtype=torch.bool)
        names = []
        indices = []
        for i, name in enumerate(self._hex_names):
            if pattern.match(name):
                mask[i] = True
                names.append(name)
                indices.append(i)
        return mask, names, indices

    # ------------------------------------------------------------------
    # Internal – USD geometry creation
    # ------------------------------------------------------------------

    def _spawn_hex_geometry(self):
        """Create hexagonal collision shapes in the USD stage.

        This is called from ``__init__`` during scene construction, BEFORE Newton builds its
        model.  It creates a thin hexagonal prism mesh for each sensing unit and applies
        ``CollisionAPI`` so Newton treats them as collision shapes of the parent link.

        The method handles two cases for ``cfg.prim_path``:

        * **Template path** (``/World/template/.../TactileSensor/proto_asset_.*``):
          Created by the scene's cloner machinery; we strip the ``/proto_asset_.*`` suffix to
          get the actual prototype root and create prims there.  The cloner then replicates
          them to every environment.
        * **Direct path** (``/World/envs/env_.*/TactileSensor`` or similar):
          Used in single-environment or pre-cloned setups.
        """
        stage = self.stage
        prim_path = self.cfg.prim_path

        # Determine the concrete sensor root path(s) to create prims at
        template_match = re.match(r"^(.*)/[^/]+_\.\*$", prim_path)
        if template_match:
            # Template case — create prims at the prototype root
            sensor_paths = [template_match.group(1)]
        else:
            # Non-template (global or already-expanded) path — use as-is, strip trailing .*
            expanded = prim_path.replace(".*", "0")
            sensor_paths = [expanded]

        # Mesh geometry (shared across all hex instances, positions differ)
        base_pts, face_counts, face_indices = create_hex_prism_vertices(
            self.cfg.hex_radius, self.cfg.hex_height, self.cfg.pointy_top
        )
        hex_positions = compute_hex_grid_positions(
            self.cfg.num_rings, self.cfg.hex_radius, self.cfg.pointy_top
        )  # (H, 3)

        for sensor_root in sensor_paths:
            # Ensure the sensor root Xform exists (pure transform container — no RigidBodyAPI)
            if not stage.GetPrimAtPath(sensor_root):
                xform = UsdGeom.Xform.Define(stage, sensor_root)
            else:
                xform = UsdGeom.Xform(stage.GetPrimAtPath(sensor_root))

            # Apply sensor_pos / sensor_quat as a local transform
            sensor_pos = Gf.Vec3d(*self.cfg.sensor_pos)
            sensor_quat = self.cfg.sensor_quat  # (w, x, y, z)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(sensor_pos)
            xform.AddOrientOp().Set(Gf.Quatd(sensor_quat[0], sensor_quat[1], sensor_quat[2], sensor_quat[3]))

            # Create one Mesh prim per hexagon
            for i, pos in enumerate(hex_positions):
                hex_path = f"{sensor_root}/hex_{i}"
                mesh = UsdGeom.Mesh.Define(stage, hex_path)

                # Offset base vertices by this hex's centre position
                offset_pts = [
                    Gf.Vec3f(float(p[0] + pos[0]), float(p[1] + pos[1]), float(p[2] + pos[2]))
                    for p in base_pts
                ]
                mesh.GetPointsAttr().Set(offset_pts)
                mesh.GetFaceVertexCountsAttr().Set(face_counts)
                mesh.GetFaceVertexIndicesAttr().Set(face_indices)

                # Mark as a collision shape (inherits rigid body from parent link)
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

        logger.info(
            f"TactileSensor: spawned {len(hex_positions)} hexagonal collision shapes"
            f" at {sensor_paths[0]}/hex_[0..{len(hex_positions) - 1}]."
        )

    # ------------------------------------------------------------------
    # Internal – sensor lifecycle
    # ------------------------------------------------------------------

    def _initialize_impl(self):
        """Initialise Newton contact sensor and allocate data buffers.

        Called once when the Newton simulation starts (via the ``add_on_start_callback``
        registered in :class:`~isaaclab.sensors.sensor_base.SensorBase`).  At this point
        the USD stage has been fully cloned and Newton has built its model.
        """
        # Let SensorBase set up device, num_envs, timestamps, etc.
        super()._initialize_impl()

        # ------------------------------------------------------------------
        # Resolve {ENV_REGEX_NS} in filter paths using the already-expanded prim_path
        # ------------------------------------------------------------------
        # After scene cloning, self.cfg.prim_path = "/World/envs/env_.*/Robot/.../TactileSensor"
        env_ns_match = re.match(r"^(.+/env_\.\*)", self.cfg.prim_path)
        if env_ns_match:
            env_regex_ns = env_ns_match.group(1)
            if self.cfg.filter_prim_paths_expr is not None:
                self.cfg.filter_prim_paths_expr = [
                    p.format(ENV_REGEX_NS=env_regex_ns) for p in self.cfg.filter_prim_paths_expr
                ]
            if self.cfg.filter_shape_paths_expr is not None:
                self.cfg.filter_shape_paths_expr = [
                    p.format(ENV_REGEX_NS=env_regex_ns) for p in self.cfg.filter_shape_paths_expr
                ]

        # ------------------------------------------------------------------
        # Build Newton shape regex
        # ------------------------------------------------------------------
        shape_regex = f"{self.cfg.prim_path}/hex_.*"

        # Build contact-partner regexes
        if self.cfg.filter_prim_paths_expr is not None and self.cfg.filter_shape_paths_expr is not None:
            raise ValueError(
                "TactileSensorCfg: only one of 'filter_prim_paths_expr' or "
                "'filter_shape_paths_expr' may be set, not both."
            )
        contact_partners_body_regex = None
        contact_partners_shape_regex = None
        if self.cfg.filter_prim_paths_expr is not None:
            contact_partners_body_regex = "(" + "|".join(self.cfg.filter_prim_paths_expr) + ")"
        if self.cfg.filter_shape_paths_expr is not None:
            contact_partners_shape_regex = "(" + "|".join(self.cfg.filter_shape_paths_expr) + ")"

        self._generate_force_matrix = (
            self.cfg.filter_prim_paths_expr is not None or self.cfg.filter_shape_paths_expr is not None
        )

        # ------------------------------------------------------------------
        # Register with Newton
        # ------------------------------------------------------------------
        NewtonManager.add_contact_sensor(
            shape_names_expr=shape_regex,
            contact_partners_body_expr=contact_partners_body_regex,
            contact_partners_shape_expr=contact_partners_shape_regex,
        )

        self._create_buffers()

    def _create_buffers(self):
        """Allocate data buffers from Newton contact view shape."""
        contact_view = self.contact_view
        # contact_view.shape = (total_hex_instances, num_filters)
        # total_hex_instances = num_envs * num_hexagons
        total = contact_view.shape[0]
        if total % self._num_envs != 0:
            raise RuntimeError(
                f"TactileSensor: Newton contact view has {total} shape entries, which is not"
                f" divisible by num_envs={self._num_envs}. Check that the shape regex"
                f" '{self.cfg.prim_path}/hex_.*' matches exactly (num_envs * num_hexagons) shapes."
            )
        self._num_hexagons = total // self._num_envs

        # Extract hex names from Newton model (first env only — all envs are identical)
        self._hex_names = [
            NewtonManager._model.shape_key[idx].split("/")[-1]
            for idx, kind in contact_view.sensing_objs
            if kind == MatchKind.SHAPE
        ][: self._num_hexagons]

        num_filters = max(contact_view.shape[1] - 1, 0)
        logger.info(
            f"TactileSensor: {self._num_hexagons} hexagons × {self._num_envs} envs, "
            f"{num_filters} contact filters."
        )

        self._data.create_buffers(
            num_envs=self._num_envs,
            num_hexagons=self._num_hexagons,
            num_filters=num_filters,
            history_length=self.cfg.history_length,
            generate_force_matrix=self._generate_force_matrix,
            track_air_time=self.cfg.track_air_time,
            device=self._device,
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Read Newton contact data into data buffers.

        Args:
            env_ids: Environment indices to update.
        """
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # Newton net_force shape: (num_hexagons * num_envs, num_filters+1, 3)
        # Column 0 = total force; columns 1.. = per-filter forces
        raw = wp.to_torch(self.contact_view.net_force).clone()

        # Total force per hex per env
        self._data._net_forces_w[env_ids] = raw[:, 0, :].reshape(
            self._num_envs, self._num_hexagons, 3
        )[env_ids]

        # History
        if self._data._net_forces_w_history is not None:
            self._data._net_forces_w_history[env_ids, 1:] = (
                self._data._net_forces_w_history[env_ids, :-1].clone()
            )
            self._data._net_forces_w_history[env_ids, 0] = self._data._net_forces_w[env_ids]

        # Filtered force matrix
        if self._generate_force_matrix:
            num_filters = self.contact_view.shape[1] - 1
            self._data._force_matrix_w[env_ids] = raw[:, 1:, :].reshape(
                self._num_envs, self._num_hexagons, num_filters, 3
            )[env_ids]

        # Air / contact time tracking
        if self.cfg.track_air_time:
            elapsed = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            is_contact = (
                torch.norm(self._data._net_forces_w[env_ids], dim=-1) > self.cfg.force_threshold
            )
            is_first_contact = (self._data._current_air_time[env_ids] > 0) & is_contact
            is_first_detach = (self._data._current_contact_time[env_ids] > 0) & (~is_contact)

            self._data._last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data._current_air_time[env_ids] + elapsed.unsqueeze(-1),
                self._data._last_air_time[env_ids],
            )
            self._data._current_air_time[env_ids] = torch.where(
                ~is_contact,
                self._data._current_air_time[env_ids] + elapsed.unsqueeze(-1),
                torch.zeros_like(self._data._current_air_time[env_ids]),
            )
            self._data._last_contact_time[env_ids] = torch.where(
                is_first_detach,
                self._data._current_contact_time[env_ids] + elapsed.unsqueeze(-1),
                self._data._last_contact_time[env_ids],
            )
            self._data._current_contact_time[env_ids] = torch.where(
                is_contact,
                self._data._current_contact_time[env_ids] + elapsed.unsqueeze(-1),
                torch.zeros_like(self._data._current_contact_time[env_ids]),
            )

    # ------------------------------------------------------------------
    # Internal – debug visualisation (minimal stub)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_contact_visualizer"):
                self._contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self._contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_contact_visualizer"):
                self._contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._is_initialized:
            return
        # Show a marker at every hexagon that has active contact
        is_contact = torch.norm(self._data.net_forces_w, dim=-1) > self.cfg.force_threshold
        # Visualise for env 0 only to avoid clutter
        active_indices = is_contact[0].nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() > 0:
            positions = self._hex_positions_local[active_indices].to(self._device)
            self._contact_visualizer.visualize(positions.unsqueeze(0))
