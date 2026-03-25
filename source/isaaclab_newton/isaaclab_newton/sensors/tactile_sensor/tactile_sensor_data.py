# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for the hexagonal tactile sensor."""

from __future__ import annotations

import torch


class TactileSensorData:
    """Data container for the hexagonal tactile sensor.

    This class stores per-hexagon contact force data reported by the tactile sensor.
    Each sensor instance covers a patch of hexagonal collision shapes arranged in a
    hex-grid, and each hex unit is treated as an independent sensing element.

    The convention for tensor shapes follows the rest of IsaacLab sensors:

    * ``N`` — number of environments.
    * ``H`` — number of hexagonal sensing units.
    * ``M`` — number of filtered contact-partner bodies/shapes (if a filter is set).
    * ``T`` — history length (number of past time steps stored).
    """

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def net_forces_w(self) -> torch.Tensor:
        """Net normal contact forces on each hexagon in world frame.

        Shape: ``(N, H, 3)``.
        """
        return self._net_forces_w

    @property
    def net_forces_w_history(self) -> torch.Tensor | None:
        """History of net normal contact forces in world frame.

        Shape: ``(N, T, H, 3)``. Index ``[:, 0, :, :]`` is the most recent step.

        Note:
            ``None`` when :attr:`TactileSensorCfg.history_length` is 0.
        """
        return self._net_forces_w_history

    @property
    def force_matrix_w(self) -> torch.Tensor | None:
        """Contact forces filtered by contact partners in world frame.

        Shape: ``(N, H, M, 3)``.

        Note:
            ``None`` when no ``filter_prim_paths_expr`` / ``filter_shape_paths_expr``
            is specified in the sensor configuration.
        """
        return self._force_matrix_w

    @property
    def current_contact_time(self) -> torch.Tensor | None:
        """Time (s) each hexagon has been in continuous contact.

        Shape: ``(N, H)``.

        Note:
            ``None`` when :attr:`TactileSensorCfg.track_air_time` is False.
        """
        return self._current_contact_time

    @property
    def current_air_time(self) -> torch.Tensor | None:
        """Time (s) each hexagon has been in the air since last contact.

        Shape: ``(N, H)``.

        Note:
            ``None`` when :attr:`TactileSensorCfg.track_air_time` is False.
        """
        return self._current_air_time

    @property
    def last_contact_time(self) -> torch.Tensor | None:
        """Duration (s) of the most recent completed contact event.

        Shape: ``(N, H)``.

        Note:
            ``None`` when :attr:`TactileSensorCfg.track_air_time` is False.
        """
        return self._last_contact_time

    @property
    def last_air_time(self) -> torch.Tensor | None:
        """Duration (s) of the most recent completed air (non-contact) period.

        Shape: ``(N, H)``.

        Note:
            ``None`` when :attr:`TactileSensorCfg.track_air_time` is False.
        """
        return self._last_air_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def create_buffers(
        self,
        num_envs: int,
        num_hexagons: int,
        num_filters: int,
        history_length: int,
        generate_force_matrix: bool,
        track_air_time: bool,
        device: str,
    ) -> None:
        """Allocate all data buffers on the target device.

        Args:
            num_envs: Number of parallel environments.
            num_hexagons: Number of hexagonal sensing units per environment.
            num_filters: Number of filtered contact-partner bodies/shapes.
            history_length: Number of past time-steps to store (0 = disabled).
            generate_force_matrix: Whether to allocate the filtered force matrix.
            track_air_time: Whether to allocate contact/air-time buffers.
            device: PyTorch device string (e.g. ``"cuda:0"``).
        """
        self._net_forces_w = torch.zeros((num_envs, num_hexagons, 3), dtype=torch.float32, device=device)

        if history_length > 0:
            self._net_forces_w_history = torch.zeros(
                (num_envs, history_length, num_hexagons, 3), dtype=torch.float32, device=device
            )
        else:
            self._net_forces_w_history = None

        if generate_force_matrix:
            self._force_matrix_w = torch.zeros(
                (num_envs, num_hexagons, num_filters, 3), dtype=torch.float32, device=device
            )
        else:
            self._force_matrix_w = None

        if track_air_time:
            self._last_air_time = torch.zeros((num_envs, num_hexagons), dtype=torch.float32, device=device)
            self._current_air_time = torch.zeros((num_envs, num_hexagons), dtype=torch.float32, device=device)
            self._last_contact_time = torch.zeros((num_envs, num_hexagons), dtype=torch.float32, device=device)
            self._current_contact_time = torch.zeros((num_envs, num_hexagons), dtype=torch.float32, device=device)
        else:
            self._last_air_time = None
            self._current_air_time = None
            self._last_contact_time = None
            self._current_contact_time = None
