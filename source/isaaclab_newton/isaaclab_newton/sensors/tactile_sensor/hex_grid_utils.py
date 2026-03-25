# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for hexagonal grid layouts and hexagonal prism mesh generation."""

from __future__ import annotations

import numpy as np


def compute_hex_grid_positions(num_rings: int, hex_radius: float, pointy_top: bool = False) -> np.ndarray:
    """Compute the center positions of hexagons in a hex-grid layout.

    Uses axial coordinates (q, r) to enumerate all hexagons within ``num_rings`` rings
    of the central hexagon, then converts to Cartesian (x, y) coordinates. The z
    coordinate is always 0 (the caller is responsible for applying any offset along the
    sensor normal axis).

    The total number of hexagons is ``1 + 3 * num_rings * (num_rings + 1)``:
    * num_rings = 0 → 1 hexagon
    * num_rings = 1 → 7 hexagons
    * num_rings = 2 → 19 hexagons

    Args:
        num_rings: Number of rings around the central hexagon (0 = single hex).
        hex_radius: Circumradius of each hexagon, i.e. the distance from the center to
            a vertex (in metres).
        pointy_top: If False (default), hexagons have a flat top/bottom edge.
            If True, hexagons have a pointy top/bottom vertex.

    Returns:
        Array of shape (N, 3) containing the (x, y, 0) centre positions of each hex
        in the local sensor frame.
    """
    positions = []
    for q in range(-num_rings, num_rings + 1):
        r_min = max(-num_rings, -q - num_rings)
        r_max = min(num_rings, -q + num_rings)
        for r in range(r_min, r_max + 1):
            if pointy_top:
                x = hex_radius * np.sqrt(3) * (q + r / 2.0)
                y = hex_radius * 1.5 * r
            else:
                x = hex_radius * 1.5 * q
                y = hex_radius * np.sqrt(3) * (r + q / 2.0)
            positions.append([x, y, 0.0])
    return np.array(positions, dtype=np.float32)


def create_hex_prism_vertices(hex_radius: float, hex_height: float, pointy_top: bool = False) -> tuple[np.ndarray, list[int], list[int]]:
    """Generate vertices and face topology for a hexagonal prism centered at the origin.

    The prism extends ``hex_height / 2`` above and below the XY plane, and has a
    regular hexagonal cross-section with the given circumradius.

    Args:
        hex_radius: Circumradius of the hexagonal cross-section (m).
        hex_height: Total height (thickness) of the prism along Z (m).
        pointy_top: Orientation of the hexagon. False = flat-top, True = pointy-top.

    Returns:
        A 3-tuple ``(points, face_vertex_counts, face_vertex_indices)`` suitable for
        populating a ``UsdGeom.Mesh``:

        * ``points``: (12, 3) float32 array — 6 bottom vertices + 6 top vertices.
        * ``face_vertex_counts``: list of ints, one entry per face (value = 4 for quads,
          3 for triangles). Total 8 faces: 6 side quads + 2 hexagonal caps.
        * ``face_vertex_indices``: flat list of vertex indices for every face.
    """
    angle_offset = 0.0 if pointy_top else np.pi / 6.0
    angles = [angle_offset + i * np.pi / 3.0 for i in range(6)]
    bottom_verts = np.array([[hex_radius * np.cos(a), hex_radius * np.sin(a), -hex_height / 2.0] for a in angles], dtype=np.float32)
    top_verts = bottom_verts.copy()
    top_verts[:, 2] = hex_height / 2.0
    points = np.vstack([bottom_verts, top_verts])  # indices 0-5 = bottom, 6-11 = top

    face_vertex_counts = []
    face_vertex_indices = []

    # 6 side quad faces (each quad: bottom[i], bottom[i+1], top[i+1], top[i])
    for i in range(6):
        j = (i + 1) % 6
        face_vertex_counts.append(4)
        face_vertex_indices.extend([i, j, j + 6, i + 6])

    # Bottom cap (triangle fan from vertex 0)
    for i in range(1, 5):
        face_vertex_counts.append(3)
        face_vertex_indices.extend([0, i + 1, i])

    # Top cap (triangle fan from vertex 6)
    for i in range(1, 5):
        face_vertex_counts.append(3)
        face_vertex_indices.extend([6, 6 + i, 6 + i + 1])

    return points, face_vertex_counts, face_vertex_indices
