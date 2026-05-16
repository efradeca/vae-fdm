"""Geometry export functions for structural form-finding results.

Exports mesh geometry and force density data to standard formats
importable by CAD and structural analysis software.
"""

from __future__ import annotations

import numpy as np


def export_dxf(
    path: str,
    xyz: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Export centerline wireframe as DXF.

    Each structural member is written as a LINE entity on layer STRUCTURE.
    Importable in AutoCAD, ETABS, SAP2000, Robot Structural, Rhino.

    Parameters
    ----------
    path : str
        Output file path (.dxf).
    xyz : ndarray (N, 3)
        Vertex positions.
    edges : ndarray (E, 2)
        Edge connectivity.
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError(
            "ezdxf is required for DXF export. "
            "Install with: pip install ezdxf"
        )

    pts = np.asarray(xyz).reshape(-1, 3)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.new("STRUCTURE", dxfattribs={"color": 7})

    for e in edges:
        i, j = int(e[0]), int(e[1])
        p1 = tuple(float(v) for v in pts[i])
        p2 = tuple(float(v) for v in pts[j])
        msp.add_line(p1, p2, dxfattribs={"layer": "STRUCTURE"})

    doc.saveas(path)


def export_obj(
    path: str,
    xyz: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Export mesh geometry as Wavefront OBJ.

    Parameters
    ----------
    path : str
        Output file path (.obj).
    xyz : ndarray (N, 3)
        Vertex positions.
    faces : ndarray (F, 3 or 4)
        Face vertex indices (0-based).
    """
    pts = np.asarray(xyz).reshape(-1, 3)
    with open(path, "w") as f:
        f.write("# VAE-FDM exported mesh\n")
        for v in pts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            indices = " ".join(str(int(i) + 1) for i in face)
            f.write(f"f {indices}\n")
