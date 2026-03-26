"""
Convert text to a mesh to label the bricks of a masonry shell.

This script uses the FreeType library to convert text to a mesh.

FreeType high-level python API - Copyright 2011 Nicolas P. Rougier.
Distributed under the terms of the new BSD license.
"""

import numpy as np
from matplotlib.path import Path

from freetype import Face

from compas.geometry import bounding_box
from compas.geometry import Box
from compas.geometry import Translation
from compas.datastructures import Mesh
from compas.datastructures import meshes_join

from compas_cgal.triangulation import constrained_delaunay_triangulation


def char_2_mesh(char, filepath="Vera.ttf"):
    """
    Convert a single character to a mesh.

    Parameters
    ----------
    char: `str`
        The character to convert.
    filepath: `str`, optional
        The path to the font file.

    Returns
    -------
    mesh: `compas.datastructures.Mesh`
        The mesh representing the character.
    """
    face = Face(filepath)
    face.set_char_size(48*64)
    face.load_char(char)
    slot = face.glyph

    outline = slot.outline
    points = np.array(outline.points, dtype=[('x',float), ('y',float)])

    # Iterate over each contour
    start, end = 0, 0
    VERTS, CODES = [], []
    for i in range(len(outline.contours)):

        end    = outline.contours[i]
        points = outline.points[start:end+1]
        points.append(points[0])
        tags   = outline.tags[start:end+1]
        tags.append(tags[0])

        segments = [ [points[0],], ]
        for j in range(1, len(points) ):
            segments[-1].append(points[j])
            if tags[j] & (1 << 0) and j < (len(points)-1):
                segments.append( [points[j],] )

        verts = [points[0], ]
        codes = [Path.MOVETO,]

        for segment in segments:

            if len(segment) == 2:
                verts.extend(segment[1:])
                codes.extend([Path.LINETO])
            elif len(segment) == 3:
                verts.extend(segment[1:])
                codes.extend([Path.CURVE3, Path.CURVE3])
            else:
                verts.append(segment[1])
                codes.append(Path.CURVE3)
                for i in range(1,len(segment)-2):
                    A,B = segment[i], segment[i+1]
                    C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                    verts.extend([ C, B ])
                    codes.extend([ Path.CURVE3, Path.CURVE3])
                verts.append(segment[-1])
                codes.append(Path.CURVE3)

        VERTS.extend(verts)
        CODES.extend(codes)

        start = end + 1

    path = Path(VERTS, CODES)
    polygons = path.to_polygons()
    points = polygons[-1] * 0.1
    holes = [p * 0.1 for p in polygons[:-1]]

    V, F = constrained_delaunay_triangulation(points, holes=holes)
    mesh = Mesh.from_vertices_and_faces(V, F)

    return mesh


def text_2_mesh(text, filepath="Vera.ttf"):
    """
    Convert a text to a mesh.

    Parameters
    ----------
    text: `str`
        The text to convert.
    filepath: `str`, optional
        The path to the font file.

    Returns
    -------
    mesh: `compas.datastructures.Mesh`
        The joined mesh of all characters in the text.
    """
    meshes = []
    xsize = 0.0
    for char in text:
        mesh = char_2_mesh(char, filepath="Vera.ttf")
        vertices, faces = mesh.to_vertices_and_faces()
        bbox = Box.from_bounding_box(bounding_box(vertices))
        T = Translation.from_vector([xsize, 0.0, 0.0])
        mesh.transform(T)
        xsize += bbox.xsize * 1.1
        meshes.append(mesh)

    mesh = meshes_join(meshes)

    return mesh


if __name__ == '__main__':

    from jax_fdm.visualization import Viewer

    mesh = text_2_mesh("100")

    viewer = Viewer(
        width=900,
        height=900,
        show_grid=True,
        viewmode="ghosted"
    )
    
    viewer.add(mesh)

    viewer.show()
