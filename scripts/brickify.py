"""
Generate bricks off of the faces of an optimized mesh.
"""

import os
from random import randint

from compas.colors import Color

from compas.datastructures import Mesh
from compas.datastructures import mesh_dual
from compas.datastructures import mesh_thicken
from compas.datastructures import mesh_offset
from compas.datastructures import mesh_delete_duplicate_vertices

from compas.geometry import Line
from compas.geometry import Sphere
from compas.geometry import Box
from compas.geometry import centroid_points
from compas.geometry import oriented_bounding_box_numpy
from compas.geometry import Transformation
from compas.geometry import Scale
from compas.geometry import Frame
from compas.geometry import Plane
from compas.geometry import intersection_line_plane
from compas.geometry import area_triangle
from compas.geometry import normal_triangle
from compas.geometry import circle_from_points
from compas.geometry import normalize_vector
from compas.geometry import add_vectors
from compas.geometry import subtract_vectors

from compas.utilities import pairwise
from compas.utilities import geometric_key

from compas_cgal.booleans import boolean_difference as boolean_difference_mesh_mesh
from compas_cgal.meshing import remesh as mesh_remesh

from jax_fdm.datastructures import FDMesh
from jax_fdm.visualization import Viewer

from neural_fdm import DATA

from text_2_mesh import text_2_mesh


# ===============================================================================
# Helper functions
# ===============================================================================

def triangulate_face_quad(face, reverse=False):
    """
    Triangulate a mesh quad face.

    Parameters
    ___________
    face: `list` of `int`
        The face vertices.
    reverse: `bool`, optional
        If `True`, the face is reversed.

    Returns
    _______
    new_faces: `list` of `list` of `int`
        The two triangulated faces.
    """
    a, b, c, d = face
    if not reverse:
        face_a = [a, b, d]
        face_b = [b, c, d]
    else:
        face_a = [a, b, c]
        face_b = [c, d, a]

    return [face_a, face_b]


def triangulate_face_ngon(face, vertices):
    """
    Triangulate a mesh polygonal face.

    Parameters
    ___________
    face: `list` of `int`
        The face vertices.
    vertices: `list` of `list` of `float`
        The xyz coordinates of the face vertices.

    Returns
    _______
    new_faces: `list` of `list` of `int`
        The triangulated faces.
    """
    midpoint = centroid_points([vertices[vkey] for vkey in face])
    vertices.append(midpoint)
    ckey = len(vertices) - 1

    # create new faces
    new_faces = []
    for a, b in pairwise(face + face[:1]):
        _face = [a, b, ckey]
        new_faces.append(_face)

    return new_faces


def triangulate_face(face, vertices, reverse=False):
    """
    Triangulate a mesh face based on its vertex count.

    The face is a list of indices pointing to a list with the vertices xyz coordinates.

    Parameters
    ___________
    face: `list` of `int`
        The face vertices.
    vertices: `list` of `list` of `float`
        The xyz coordinates of the face vertices.
    reverse: `bool`, optional
        If `True`, the face is reversed.

    Returns
    _______
    new_faces: `list` of `list` of `int`
        The triangulated faces.
    """
    assert len(face) > 2

    new_faces = []
    # triangle
    if len(face) == 3:
        new_faces = [face]
    # quad
    elif len(face) == 4:
        new_faces = triangulate_face_quad(face, reverse)
    # ngon
    else:
        new_faces = triangulate_face_ngon(face, vertices)

    return new_faces


def calculate_brick_thicknesses(thickness):
    """
    Calculate the top and bottom thicknesses of a brick.

    Parameters
    ___________
    thickness: `float`
        The brick thickness.

    Returns
    _______
    thickness_bottom: `float`
        The bottom thickness of the brick.
    thickness_top: `float`
        The top thickness of the brick.
    """
    thickness_bottom = thickness / 3.0
    thickness_top = 2.0 * thickness / 3.0

    return thickness_bottom, thickness_top


def generate_bricks(mesh, thickness):
    """
    Generate a solid brick per mesh face.

    Parameters
    ___________
    mesh: `compas.datastructures.Mesh`
        The mesh whose faces will be turned into bricks.
    thickness: `float`
        The global brick thickness.

    Returns
    _______
    bricks: `dict` of `compas.datastructures.Mesh`
        The bricks meshes (closed, watertight).
    meshes: `tuple` of `compas.datastructures.Mesh`
        The meshes of the bottom and top faces of the bricks to create the scaffolding.
    """
    thick_bottom, thick_top = calculate_brick_thicknesses(thickness)

    mesh_bottom = mesh_offset(mesh, thick_bottom)
    mesh_top = mesh_offset(mesh, thick_top * -1.0)

    bricks = {}
    halfedges_visited = set()

    for i, fkey in enumerate(mesh.faces()):

        # adding side faces
        faces = []

        num_vertices = len(mesh.face_vertices(fkey))
        face_bottom = list(range(num_vertices))
        face_top = list(range(num_vertices, 2 * num_vertices))

        xyz_bottom = mesh_bottom.face_coordinates(fkey)
        xyz_top = mesh_top.face_coordinates(fkey)
        vertices = xyz_bottom + xyz_top

        halfedges = mesh.face_halfedges(fkey)
        iterable = zip(
            pairwise(face_top + face_top[:1]),
            pairwise(face_bottom + face_bottom[:1]),
            halfedges
        )

        # triangulate
        for edge_1, edge_2, halfedge in iterable:
            a, b = edge_1
            d, c = edge_2
            face = [a, b, c, d]
            is_visited = halfedge in halfedges_visited
            tri_faces = triangulate_face_quad(face, is_visited)
            faces.extend(tri_faces)

        # face half edges
        halfedges_visited.update(halfedges)
        halfedges_reversed = [(v, u) for u, v in halfedges]
        halfedges_visited.update(halfedges_reversed)

        # triangulate top and bottom faces
        tri_faces = triangulate_face(face_bottom, vertices)
        faces.extend(tri_faces)

        tri_faces = triangulate_face(face_top, vertices)
        faces.extend([face[::-1] for face in tri_faces])

        # make mesh from face
        brick = Mesh.from_vertices_and_faces(vertices, faces)

        # store brick
        bricks[fkey] = brick

    return bricks, (mesh_bottom, mesh_top)


def add_text_engraving(fkey, bricks, mesh_bottom, text, depth=1):
    """
    Engrave the underside of a brick mesh with text.

    Parameters
    ___________
    fkey: `int`
        The face key of the brick to engrave.
    bricks: `dict` of `compas.datastructures.Mesh`
        The brick meshes.
    mesh_bottom: `compas.datastructures.Mesh`
        The mesh of the bottom face of the brick.
    text: `str`
        The text to engrave.
    depth: `float`, optional
        The depth of the engraving.

    Returns
    _______
    text_mesh: `compas.datastructures.Mesh`
        The thickened text mesh.
    bbox_mesh: `compas.datastructures.Mesh`
        The bounding box of the text.
    face_mesh: `compas.datastructures.Mesh`
        A mesh with the face of the brick where the text is engraved.
    brick: `compas.datastructures.Mesh`
        The engraved brick mesh.
    """
    # generate text mesh
    text_mesh = text_2_mesh(text)
    vertices, _ = text_mesh.to_vertices_and_faces()

    # calculate bounding box and properties
    bbox = oriented_bounding_box_numpy(vertices)
    bbox = Box.from_bounding_box(bbox)
    bbox_largest = max([bbox.width, bbox.depth, bbox.height])

    # find face in brick
    brick = bricks[fkey]

    # generate face plane
    vertices = mesh_bottom.face_coordinates(fkey)
    face = list(range(len(vertices)))
    tri_faces = triangulate_face(face, vertices)

    # take face with largest area
    tri_faces = sorted(tri_faces, key=lambda x: area_triangle([vertices[v] for v in x]))
    big_face = tri_faces[-1]
    big_face_vertices = [vertices[v] for v in big_face]
    _, radius = circle_from_points(*big_face_vertices)

    big_face_mesh = Mesh.from_vertices_and_faces(vertices, [big_face])

    center = centroid_points(big_face_vertices)
    normal = normal_triangle(big_face_vertices)    
    plane = Plane(center, normal)
    brick_frame = Frame.from_plane(plane)

    # box again
    ratio = (0.4 * radius) / bbox_largest
    S = Scale.from_factors(factors=[ratio] * 3)
    bbox.transform(S)

    # text again
    text_origin = [x for x in bbox.frame.point]
    text_frame = Frame(text_origin, [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    text_mesh.transform(S)
    T = Transformation.from_frame_to_frame(text_frame, brick_frame)
    text_mesh.transform(T)

    B = text_mesh.to_vertices_and_faces(triangulated=True)
    B = mesh_remesh(B, radius/30.0, 10)
    text_mesh = Mesh.from_vertices_and_faces(*B)

    text_mesh = mesh_thicken(text_mesh, depth)

    vertices, faces = bbox.to_vertices_and_faces()
    bbox_mesh = Mesh.from_vertices_and_faces(vertices, faces)

    A = brick.to_vertices_and_faces()
    B = text_mesh.to_vertices_and_faces(triangulated=True)

    V, F = boolean_difference_mesh_mesh(A, B)
    brick = Mesh.from_vertices_and_faces(V, F)

    return text_mesh, bbox_mesh, big_face_mesh, brick


def generate_scaffolding(mesh, thickness):
    """
    Generate the scaffolding platform under the bricks.

    Parameters
    ----------
    mesh: `compas.datastructures.Mesh`
        The shell the scaffolding will support.
    thickness: `float`
        The scaffolding thickness.
    """
    # convert
    scaffold_mesh = mesh_offset(mesh, 1.0 * thickness / 2.0)
    vertices, faces = scaffold_mesh.to_vertices_and_faces()

    # triangulate existing faces
    faces_tri = []
    for face in faces:
        faces_tri.extend(triangulate_face(face, vertices))

    # add boundary face
    key_index = scaffold_mesh.key_index()
    face_bnd = [key_index[fkey] for fkey in scaffold_mesh.vertices_on_boundary()][::-1]
    faces_tri.extend(triangulate_face(face_bnd, vertices))

    # create mesh
    return Mesh.from_vertices_and_faces(vertices, faces_tri)


def calculate_vertex_nbr_line(vkey, mesh):
    """
    Find the line connecting a boundary vertex to its 'perpendicular' neighbor on the interior.

    Parameters
    ----------
    vkey: `int`
        The vertex key.
    mesh: `compas.datastructures.Mesh`
        The mesh.

    Returns
    -------
    line: `tuple` of `list` of `float`
        The line connecting the vertex to its neighbor.
    """
    # sift neighbors
    nbrs_boundary = []
    nbrs_interior = []
    for nkey in mesh.vertex_neighbors(vkey):
        if nkey == vkey:
            continue
        if mesh.is_vertex_on_boundary(nkey):
            nbrs_boundary.append(nkey)
        else:
            nbrs_interior.append(nkey)
    # cases
    num_nbrs_interior = len(nbrs_interior)
    if num_nbrs_interior == 1:
        start = mesh.vertex_coordinates(vkey)
        end = mesh.vertex_coordinates(nbrs_interior[0])
        line = (start, end)

    elif num_nbrs_interior == 0:
        assert len(nbrs_boundary) == 2
        start = mesh.vertex_coordinates(vkey)
        lines = []
        for nkey in nbrs_boundary:
            line = calculate_vertex_nbr_line(nkey, mesh)
            lines.append(line)

        assert len(lines) > 0

        end = start
        for line in lines:
            a, b = line
            vector = normalize_vector(subtract_vectors(b, a))
            end = add_vectors(end, vector)
        line = (start, end)
    else:
        raise ValueError

    return line


def generate_boundary_support(mesh, thickness):
    """
    Generate the mesh of the support ring bearing the bricks at the boundary.

    Parameters
    ----------
    mesh: `compas.datastructures.Mesh`
        The middle surface of the masonry shell.
    thickness: `float`
        The thickness of the ring.

    Returns
    -------
    support_mesh: `compas.datastructures.Mesh`
        The support ring mesh.
    """
    thick_bottom, thick_top = calculate_brick_thicknesses(thickness)

    mesh_bottom = mesh_offset(mesh, thick_bottom)
    mesh_top = mesh_offset(mesh, thick_top * -2.0)

    # generate xyz polygon bottom
    polygon_bottom = [mesh_bottom.vertex_coordinates(vkey) for vkey in mesh.vertices_on_boundary()]

    # generate xyz polygon top
    polygon_top = [mesh_top.vertex_coordinates(vkey) for vkey in mesh.vertices_on_boundary()]

    # generate xyz polygon that intersects with ground plane
    lines = []
    polygon_offset = []
    polygon_squashed = []
    ground_level = -thick_bottom
    plane = Plane([0.0, 0.0, ground_level], [0.0, 0.0, 1.0])

    for vkey in mesh.vertices_on_boundary():

        start = mesh_top.vertex_coordinates(vkey)
        line = calculate_vertex_nbr_line(vkey, mesh_top)
        point = intersection_line_plane(line, plane)
        if not point:
            raise ValueError("No intersection found!")

        line = Line(start, point)
        point = line.point(0.75)

        line = Line(start, point)
        lines.append(line)

        polygon_offset.append(point)
        point = point[:]
        point[2] = ground_level
        polygon_squashed.append(point)

    # now, weave polygons to make mesh
    polygons = [
        polygon_bottom,
        polygon_top,
        polygon_offset,
        polygon_squashed,
    ]

    # add polygons' points to main vertex list
    max_vkey = 0
    gkey_vkey = {}
    points = []

    for polygon in polygons:
        for point in polygon:
            gkey = geometric_key(point)
            if gkey in gkey_vkey:
                continue
            points.append(point)
            gkey_vkey[gkey] = max_vkey
            max_vkey += 1

    # weave faces
    faces = []
    for polyline in zip(*(pairwise(polygon) for polygon in polygons)):
        for line_a, line_b in pairwise(polyline + polyline[:1]):
            a, b = (gkey_vkey[geometric_key(pt)] for pt in line_a)
            d, c = (gkey_vkey[geometric_key(pt)] for pt in line_b)
            face = [a, b, c, d]
            faces.append(face)

    support_mesh = Mesh.from_vertices_and_faces(points, faces)

    return support_mesh


# ===============================================================================
# Script function
# ===============================================================================

def brickify(
        name,
        thickness,
        scale=1.0,
        dual=True,
        do_bricks=True,
        do_label=True,
        do_scaffold=False,
        do_support=False,
        save=False
):
    """
    Generate bricks on the faces of a mesh. One face = one brick.

    Parameters
    ----------
    name: `str`
        The mesh name (without extension).
    thickness: `float`
        The brick thickness.
    scale: `float`, optional
        The mesh scale, whether to scale it down or up to fit in a printer's bed.
    dual: `bool`, optional
        If `True`, the script will work on the dual of the input mesh.
    do_bricks: `bool`, optional
        If `True`, generate the bricks as closed, watertight meshes.
    do_label: `bool`, optional
        If `True`, engrave the bricks with labels via mesh boolean differences.
    do_scaffold: `bool`, optional
        If `True`, generate scaffolding platform.
    do_support: `bool`, optional
        If `True`, create the perimetral support for the bricks.
    save: `bool`, optional
        If `True`, save all generated data as both JSON and OBJ files.
    """
    CAMERA_CONFIG = {
        "position": (30.34, 30.28, 42.94),
        "target": (0.956, 0.727, 1.287),
        "distance": 20.0,
    }

    # load mesh
    filepath = os.path.join(DATA, f"{name}.json")
    mesh = FDMesh.from_json(filepath)
    print(mesh)

    # calculate dual mesh
    if dual:
        mesh = mesh_dual(mesh, include_boundary=True)
        mesh_delete_duplicate_vertices(mesh)

    # scale mesh
    if scale != 1.0:
        S = Scale.from_factors(factors=[scale] * 3)
        mesh.transform(S)

    # do bricks
    if do_bricks:
        bricks, meshes = generate_bricks(mesh, thickness)
        mesh_bottom, mesh_top = meshes

        if save:
            filepath = os.path.join(DATA, f"{name}_top.json")
            mesh_top.to_json(filepath)
            print(f"Saved mesh top to {filepath}")
            filepath = os.path.join(DATA, f"{name}_bottom.json")
            mesh_bottom.to_json(filepath)
            print(f"Saved mesh bottom to {filepath}")

    # generate scaffolding
    if do_scaffold:
        scaffold_mesh = generate_scaffolding(mesh, thickness)

    # generate perimetral support base
    if do_support:
        support_mesh = generate_boundary_support(mesh, thickness)

    # engrave labels
    if do_bricks and do_label:
        for i, fkey in enumerate(mesh.faces()):
            data = add_text_engraving(
                fkey,
                bricks,
                mesh_bottom,
                f"{i}",
                depth=thickness/2.0
            )
            text_mesh, bbox_mesh, face_mesh, brick = data
            bricks[fkey] = brick

    if save:
        if do_bricks:
            for i, brick in enumerate(bricks.values()):
                filepath = os.path.join(DATA, f"brick_{i}.json")
                brick.to_json(filepath)
                print(f"Saved brick to {filepath}")
                filepath = os.path.join(DATA, f"brick_{i}.obj")
                brick.to_obj(filepath)
                print(f"Saved brick to {filepath}")

        if do_scaffold:
            filepath = os.path.join(DATA, "scaffold.json")
            scaffold_mesh.to_json(filepath)
            print(f"Saved scaffold to {filepath}")
            filepath = os.path.join(DATA, "scaffold.obj")
            scaffold_mesh.to_obj(filepath)
            print(f"Saved scaffold to {filepath}")

        if do_support:
            filepath = os.path.join(DATA, "support.json")
            support_mesh.to_json(filepath)
            print(f"Saved scaffold to {filepath}")
            filepath = os.path.join(DATA, "support.obj")
            support_mesh.to_obj(filepath)
            print(f"Saved scaffold to {filepath}")

    # visualization
    viewer = Viewer(
        width=900,
        height=900,
        show_grid=True,
        viewmode="lighted"
    )

    # modify view
    viewer.view.camera.position = CAMERA_CONFIG["position"]
    viewer.view.camera.target = CAMERA_CONFIG["target"]
    viewer.view.camera.distance = CAMERA_CONFIG["distance"]

    if do_scaffold:
        viewer.add(scaffold_mesh)

    if do_support:
        viewer.add(support_mesh)

    if do_bricks:
        for fkey, brick in bricks.items():
            r, g, b = [randint(0, 255) for _ in range(3)]
            color = Color.from_rgb255(r, g, b)

            viewer.add(
                brick,
                color=color,
                show_points=False,
                show_edges=True
            )

    # show le crème
    viewer.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(brickify)
