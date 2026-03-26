import jax.numpy as jnp

# ===============================================================================
# Transformations
# ===============================================================================

def get_world_mirror_matrix(plane):
    """
    Create a mirror matrix for a given plane.

    Parameters
    ----------
    plane: `str`
        The plane to mirror across. Must be one of 'xy', 'yz', or 'xz'.

    Returns
    -------
    mirror_matrix: `jax.Array`
        The mirror matrix for the given plane.
    """
    if plane.lower() == 'xy':
        # Mirroring across the XY plane (change Z coordinate)
        mirror_matrix = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif plane.lower() == 'yz':
        # Mirroring across the YZ plane (change X coordinate)
        mirror_matrix = jnp.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif plane.lower() == 'xz':
        # Mirroring across the XZ plane (change Y coordinate)
        mirror_matrix = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid plane. Choose 'xy', 'yz', or 'xz'.")

    return mirror_matrix


def mirror_points(points, mirror_matrix):
    """
    Mirror points across a given plane.

    Parameters
    ----------
    points: `jax.Array`
        The points to mirror.
    mirror_matrix: `jax.Array`
        The mirror matrix.

    Returns
    -------
    mirrored_points: `jax.Array`
        The mirrored points.
    """
    return points @ mirror_matrix


# ===============================================================================
# Grid functions
# ===============================================================================

def get_grid_tile_quarter(grid_size, grid_num_pts):
    """
    Get the 3D coordinates of a quarter tile of a control point grid.

    Parameters
    ----------
    grid_size: `int`
        The size of the grid.
    grid_num_pts: `int`
        The number of points along one side of the grid.

    Returns
    -------
    tile: `jax.Array`
        The 3D coordinates.
    """
    half_grid_size = grid_size / 2.0
    grid_step = half_grid_size / (grid_num_pts - 1.0)

    pt0 = [grid_step, grid_step, 0.0]
    pt1 = [half_grid_size, grid_step, 0.0]
    pt2 = [grid_step, half_grid_size, 0.0]
    pt3 = [half_grid_size, half_grid_size, 0.0]

    return jnp.array([pt0, pt1, pt2, pt3])


def calculate_grid_from_tile_quarter(tile):
    """
    Generate an ordered grid of control points from a quarter tile.

    Parameters
    ----------
    tile: `jax.Array`
        The 3D coordinates of a quarter tile.

    Returns
    -------
    grid_points: `jax.Array`
        The 3D coordinates of the grid.
    """
    grid_points = tile

    # mirror tile once
    mirrored_points = mirror_points(grid_points, get_world_mirror_matrix("yz"))
    grid_points = jnp.concatenate((grid_points, mirrored_points))

    # mirror tile again
    mirrored_points = mirror_points(grid_points, get_world_mirror_matrix("xz"))
    grid_points = jnp.concatenate((grid_points, mirrored_points))

    return grid_points


def get_grid_tile_half(grid_size, grid_num_pts):
    """
    Get the 3D coordinates of a half tile of a control point grid.

    Parameters
    ----------
    grid_size: `int`
        The size of the grid.
    grid_num_pts: `int`
        The number of points along one side of the grid.

    Returns
    -------
    tile: `jax.Array`
        The 3D coordinates.
    """
    tile_quarter = get_grid_tile_quarter(grid_size, grid_num_pts)

    # mirror tile once
    mirrored_points = mirror_points(tile_quarter, get_world_mirror_matrix("yz"))

    return jnp.concatenate((tile_quarter, mirrored_points))


def calculate_grid_from_tile_half(tile):
    """
    Generate an ordered grid of control points from a half tile.

    Parameters
    ----------
    tile: `jax.Array`
        The 3D coordinates of a half tile.

    Returns
    -------
    grid_points: `jax.Array`
        The 3D coordinates of the grid.
    """
    grid_points = tile

    # mirror tile once
    mirrored_points = mirror_points(grid_points, get_world_mirror_matrix("xz"))
    grid_points = jnp.concatenate((grid_points, mirrored_points))

    return grid_points


def get_grid_tile_full(grid_size, grid_num_pts):
    """
    Get the 3D coordinates of a full tile of a control point grid.

    Parameters
    ----------
    grid_size: `int`
        The size of the grid.
    grid_num_pts: `int`
        The number of points along one side of the grid.

    Returns
    -------
    tile: `jax.Array`
        The 3D coordinates.
    """
    tile = get_grid_tile_quarter(grid_size, grid_num_pts)

    return calculate_grid_from_tile_quarter(tile)


def calculate_grid_from_tile_full(tile):
    """
    Generate an ordered grid of control points from a full tile.

    Parameters
    ----------
    tile: `jax.Array`
        The 3D coordinates of a full tile.

    Returns
    -------
    grid_points: `jax.Array`
        The 3D coordinates of the grid.
    """
    grid_points = tile

    return grid_points


# ===============================================================================
# Grids
# ===============================================================================

class PointGrid:
    """
    A grid of control points.

    Parameters
    ----------
    tile: `jax.Array`
        The 3D coordinates of a tile.
    num_pts: `int`
        The number of points along one side of the grid.

    Notes
    -----
    The order of the points in a 4x4 grid must be:

    3 7 11 15
    2 6 10 14
    1 5  9 13
    0 4  8 12
    """
    def __init__(self, tile, num_pts) -> None:
        self.tile = tile
        self.num_pts = num_pts

        # NOTE: indices are hard-coded from Rhino to match expected grid order.
        self.indices = [15, 13, 5, 7, 14, 12, 4, 6, 10, 8, 0, 2, 11, 9, 1, 3]

    def points(self, transform=None):
        """
        Get the reindexed and transformed control points of the grid.

        Parameters
        ----------
        transform: `jax.Array` or `None`, optional
            The translation vector.
            If `None`, the control points are returned without any transformation.

        Returns
        -------
        points: `jax.Array`
            The control points.
        """
        tile = self.tile
        if transform is not None:
            tile = self.tile + transform

        points = self.points_grid(tile)
        grid_points = self.reindex_grid(points)

        return jnp.reshape(grid_points, (self.num_pts, self.num_pts, 3))

    def reindex_grid(self, points):
        """
        Reconfigure the grid using hard-coded indices.

        Parameters
        ----------
        points: `jax.Array`
            The control points.

        Returns
        -------
        reindexed_points: `jax.Array`
            The reindexed control points.
        """
        return points[self.indices, :]

    def points_grid(self, tile):
        """
        Generate the control points of the grid from a tile.

        Parameters
        ----------
        tile: `jax.Array`
            The 3D coordinates of a tile.

        Returns
        -------
        points: `jax.Array`
            The control points.
        """
        raise NotImplementedError


class PointGridSymmetricDouble(PointGrid):
    """
    A doubly-symmetric grid of control points.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    """
    def __init__(self, size, num_pts):
        tile = get_grid_tile_quarter(size, num_pts)
        super().__init__(tile, num_pts)

    def points_grid(self, tile):
        return calculate_grid_from_tile_quarter(tile)


class PointGridSymmetric(PointGrid):
    """
    A symmetric grid of control points.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    """
    def __init__(self, size, num_pts):
        tile = get_grid_tile_half(size, num_pts)
        super().__init__(tile, num_pts)

    def points_grid(self, tile):
        """
        Generate the control points of the grid from a tile.

        Parameters
        ----------
        tile: `jax.Array`
            The 3D coordinates of a tile.

        Returns
        -------
        points: `jax.Array`
            The control points.
        """
        return calculate_grid_from_tile_half(tile)


class PointGridAsymmetric(PointGrid):
    """
    An asymmetric grid of control points.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    """
    def __init__(self, size, num_pts):
        tile = get_grid_tile_full(size, num_pts)
        super().__init__(tile, num_pts)

    def points_grid(self, tile):
        """
        Generate the control points of the grid from a tile.

        Parameters
        ----------
        tile: `jax.Array`
            The 3D coordinates of a tile.

        Returns
        -------
        points: `jax.Array`
            The control points.
        """
        return calculate_grid_from_tile_full(tile)
