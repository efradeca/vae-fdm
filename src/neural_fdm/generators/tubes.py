import jax.numpy as jnp
import jax.random as jrn
from jax import vmap

from neural_fdm.generators.generator import PointGenerator

# ===============================================================================
# Generators
# ===============================================================================

class TubePointGenerator(PointGenerator):
    """
    A generator that outputs point evaluated on a wiggled tube.
    """
    pass


class EllipticalTubePointGenerator(TubePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled elliptical tube.

    Parameters
    ----------
    height: `float`
        The height of the tube.
    radius: `float`
        The reference radius of the tube.
    num_sides: `int`
        The number of sides per ellipse.
    num_levels: `int`
        The number of levels along the height of the tube.
    num_rings: `int`
        The number of levels that will work as compression rings. The first and last levels are fully supported.
    minval: `jax.Array`
        The minimum values of the space of random transformations.
    maxval: `jax.Array`
        The maximum values of the space of random transformations.
    """
    def __init__(
            self,
            height,
            radius,
            num_sides,
            num_levels,
            num_rings,
            minval,
            maxval):

        # sanity checks
        assert num_rings >= 3, "Must include at least 1 ring in the middle!"
        self._check_array_shapes(num_rings, minval, maxval)

        self.height = height
        self.radius = radius

        self.num_sides = num_sides
        self.num_levels = num_levels
        self.num_rings = num_rings

        self.minval = minval
        self.maxval = maxval

        self.levels_rings_comp = self._levels_rings_compression()
        self.indices_rings_comp_ravel = self._indices_rings_compression_ravel()
        self.indices_rings_comp_interior_ravel = self._indices_rings_compression_interior_ravel()

        self.levels_rings_tension = self._levels_rings_tension()

        self.shape_tube = (num_levels, num_sides, 3)
        self.shape_rings = (num_rings, num_sides, 3)

    def __call__(self, key, wiggle=True):
        """
        Generate points.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.
        wiggle: `bool`, optional
            Whether to wiggle the points at random.

        Returns
        -------
        points: `jax.Array`
            The points on the tube.
        """
        points = self.points_on_tube(key, wiggle)

        return jnp.ravel(points)

    def _levels_rings_tension(self):
        """
        Compute the integer indices of the levels that work as tension rings.

        Returns
        -------
        indices: `jax.Array`
            The indices.
        """
        indices = [i for i in range(self.num_levels) if i not in self.levels_rings_comp]
        indices = jnp.array(indices, dtype=jnp.int64)

        assert indices.size == self.num_levels - self.num_rings

        return indices

    def _levels_rings_compression(self):
        """
        Compute the integer indices of the levels that work as compression rings.

        Returns
        -------
        indices: `jax.Array`
            The indices.
        """
        step = int(self.num_levels / (self.num_rings - 1))

        indices = [0] + list(range(step, self.num_levels - 1, step)) + [self.num_levels - 1]
        indices = jnp.array(indices, dtype=jnp.int64)

        assert indices.size == self.num_rings

        return indices

    def _indices_rings_compression_ravel(self):
        """
        Compute the integer indices of the vertices in the compression rings.

        Returns
        -------
        indices: `jax.Array`
            The indices.
        """
        indices = []
        for index in self.levels_rings_comp:
            start = index * self.num_sides
            end = start + self.num_sides
            indices.extend(range(start, end))

        indices = jnp.array(indices, dtype=jnp.int64)

        return indices

    def _indices_rings_compression_interior_ravel(self):
        """
        Compute the integer indices of the vertices in the unsupported compression rings.

        Returns
        -------
        indices: `jax.Array`
            The indices.
        """
        indices = []
        for index in self.levels_rings_comp[1:-1]:
            start = index * self.num_sides
            end = start + self.num_sides
            indices.extend(range(start, end))

        indices = jnp.array(indices, dtype=jnp.int64)

        return indices

    def wiggle(self, key):
        """
        Sample random radii and angles from a uniform distribution.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.

        Returns
        -------
        transform: tuple of `jax.Array`
            The transformation factors for the radii and angles.
        """
        return self.wiggle_radii(key), self.wiggle_angle(key)

    def wiggle_radii(self, key):
        """
        Sample random radii from a uniform distribution.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.

        Returns
        -------
        radii: `jax.Array`
            The random radii.
        """
        shape = (self.num_rings, 2)
        minval = self.minval[:2]
        maxval = self.maxval[:2]

        return jrn.uniform(key, shape=shape, minval=minval, maxval=maxval)

    def wiggle_angle(self, key):
        """
        Sample random angles from a uniform distribution.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.

        Returns
        -------
        angles: `jax.Array`
            The random angles.
        """
        shape = (self.num_rings,)
        minval = self.minval[2]
        maxval = self.maxval[2]

        return jrn.uniform(key, shape=shape, minval=minval, maxval=maxval)

    def evaluate_points(self, transform):
        """
        Generate wiggled points.

        Parameters
        ----------
        transform: tuple of `jax.Array`
            The random radii and angles.

        Returns
        -------
        points: `jax.Array`
            The points.
        """
        heights = jnp.linspace(0.0, self.height, self.num_levels)
        radii = jnp.ones(shape=(self.num_levels, 2)) * self.radius
        angles = jnp.ones(shape=(self.num_levels,))

        wiggle_radii, wiggle_angle = transform
        wiggle_radii = wiggle_radii * self.radius
        radii = radii.at[self.levels_rings_comp, :].set(wiggle_radii)
        angles = angles.at[self.levels_rings_comp].set(wiggle_angle)

        points = points_on_ellipses(
            radii[:, 0],
            radii[:, 1],
            heights,
            self.num_sides,
            angles,
        )

        return jnp.ravel(points)

    def points_on_tube(self, key=None, wiggle=False):
        """
        Evaluate wiggled points on the tube.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.
        wiggle: `bool`, optional
            Whether to wiggle the points at random.

        Returns
        -------
        points: `jax.Array`
            The points on the tube.
        """
        heights = jnp.linspace(0.0, self.height, self.num_levels)
        radii = jnp.ones(shape=(self.num_levels, 2)) * self.radius
        angles = jnp.ones(shape=(self.num_levels,))

        if wiggle:
            wiggle_radii, wiggle_angle = self.wiggle(key)
            wiggle_radii = wiggle_radii * self.radius
            radii = radii.at[self.levels_rings_comp, :].set(wiggle_radii)
            angles = angles.at[self.levels_rings_comp].set(wiggle_angle)

        points = points_on_ellipses(
            radii[:, 0],
            radii[:, 1],
            heights,
            self.num_sides,
            angles,
        )

        return points

    def _check_array_shapes(self, num_rings, minval, maxval):
        """
        Verify that input shapes are consistent.

        Parameters
        ----------
        num_rings: `int`
            The number of rings.
        minval: `jax.Array`
            The minimum values of the space of random transformations.
        maxval: `jax.Array`
            The maximum values of the space of random transformations.
        """
        shape = (3, )
        minval_shape = minval.shape
        maxval_shape = maxval.shape

        assert minval_shape == shape, f"{minval_shape} vs. {shape}"
        assert maxval_shape == shape, f"{maxval_shape} vs. {shape}"


class CircularTubePointGenerator(EllipticalTubePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled circular tube.
    """
    def wiggle_radii(self, key):
        """
        Sample random radii from a uniform distribution.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.

        Returns
        -------
        radii: `jax.Array`
            The random radii.
        """
        shape = (self.num_rings,)
        minval = self.minval[0]
        maxval = self.maxval[0]

        return jrn.uniform(key, shape=shape, minval=minval, maxval=maxval)

    def points_on_tube(self, key=None, wiggle=False):
        """
        Evaluate wiggled points on the tube.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.
        wiggle: `bool`, optional
            Whether to wiggle the points at random.

        Returns
        -------
        points: `jax.Array`
            The points on the tube.
        """
        heights = jnp.linspace(0.0, self.height, self.num_levels)
        radii = jnp.ones(shape=(self.num_levels,)) * self.radius
        angles = jnp.ones(shape=(self.num_levels,))

        if wiggle:
            wiggle_radii, wiggle_angle = self.wiggle(key)
            wiggle_radii = wiggle_radii * self.radius
            radii = radii.at[self.levels_rings_comp].set(wiggle_radii)
            angles = angles.at[self.levels_rings_comp].set(wiggle_angle)

        points = points_on_ellipses(
            radii,
            radii,
            heights,
            self.num_sides,
            angles,
        )

        return points


# ===============================================================================
# Helper functions
# ===============================================================================

def points_on_ellipse_xy(radius_1, radius_2, num_sides, angle=0.0):
    """
    Sample points on an ellipse on the XY plane.

    Parameters
    ----------
    radius_1: `float`
        The radius of the ellipse along the X axis.
    radius_2: `float`
        The radius of the ellipse along the Y axis.
    num_sides: `int`
        The number of sides of the ellipse.
    angle: `float`, optional
        The angle of the ellipse in degrees relative to the X axis.

    Returns
    -------
    points: `jax.Array`
        The points.

    Notes
    -----
    The first and last points are not equal.
    """
    angles = 2 * jnp.pi * jnp.linspace(0.0, 1.0, num_sides + 1)
    angles = jnp.reshape(angles, (-1, 1))
    xs = radius_1 * jnp.cos(angles)
    ys = radius_2 * jnp.sin(angles)

    points = jnp.hstack((xs, ys))[:-1]

    # Calculate rotation matrix
    theta = jnp.radians(angle)
    rotation_matrix = jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

    # Rotate points
    points = points @ rotation_matrix.T

    return points


def points_on_ellipse(radius_1, radius_2, height, num_sides, angle=0.0):
    """
    Sample points on a planar ellipse at a given height.

    Parameters
    ----------
    radius_1: `float`
        The radius of the ellipse along the X axis.
    radius_2: `float`
        The radius of the ellipse along the Y axis.
    height: `float`
        The height of the ellipse.
    num_sides: `int`
        The number of sides of the ellipse.
    angle: `float`, optional
        The angle of the ellipse in degrees relative to the X axis.

    Returns
    -------
    points: `jax.Array`
        The points.

    Notes
    -----
    The first and last points are not equal.
    """
    xy = points_on_ellipse_xy(radius_1, radius_2, num_sides, angle)
    z = jnp.ones((num_sides, 1)) * height

    return jnp.hstack((xy, z))


def points_on_ellipses(radius_1, radius_2, heights, num_sides, angles):
    """
    Sample points on an sequence of ellipses distributed over an array of heights.

    Parameters
    ----------
    radius_1: `jax.Array`
        The radii of the ellipses along the X axis.
    radius_2: `jax.Array`
        The radii of the ellipses along the Y axis.
    heights: `jax.Array`
        The heights of the ellipses.
    num_sides: `int`
        The number of sides of the ellipses.
    angles: `jax.Array`
        The angles of the ellipses in degrees relative to the X axis.

    Returns
    -------
    points: `jax.Array`
        The points on the ellipses.

    Notes
    -----
    The first and last points per ellipse are not equal.
    """
    polygon_fn = vmap(points_on_ellipse, in_axes=(0, 0, 0, None, 0))

    return polygon_fn(radius_1, radius_2, heights, num_sides, angles)


# ===============================================================================
# Main
# ===============================================================================

