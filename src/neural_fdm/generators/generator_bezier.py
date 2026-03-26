import jax.numpy as jnp
import jax.random as jrn

from neural_fdm.generators.bezier import (
    BezierSurfaceAsymmetric,
    BezierSurfaceSymmetric,
    BezierSurfaceSymmetricDouble,
    evaluate_bezier_surface,
)
from neural_fdm.generators.generator import PointGenerator

# ===============================================================================
# Generators
# ===============================================================================

class BezierSurfacePointGenerator(PointGenerator):
    """
    A generator that outputs point evaluated on a wiggled bezier surface.

    Parameters
    ----------
    surface: `BezierSurface`
        The surface to sample points from.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].
    minval: `jax.Array`
        The minimum values of the space of random translations.
    maxval: `jax.Array`
        The maximum values of the space of random translations.
    """
    def __init__(self, surface, u, v, minval, maxval):
        self._check_array_shapes(surface, minval, maxval)

        self.surface = surface
        self.u = u
        self.v = v
        self.minval = minval
        self.maxval = maxval

    def _check_array_shapes(self, surface, minval, maxval):
        """
        Verify that input shapes are consistent.

        Parameters
        ----------
        surface: `BezierSurface`
            The surface to sample points from.
        minval: `jax.Array`
            The minimum values of the space of random translations.
        maxval: `jax.Array`
            The maximum values of the space of random translations.
        """
        tile_shape = surface.grid.tile.shape
        minval_shape = minval.shape
        maxval_shape = maxval.shape

        assert minval_shape == tile_shape, f"{minval_shape} vs. {tile_shape}"
        assert maxval_shape == tile_shape, f"{maxval_shape} vs. {tile_shape}"

    def wiggle(self, key):
        """
        Sample a translation vector from a uniform distribution.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.

        Returns
        -------
        transform: `jax.Array`
            The translation vector.
        """
        shape = self.surface.grid.tile.shape
        return jrn.uniform(key, shape=shape, minval=self.minval, maxval=self.maxval)

    def evaluate_points(self, transform):
        """
        Generate transformed points.

        Parameters
        ----------
        transform: `jax.Array`
            The translation vector.

        Returns
        -------
        points: `jax.Array`
            The transformed points.
        """
        points = self.surface.evaluate_points(self.u, self.v, transform)

        return jnp.ravel(points)

    def __call__(self, key, wiggle=True):
        """
        Generate (wiggled) points.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.
        wiggle: `bool`, optional
            Whether to wiggle the points at random.

        Returns
        -------
        points: `jax.Array`
            The points on the surface.
        """
        if wiggle:
            transform = self.wiggle(key)

        return self.evaluate_points(transform)


class BezierSurfaceSymmetricDoublePointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled, doubly-symmetric bezier surface.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].
    minval: `jax.Array`
        The minimum values of the space of random translations.
    maxval: `jax.Array`
        The maximum values of the space of random translations.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, *args, **kwargs):
        surface = BezierSurfaceSymmetricDouble(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceSymmetricPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled, symmetric bezier surface.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].
    minval: `jax.Array`
        The minimum values of the space of random translations.
    maxval: `jax.Array`
        The maximum values of the space of random translations.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, *args, **kwargs):
        surface = BezierSurfaceSymmetric(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceAsymmetricPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled bezier surface.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].
    minval: `jax.Array`
        The minimum values of the space of random translations.
    maxval: `jax.Array`
        The maximum values of the space of random translations.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, *args, **kwargs):
        surface = BezierSurfaceAsymmetric(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceLerpPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs points interpolated between two wiggled bezier surfaces.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].
    minval: `jax.Array`
        The minimum values of the space of random translations.
    maxval: `jax.Array`
        The maximum values of the space of random translations.
    alpha: `float`
        The interpolation value.

    Notes
    -----
    One surface is doubly-symmetric and the other asymmetric.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, alpha, *args, **kwargs):
        minval_a, minval_b = minval
        maxval_a, maxval_b = maxval

        surface = BezierSurfaceSymmetricDouble(size, num_pts)
        super().__init__(surface, u, v, minval_a, maxval_a)
        self.generator_other = BezierSurfaceAsymmetricPointGenerator(
            size,
            num_pts,
            u,
            v,
            minval_b,
            maxval_b)
        self.alpha = alpha

    def __call__(self, key, wiggle=True):
        """
        Generate (wiggled) points.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.
        wiggle: `bool`, optional
            Whether to wiggle the points at random.

        Returns
        -------
        points: `jax.Array`
            The points on the surface.
        """
        if wiggle:
            transform_this = self.wiggle(key)
            transform_other = self.generator_other.wiggle(key)

        control_points_this = self.surface.control_points(transform_this)
        control_points_other = self.generator_other.surface.control_points(transform_other)
        control_points = (1.0 - self.alpha) * control_points_this + self.alpha * control_points_other

        points = evaluate_bezier_surface(control_points, self.u, self.v)

        return jnp.ravel(points)
