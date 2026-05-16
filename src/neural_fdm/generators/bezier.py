import jax.numpy as jnp
from jax import lax, vmap

from neural_fdm.generators.grids import (
    PointGridAsymmetric,
    PointGridSymmetric,
    PointGridSymmetricDouble,
)

# ===============================================================================
# Functions
# ===============================================================================

def factorial(n):
    """
    Calculate the factorial of a number.

    Parameters
    ----------
    n: `int`
        The number to calculate the factorial of.

    Returns
    -------
    factorial: `float`
        The factorial of the number.
    """
    return jnp.where(n < 0, 0, lax.exp(lax.lgamma(n + 1)))


def binomial_coefficient(n, i):
    """
    Compute the binomial coefficient.

    Parameters
    ----------
    n: `int`
        The number to calculate the binomial coefficient of.
    i: `int`
        The index of the binomial coefficient.

    Returns
    -------
    coefficient: `float`
        The binomial coefficient of the number.
    """
    return factorial(n) / (factorial(i) * factorial(n - i))


def bernstein_poly(n, t):
    """
    Compute all Bernstein polynomials of degree `n` at `t` using vectorized operations.

    Parameters
    ----------
    n: `int`
        The degree of the Bernstein polynomial.
    t: `jax.Array`
        The parameter values.

    Returns
    -------
    bernstein_poly: `jax.Array`
        The Bernstein polynomials of the degree `n` at the parameters `t`.
    """
    i = jnp.arange(n + 1)
    binomial_coeff = binomial_coefficient(n, i)

    return binomial_coeff * (t ** i) * ((1 - t) ** (n - i))


def degree_u(control_points):
    """
    Get the degree along the `u` direction of the Bezier surface.

    Parameters
    ----------
    control_points: `jax.Array`
        The control points of the Bezier surface.

    Returns
    -------
    degree: `int`
        The degree along the `u` direction of the Bezier surface.
    """
    n, m, _ = control_points.shape
    n -= 1.0

    return n


def degree_v(control_points):
    """
    Get the degree along the `v` direction of the Bezier surface.

    Parameters
    ----------
    control_points: `jax.Array`
        The control points of the Bezier surface.

    Returns
    -------
    degree: `int`
        The degree along the `v` direction of the Bezier surface.
    """
    n, m, _ = control_points.shape
    m -= 1.0

    return m


def bezier_surface_point(control_points, u, v):
    """
    Evaluate a point on a Bezier surface using a series of dot products.

    Parameters
    ----------
    control_points: `jax.Array`
        The control points of the Bezier surface in the shape (n+1, m+1, 3).
    u: `float`
        The parameter value along the `u` direction in the range [0, 1].
    v: `float`
        The parameter value along the `v` direction in the range [0, 1].

    Returns
    -------
    point: `jax.Array`
        The point on the Bezier surface.
    """
    n = degree_u(control_points)
    m = degree_v(control_points)

    # Compute the Bernstein polynomial values at u and v
    bernstein_u = bernstein_poly(n, u)
    bernstein_v = bernstein_poly(m, v)

    # Calculate the weighted sum of control points along the u direction
    weighted_u = jnp.dot(bernstein_u, control_points)  # shape becomes (m+1, 3)

    # Calculate the final point on the surface by combining the results across v
    point = jnp.dot(weighted_u.T, bernstein_v)

    return point


# Evaluate points
def evaluate_bezier_surface(control_points, u, v):
    """
    Sample a series of 3D points on a Bezier surface with `vmap`.

    Parameters
    ----------
    control_points: `jax.Array`
        The control points of the Bezier surface in the shape (n+1, m+1, 3).
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].

    Returns
    -------
    points: `jax.Array`
        The points on the Bezier surface.
    """
    fn = vmap(vmap(bezier_surface_point,
                   in_axes=(None, 0, None)),
              in_axes=(None, None, 0))

    return fn(control_points, u, v)


def evaluate_bezier_surface_einsum(control_points, u, v):
    """
    Vectorized computation of a point on a Bezier surface via `einsum`.

    Parameters
    ----------
    control_points: `jax.Array`
        The control points of the Bezier surface in the shape (n+1, m+1, 3).
    u: `jax.Array`
        The parameter values along the `u` direction in the range [0, 1].
    v: `jax.Array`
        The parameter values along the `v` direction in the range [0, 1].

    Returns
    -------
    points: `jax.Array`
        The points on the Bezier surface.
    """
    n = degree_u(control_points)
    m = degree_v(control_points)

    # Compute the Bernstein polynomial values at u and v
    bernstein_u = bernstein_poly(n, u[:, :, None])
    bernstein_v = bernstein_poly(m, v[:, :, None])

    # Calculate the weighted sum of control points along the u and v directions
    surface_points = jnp.einsum('ijk,lmi,lmj->lmk',
                                control_points,
                                bernstein_u,
                                bernstein_v)

    return surface_points


# ===============================================================================
# Surfaces
# ===============================================================================


class BezierSurface:
    """
    A Bezier surface.

    Parameters
    ----------
    grid: `PointGrid`
        The grid of points that define the Bezier surface.
    """
    def __init__(self, grid):
        self.grid = grid

    def control_points(self, transform=None):
        """
        """
        return self.grid.points(transform)

    def evaluate_points(self, u, v, transform=None):
        """
        """
        control_points = self.control_points(transform)
        return evaluate_bezier_surface(control_points, u, v)


class BezierSurfaceSymmetric(BezierSurface):
    """
    A symmetric Bezier surface.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    """
    def __init__(self, size, num_pts):
        grid = PointGridSymmetric(size, num_pts)
        super().__init__(grid)


class BezierSurfaceSymmetricDouble(BezierSurface):
    """
    A Bezier surface with double symmetry.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    """
    def __init__(self, size, num_pts):
        grid = PointGridSymmetricDouble(size, num_pts)
        super().__init__(grid)


class BezierSurfaceAsymmetric(BezierSurface):
    """
    A Bezier surface without symmetry.

    Parameters
    ----------
    size: `int`
        The size of the grid.
    num_pts: `int`
        The number of points along one side of the grid.
    """
    def __init__(self, size, num_pts):
        grid = PointGridAsymmetric(size, num_pts)
        super().__init__(grid)


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    grid_size = 4
    num_u = 11  # 12 for youtube
    num_v = 11

    # Control points rhino
    points = [
        [-5, -5, 0],
        [-5, -1.666667, 0],
        [-5, 1.666667, 0],
        [-5, 5, 0],
        [-1.666667, -5, 0],
        [-1.666667, -1.666667, 10],
        [-1.666667, 1.666667, 10],
        [-1.666667, 5, 0],
        [1.666667, -5, 0],
        [1.666667, -1.666667, 10],
        [1.666667, 1.666667, 10],
        [1.666667, 5, 0],
        [5, -5, 0],
        [5, -1.666667, 0],
        [5, 1.666667, 0],
        [5, 5, 0]
    ]
    # Control points Youtube
    # cx = [[-0.5, -2.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    # cy = [[2.0, 1.0, 0.0], [2.0, 0.0, -1.0], [2.0, 1.0, 1.0]]
    # cz = [[1.0, -1.0, 2.0], [0.0, -0.5, 2.0], [0.5, 1.0, 2.0]]
    # cx = jnp.array(cx)
    # cy = jnp.array(cy)
    # cz = jnp.array(cz)

    assert len(points) % grid_size == 0
    points = jnp.array(points)
    control_points = jnp.reshape(points, (grid_size, grid_size, 3))

    # U, V
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)
    u_grid, v_grid = jnp.meshgrid(u, v)

    # surface_points = bezier_surface(control_points, u_grid, v_grid)
    surface_points = evaluate_bezier_surface(control_points, u, v)
    print("Surface Points Shape:", surface_points.shape)  # Should be (10, 10, 3)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(surface_points[:, :, 0],
                    surface_points[:, :, 1],
                    surface_points[:, :, 2])

    ax.scatter(control_points[:, :, 0],
               control_points[:, :, 1],
               control_points[:, :, 2],
               edgecolors="face")

    plt.show()
