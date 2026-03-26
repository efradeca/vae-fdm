class PointGenerator:
    """
    A generator that samples random points on a target shape.
    """
    def __call__(self, key, wiggle=True):
        """
        Generate points.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            The random key.
        wiggle: `bool`
            If True, the points are wiggled.

        Returns
        -------
        points: `jax.Array`
            The points on the target shape.
        """
        raise NotImplementedError("Subclasses must implement this method.")
