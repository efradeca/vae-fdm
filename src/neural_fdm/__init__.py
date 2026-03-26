"""Neural FDM: Differentiable form-finding with neural networks for architectural structures."""

import os

__version__ = "0.1.0"

HERE = os.path.dirname(__file__)
HOME = os.path.abspath(os.path.join(HERE, "../../"))
DATA = os.path.abspath(os.path.join(HOME, "data"))
FIGURES = os.path.abspath(os.path.join(HOME, "figures"))
SCRIPTS = os.path.abspath(os.path.join(HOME, "scripts"))

__all__ = [
    "__version__",
    "DATA",
    "FIGURES",
    "SCRIPTS",
]
