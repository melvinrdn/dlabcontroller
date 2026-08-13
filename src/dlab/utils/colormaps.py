from __future__ import annotations

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, ListedColormap

COLORMAPS = [
    "cmr.rainforest",
    "cmr.neutral",
    "cmr.sunburst",
    "cmr.freeze",
    "turbo",
    "viridis",
    "plasma",
    "nipy_spectral",
    "saturation",
]


def _build_saturation_cmap() -> ListedColormap:
    """Linear black-to-white colormap with the top level shown in red.

    Pair with the "Fix Colorbar" max value set to the sensor's true
    saturation level (e.g. 255, 4095, 65535) to spot clipped pixels.
    """
    colors = np.ones((256, 4))
    gray = np.linspace(0.0, 1.0, 255)
    colors[:255, 0] = gray
    colors[:255, 1] = gray
    colors[:255, 2] = gray
    colors[255] = [1.0, 0.0, 0.0, 1.0]
    return ListedColormap(colors, name="saturation")


_SATURATION_CMAP = _build_saturation_cmap()


def resolve_cmap(key: str) -> Colormap:
    """Resolve a name from COLORMAPS to a matplotlib Colormap."""
    if key == "saturation":
        return _SATURATION_CMAP
    if key.startswith("cmr."):
        name = key.split(".", 1)[1]
        return getattr(cmr, name)
    return plt.get_cmap(key)
