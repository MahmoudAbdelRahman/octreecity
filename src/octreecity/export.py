"""Old name, kept so existing code keeps working: use octreecity.exports
(write_vtk) and octreecity.render (plot_slice) instead."""

import warnings

from .exports import write_vtk
from .render import plot_slice

warnings.warn("octreecity.export is deprecated: use octreecity.exports.write_vtk and "
              "octreecity.render.plot_slice", DeprecationWarning, stacklevel=2)

__all__ = ["write_vtk", "plot_slice"]
