from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Tuple

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.style
import seaborn

from .taskcontext import TaskContext

MATPLOTLIB_BACKEND = "AGG"
MATPLOTLIB_PLOT_FORMAT = ".png"
MATPLOTLIB_STYLE_PATH = Path(__file__).parent / "styles/plot.mplstyle"
MATPLOTLIB_COLOR_MAP_NAME = "daseen"


@asynccontextmanager
async def get_pyplot_writer(
    ctx: TaskContext, name: PurePosixPath, **fig_kw
) -> AsyncIterator[Tuple[matplotlib.axes.Axes, PurePosixPath]]:
    """Context manager for a matplotlib `Axes` object.

    `fig_kw` can be used to pass additional parameters to subplots(), e.g. `figsize`.
    The caller should use the yielded `Axes` object to plot her graph.
    This is saved to an image when the context is exited.
    Before using this function `setup_matplotlib()` must be called exactly once."""

    _setup_matplotlib_if_needed()
    save_path = ctx.prepare_output_path(str(name.with_suffix(MATPLOTLIB_PLOT_FORMAT)))
    relative_save_path = save_path.relative_to(ctx.output_path)
    figure, axes = matplotlib.pyplot.subplots(**fig_kw)
    axes.autoscale(True)
    yield axes, PurePosixPath(relative_save_path)
    figure.tight_layout()
    figure.savefig(save_path)
    matplotlib.pyplot.close(figure)
    ctx.logger.debug('Generated plot "%s"', relative_save_path)


def _setup_matplotlib_if_needed():
    """Customize matplotlib for our plots.

    Gets called by get_pyplot_writer() to customize the plots with the beebucket style.
    """
    matplotlib.use(MATPLOTLIB_BACKEND)
    seaborn.reset_orig()
    matplotlib.style.use(str(MATPLOTLIB_STYLE_PATH))
    if matplotlib.colormaps.get(MATPLOTLIB_COLOR_MAP_NAME) is None:
        colormap = _get_default_colormap()
        matplotlib.colormaps.register(colormap)
        matplotlib.pyplot.set_cmap(colormap)


def _get_default_colormap() -> matplotlib.colors.Colormap:
    BLUE = "#43ACFF"
    GRAY = "#D9D9D9"
    PINK = "#FF3FFF"
    return matplotlib.colors.LinearSegmentedColormap.from_list(MATPLOTLIB_COLOR_MAP_NAME, [BLUE, GRAY, PINK])
