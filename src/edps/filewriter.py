import asyncio
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Tuple
from warnings import warn

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.style
import seaborn
from extended_dataset_profile import ExtendedDatasetProfile

from .file import build_real_sub_path, sanitize_path
from .taskcontext import TaskContext

TEXT_ENCODING = "utf-8"

MATPLOTLIB_BACKEND = "AGG"
MATPLOTLIB_PLOT_FORMAT = ".png"
MATPLOTLIB_STYLE_PATH = Path(__file__).parent / "styles/plot.mplstyle"
MATPLOTLIB_COLOR_MAP_NAME = "daseen"


async def write_edp(ctx: TaskContext, name: PurePosixPath, edp: ExtendedDatasetProfile):
    """Write EDP to a JSON file.

    Create a file with the given name (and ".json" extension) in ctx.output_path.
    Return the path of the new file relative to ctx.output_path."""

    save_path = _prepare_save_path(ctx, name.with_suffix(".json"))
    relative_save_path: Path = save_path.relative_to(ctx.output_path)
    with open(save_path, "wt", encoding=TEXT_ENCODING) as io_wrapper:
        json: str = edp.model_dump_json(by_alias=True)
        await asyncio.to_thread(io_wrapper.write, json)
    ctx.logger.debug('Generated EDP file "%s"', relative_save_path)


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
    save_path = _prepare_save_path(ctx, name.with_suffix(MATPLOTLIB_PLOT_FORMAT))
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


def _prepare_save_path(ctx: TaskContext, name: PurePosixPath):
    save_path = build_real_sub_path(ctx.output_path, sanitize_path(str(name)))
    if save_path.exists():
        message = f'The path "{save_path}" already exists, will overwrite! This is most likely an implementation error.'
        warn(message, RuntimeWarning)
        ctx.logger.warning(message)
        save_path.unlink()
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path
