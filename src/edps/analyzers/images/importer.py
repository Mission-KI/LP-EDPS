import asyncio
from pathlib import Path

import numpy as np
from extended_dataset_profile import ImageColorMode, ImageDataSet, ImageDPI, Resolution
from PIL.Image import Image
from PIL.Image import open as open_image

from edps.analyzers.images import ImageAnalyzer, ImageMetadata
from edps.taskcontext import TaskContext


async def raster_image_importer(ctx: TaskContext, path: Path) -> ImageDataSet:
    ctx.logger.info("Analyzing raster image '%s'", ctx.relative_path(path))

    def runner():
        with open_image(path) as img:
            return parse_raster_image(img)

    img_metadata, img_array = await asyncio.to_thread(runner)
    analyzer = ImageAnalyzer(img_metadata, img_array)
    return await analyzer.analyze(ctx)


async def raster_image_importer_from_pilimage(ctx: TaskContext, image: Image) -> ImageDataSet:
    ctx.logger.info("Analyzing raster image '%s'", ctx.qualified_path)
    img_metadata, img_array = parse_raster_image(image)
    analyzer = ImageAnalyzer(img_metadata, img_array)
    return await analyzer.analyze(ctx)


def parse_raster_image(img: Image) -> tuple[ImageMetadata, np.ndarray]:
    codec = img.format
    color_mode = img.mode
    resolution = img.size
    dpi = img.info.get("dpi")
    img_rgb = img.convert(ImageColorMode.RGB)
    img_array = np.array(img_rgb)
    img_metadata = ImageMetadata(
        codec=codec or "UNKNOWN",
        color_mode=ImageColorMode(color_mode),
        resolution=Resolution(width=resolution[0], height=resolution[1]),
        dpi=ImageDPI(x=dpi[0], y=dpi[1]) if dpi else None,
    )
    return img_metadata, img_array
