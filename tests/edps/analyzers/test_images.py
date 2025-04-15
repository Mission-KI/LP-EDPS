from pathlib import Path

from extended_dataset_profile import ImageColorMode, ImageDPI, Resolution
from PIL.Image import open as open_image

from edps.analyzers.images import ImageMetadata
from edps.analyzers.images.importer import parse_raster_image


async def test_import_png(path_data_test_png):
    metadata = load_image_metadata(path_data_test_png)
    assert metadata.codec == "PNG"
    assert metadata.color_mode == ImageColorMode.PALETTED
    assert metadata.resolution == Resolution(width=600, height=400)
    assert metadata.dpi == ImageDPI(x=96.012, y=96.012)


async def test_import_jpg(path_data_test_jpg):
    metadata = load_image_metadata(path_data_test_jpg)
    assert metadata.codec == "JPEG"
    assert metadata.color_mode == ImageColorMode.RGB
    assert metadata.resolution == Resolution(width=600, height=400)
    assert metadata.dpi == ImageDPI(x=96, y=96)


def load_image_metadata(path: Path) -> ImageMetadata:
    with open_image(path) as img:
        img_metadata, img_array = parse_raster_image(img)
        return img_metadata
