import asyncio
import io
import threading
from typing import Optional

import brisque
import cv2
import numpy as np
from extended_dataset_profile import ImageColorMode, ImageDataSet, ImageDPI, Resolution
from pandas import DataFrame
from PIL.Image import fromarray as from_array
from PIL.Image import open as open_image
from skimage.exposure import is_low_contrast
from skimage.restoration import estimate_sigma

from edps.analyzers.images.ocr import OCR
from edps.taskcontext import TaskContext


class ImageMetadata:
    def __init__(self, codec: str, color_mode: ImageColorMode, resolution: Resolution, dpi: Optional[ImageDPI]):
        self.codec = codec
        self.color_mode = color_mode
        self.resolution = resolution
        self.dpi = dpi


class ImageAnalyzer:
    def __init__(self, metadata: ImageMetadata, data: np.ndarray):
        self._data = data
        self._metadata = metadata
        self._detected_texts = DataFrame()

    async def analyze(self, ctx: TaskContext) -> ImageDataSet:
        height, width, _ = self._data.shape
        ctx.logger.info(
            "Started analysis for image dataset of dimensions %d (width) x %d (height)",
            width,
            height,
        )
        brightness = await self._compute_brightness(self._data)
        blurriness = await self._compute_blurriness(self._data)
        sharpness = await self._compute_sharpness(self._data)
        brisque = await self._compute_brisque(self._data)
        noise = await self._compute_noise(self._data)
        low_contrast = await self._is_low_contrast(self._data)
        ela_score = await self._compute_ela_score(self._data, self._metadata)

        # TODO: Process in further analysis steps
        self._detected_texts = await self._detect_texts(self._data)

        return ImageDataSet(
            codec=self._metadata.codec,
            colorMode=self._metadata.color_mode,
            resolution=self._metadata.resolution,
            dpi=self._metadata.dpi,
            brightness=brightness,
            blurriness=blurriness,
            sharpness=sharpness,
            brisque=brisque,
            noise=noise,
            lowContrast=low_contrast,
            elaScore=ela_score,
        )

    async def _compute_brightness(self, img: np.ndarray) -> float:
        img_converted = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        brightness_channel = img_converted[:, :, 1]
        brightness_indication = cv2.mean(brightness_channel)[0]
        return brightness_indication

    async def _compute_blurriness(self, img: np.ndarray) -> float:
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    async def _compute_sharpness(self, img: np.ndarray) -> float:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        gnorm = np.sqrt(laplacian**2)
        sharpness = np.average(gnorm)
        return float(sharpness)

    async def _compute_brisque(self, img: np.ndarray) -> float:
        return float(get_brisque_model().score(img))

    async def _compute_noise(self, img: np.ndarray) -> float:
        return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))

    async def _is_low_contrast(self, img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return bool(is_low_contrast(gray))

    async def _detect_texts(self, img: np.ndarray) -> DataFrame:
        return await asyncio.to_thread(lambda: get_ocr_model().read(img))

    async def _compute_ela_score(self, img: np.ndarray, metadata: ImageMetadata, ref_quality: int = 90):
        if metadata.codec not in ("JPG", "JPEG"):
            return None

        # Save to a buffer in JPEG format, then rewind the buffer
        buffer = io.BytesIO()
        from_array(img).save(buffer, format="JPEG", quality=ref_quality)
        buffer.seek(0)

        with open_image(buffer) as ref_img:
            ref_img_rgb = ref_img.convert(ImageColorMode.RGB)
            ref_img_array = np.array(ref_img_rgb)

        ela_score = np.std((30 * cv2.absdiff(img, ref_img_array)).astype(np.float32))
        return float(ela_score)


class ThreadLocal[T](threading.local):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model: Optional[T] = None


# Thread-local storage for OCR model
local_ocr: ThreadLocal[OCR] = ThreadLocal()


def get_ocr_model() -> OCR:
    if local_ocr.model is None:
        local_ocr.model = OCR(languages=["en", "de"])
    return local_ocr.model


# Thread-local storage for BRISQUE model
local_brisque: ThreadLocal[brisque.BRISQUE] = ThreadLocal()


def get_brisque_model() -> brisque.BRISQUE:
    if local_brisque.model is None:
        local_brisque.model = brisque.BRISQUE(url=False)
    return local_brisque.model
