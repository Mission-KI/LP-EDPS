import warnings
from typing import Any, Optional

from extended_dataset_profile import (
    Resolution,
    VideoDataSet,
    VideoPixelFormat,
)

from edps.taskcontext import TaskContext


async def analyse_video(ctx: TaskContext, format_info: dict[str, Any], stream_info: dict[str, Any]) -> VideoDataSet:
    codec = stream_info.get("codec_name")
    width = stream_info.get("width")
    height = stream_info.get("height")
    fps = _compute_fps(ctx, stream_info.get("avg_frame_rate"))
    # Not all formats have the duration in the stream, some only in the format data.
    duration = stream_info.get("duration", format_info.get("duration"))
    pixel_format = stream_info.get("pix_fmt")

    return VideoDataSet(
        codec=codec if codec else "UNKNOWN",
        resolution=Resolution(width=int(width), height=int(height)) if width and height else None,
        fps=fps,
        duration=float(duration) if duration else None,
        pixelFormat=VideoPixelFormat(pixel_format) if pixel_format else VideoPixelFormat.UNKNOWN,
    )


def _compute_fps(ctx: TaskContext, avg_frame_rate: Optional[str]) -> Optional[float]:
    if avg_frame_rate is None:
        return None
    try:
        num, den = avg_frame_rate.split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError) as error:
        message = f"Could not determine video FPS: {error}"
        ctx.logger.warning(message)
        warnings.warn(message)
        return None
