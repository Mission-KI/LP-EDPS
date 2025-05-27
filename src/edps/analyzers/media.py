import warnings
from pathlib import Path

from edps.analyzers.audio import analyse_audio
from edps.analyzers.video import analyse_video
from edps.taskcontext import TaskContext
from edps.types import DataSet

from .wrapped_ffmpeg import probe as ffmpeg_probe


async def media_importer(ctx: TaskContext, path: Path) -> DataSet:
    ctx.logger.info("Analyzing media file '%s'", ctx.relative_path(path))

    probe_result = ffmpeg_probe(path)
    format_info = probe_result.get("format", {})
    streams = probe_result.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if len(video_streams) > 0:
        if len(video_streams) > 1:
            message = (
                f'More than one video stream detected for "{ctx.relative_path(path)}", only the first one is analysed.'
            )
            ctx.logger.warning(message)
            warnings.warn(message)

        for i, audio_stream in enumerate(audio_streams):
            ds_name = f"audio_{i + 1}"
            await ctx.exec(ds_name, analyse_audio, path, format_info, audio_stream, i)

        return await analyse_video(ctx, format_info, video_streams[0])

    elif len(audio_streams) > 0:
        if len(audio_streams) > 1:
            message = (
                f'More than one audio stream detected for "{ctx.relative_path(path)}", only the first one is analysed.'
            )
            ctx.logger.warning(message)
            warnings.warn(message)

        return await analyse_audio(ctx, path, format_info, audio_streams[0], 0)

    else:
        raise RuntimeError(f'Could not detect any video or audio streams for "{ctx.relative_path(path)}"')
