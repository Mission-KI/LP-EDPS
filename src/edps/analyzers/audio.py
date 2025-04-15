import asyncio
import io
from pathlib import Path, PurePosixPath
from typing import Any

import ffmpeg
import numpy as np
import scipy
from extended_dataset_profile import AudioDataSet
from matplotlib.figure import figaspect
from numpy import ndarray

from edps.filewriter import get_pyplot_writer
from edps.taskcontext import TaskContext


async def analyse_audio(
    ctx: TaskContext, path: Path, format_info: dict[str, Any], stream_info: dict[str, Any], stream_number: int
) -> AudioDataSet:
    codec = stream_info.get("codec_name")
    channels = stream_info.get("channels")
    # Not all formats have the duration in the stream, some only in the format data.
    duration = stream_info.get("duration", format_info.get("duration"))
    sample_rate = stream_info.get("sample_rate")
    bit_rate = stream_info.get("bit_rate")
    bits_per_sample = stream_info.get("bits_per_sample")

    spectrogram_ref = await plot_spectrogram(ctx, path, stream_number)

    return AudioDataSet(
        codec=codec if codec else "UNKNOWN",
        channels=int(channels) if channels else 1,
        duration=float(duration) if duration else None,
        sampleRate=int(sample_rate) if sample_rate else None,
        bitRate=int(bit_rate) if bit_rate else None,
        bitsPerSample=int(bits_per_sample) if bits_per_sample else None,
        spectrogram=spectrogram_ref,
    )


async def plot_spectrogram(ctx: TaskContext, path: Path, stream_number: int) -> PurePosixPath:
    audio_config = ctx.config.audio_config

    sampling_rate, samples = load_audio_samples(path, stream_number)
    window_size = int(2 * sampling_rate / audio_config.lowest_frequency)
    n = len(samples)

    window = scipy.signal.windows.gaussian(window_size, 0.25 * window_size, sym=True)
    sfft = scipy.signal.ShortTimeFFT(window, int(0.1 * window_size), fs=sampling_rate, fft_mode="onesided")
    spectrogram = await asyncio.to_thread(lambda: sfft.spectrogram(samples, padding="odd"))
    spectrogram_log = 10 * np.log10(np.fmax(spectrogram, audio_config.dynamic_range_min))

    plot_name = ctx.build_output_reference("fft")
    async with get_pyplot_writer(ctx, plot_name, figsize=figaspect(0.5)) as (axes, reference):
        t_lo, t_hi = sfft.extent(n)[:2]  # time range of plot
        axes.set_title(f"Spectrogram for {ctx.dataset_name}")
        axes.set(
            xlabel="Time $t$ in seconds " + rf"({sfft.p_num(n)} slices, $\Delta t = {sfft.delta_t:g}\,$s)",
            ylabel="Freq. $f$ in Hz " + rf"({sfft.f_pts} bins, $\Delta f = {sfft.delta_f:g}\,$Hz)",
            xlim=(t_lo, t_hi),
        )
        axes.imshow(spectrogram_log, origin="lower", aspect="auto", extent=sfft.extent(n), cmap="magma")
        return reference


def load_audio_samples(path: Path, stream_number: int) -> tuple[int, ndarray]:
    # Convert to mono WAV format in memory, take nth audio stream.
    stream_mapping = f"0:a:{stream_number}"
    wav_data, _ = (
        ffmpeg.input(path)
        .output("pipe:", format="wav", acodec="pcm_s16le", ar=44100, ac=1, map=stream_mapping)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # RIFF WAV files have an 8 bytes header: marker "RIFF", then content size as 4 bytes in little endian.
    # For ffmpeg piping output the content size is missing. So we fix these 4 bytes.
    size_without_header = len(wav_data) - 8
    wav_data_io = io.BytesIO(wav_data)
    wav_data_io.seek(4)
    size_bytes = int.to_bytes(size_without_header, length=4, byteorder="little")
    wav_data_io.write(size_bytes)
    wav_data_io.seek(0)

    # Read WAV data using SciPy
    sampling_rate, samples = scipy.io.wavfile.read(wav_data_io)
    return sampling_rate, samples
