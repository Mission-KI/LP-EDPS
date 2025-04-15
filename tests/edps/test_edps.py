from datetime import datetime
from math import isclose
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

import pytest
from extended_dataset_profile import (
    AudioDataSet,
    Augmentation,
    DataSetType,
    ExtendedDatasetProfile,
    FileReference,
    ImageColorMode,
    ModificationState,
    Resolution,
    TemporalConsistency,
    VideoPixelFormat,
)
from pytest import fixture, mark, raises

from edps import analyse_asset
from edps.service import _compute_asset
from edps.taskcontext import TaskContext
from edps.taskcontextimpl import TaskContextImpl
from edps.types import (
    AugmentedColumn,
    ComputedEdpData,
    Config,
)
from edps.version import __version__
from tests.conftest import copy_asset_to_ctx_input_dir


@fixture(scope="session")
def config_data_with_augmentations():
    augmented_columns = [
        AugmentedColumn(
            name="aufenthalt",
            augmentation=Augmentation(sourceColumns=["einfahrt", "ausfahrt"]),
        )
    ]
    return Config(augmentedColumns=augmented_columns)


@fixture
def context_with_augmented_config(config_data_with_augmentations, logger, path_work):
    return TaskContextImpl(config_data_with_augmentations, logger, path_work)


@fixture
def analyse_asset_fn(ctx, user_provided_data) -> Callable[[Path], Awaitable[FileReference]]:
    return lambda path: copy_and_analyse_asset(ctx, path, user_provided_data)


@fixture
def compute_asset_fn(ctx) -> Callable[[Path], Awaitable[ComputedEdpData]]:
    return lambda path: copy_and_compute_asset(ctx, path)


@mark.asyncio
async def test_load_unknown_dir(analyse_asset_fn):
    with raises(FileNotFoundError):
        await analyse_asset_fn(Path("/does/not/exist/"))


async def test_analyse_pickle(path_data_test_pickle, context_with_augmented_config, config_data_with_augmentations):
    # Use config with augmentations in this test
    computed_data = await copy_and_compute_asset(context_with_augmented_config, path_data_test_pickle)

    assert len(computed_data.structuredDatasets) == 1
    assert computed_data.periodicity == "hours"

    dataset = computed_data.structuredDatasets[0]
    assert len(dataset.datetimeColumns) == 2
    assert len(dataset.numericColumns) == 2
    assert len(dataset.stringColumns) == 1

    aufenthalt = next(item for item in dataset.numericColumns if item.name == "aufenthalt")
    assert aufenthalt.dataType == "UInt32"
    assert aufenthalt.augmentation == config_data_with_augmentations.augmentedColumns[0].augmentation

    parkhaus = next(item for item in dataset.numericColumns if item.name == "parkhaus")
    assert parkhaus.dataType == "UInt8"

    einfahrt = next(item for item in dataset.datetimeColumns if item.name == "einfahrt")
    assert einfahrt.temporalCover.earliest == datetime.fromisoformat("2016-01-01 00:03:14")
    assert einfahrt.temporalCover.latest == datetime.fromisoformat("2016-01-01 11:50:45")
    _assert_pickle_temporal_consistencies(einfahrt.temporalConsistencies)
    assert einfahrt.periodicity == "hours"

    assert len(computed_data.dataTypes) == 1
    assert DataSetType.structured in computed_data.dataTypes

    assert computed_data.temporalCover is not None
    assert computed_data.temporalCover.earliest == datetime.fromisoformat("2016-01-01 00:03:14")
    assert computed_data.temporalCover.latest == datetime.fromisoformat("2016-01-01 21:13:08")


@mark.asyncio
async def test_analyse_csv(path_data_test_csv, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_csv)

    assert edp.generatedBy == f"EDP Service @ {__version__}"
    assert edp.assetSha256Hash == "fe5a587bb2b8add0773a3722c32a47d95e077296dded1d145cf7c596dae5f1fd"

    assert len(edp.datasetTree) == 1
    ds_root_node = edp.datasetTree[0]
    assert ds_root_node.parent is None
    assert ds_root_node.name == "test_csv"
    assert ds_root_node.fileProperties.name == "test.csv"
    assert ds_root_node.fileProperties.fileType == "csv"
    assert ds_root_node.fileProperties.size == 4307

    assert len(edp.archiveDatasets) == 0
    assert len(edp.structuredDatasets) == 1
    structured_dataset = edp.structuredDatasets[0]
    assert structured_dataset.columnCount == 5
    assert structured_dataset.rowCount == 50
    string_column = structured_dataset.stringColumns[0]
    assert string_column.name == "uuid"
    assert string_column.distributionGraph is not None
    assert structured_dataset.datetimeColumns[0].name == "einfahrt"
    assert structured_dataset.datetimeColumns[0].format == "ISO8601"
    assert structured_dataset.datetimeColumns[1].name == "ausfahrt"
    assert structured_dataset.datetimeColumns[1].format == "ISO8601"

    aufenthalt = structured_dataset.numericColumns[0]
    assert aufenthalt.name == "aufenthalt"
    assert aufenthalt.dataType == "UInt32"
    assert aufenthalt.percentileOutlierCount == 2
    assert aufenthalt.zScoreOutlierCount == 2
    assert aufenthalt.iqrOutlierCount == 4
    assert isclose(aufenthalt.relativeOutlierCount, 0.0533333333, rel_tol=1e-6)

    parkhaus = structured_dataset.numericColumns[1]
    assert parkhaus.name == "parkhaus"
    assert parkhaus.dataType == "UInt8"
    assert parkhaus.percentileOutlierCount == 0
    assert parkhaus.zScoreOutlierCount == 0
    assert parkhaus.iqrOutlierCount == 0
    assert isclose(parkhaus.relativeOutlierCount, 0.0)


@mark.asyncio
async def test_analyse_csv_counts(path_data_test_counts_csv, compute_asset_fn):
    with pytest.warns(UserWarning) as records:
        edp = await compute_asset_fn(path_data_test_counts_csv)
    assert len(edp.structuredDatasets) == 1
    dataset = edp.structuredDatasets[0]
    assert dataset.rowCount == 7
    assert len(dataset.stringColumns) == 1
    assert len(dataset.datetimeColumns) == 0
    assert len(dataset.numericColumns) == 1
    string_column = dataset.stringColumns[0]
    assert string_column.inconsistentCount == 0
    assert string_column.interpretableCount == 7
    assert string_column.nullCount == 2
    numeric_column = dataset.numericColumns[0]
    assert numeric_column.inconsistentCount == 1
    assert numeric_column.interpretableCount == 5
    assert numeric_column.nullCount == 1

    assert len(records) == 2
    assert str(records[0].message) == "Column 'col001' contains 1 empty of 7 overall entries."
    assert (
        str(records[1].message)
        == "Couldn't convert some entries of column 'col001' using NumericColumnConverter (1 invalid of 7 overall entries)."
    )


@mark.asyncio
async def test_analyse_roundtrip_csv(path_data_test_csv, analyse_asset_fn, ctx, user_provided_data):
    edp_file = await analyse_asset_fn(path_data_test_csv)
    edp = read_edp_file(ctx.output_path / edp_file)
    assert edp.assetRefs[0].assetId == user_provided_data.assetRefs[0].assetId
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_seasonality_bast_csv(path_data_bast_csv, analyse_asset_fn, ctx):
    edp_file = await analyse_asset_fn(path_data_bast_csv)
    edp = read_edp_file(ctx.output_path / edp_file)
    assert len(edp.structuredDatasets) == 1
    dataset = edp.structuredDatasets[0]
    columns_with_seasonality = list(filter(lambda col: len(col.seasonalities) > 0, dataset.numericColumns))
    assert len(columns_with_seasonality) == 10


@mark.asyncio
async def test_analyse_csv_no_headers(path_data_test_headerless_csv, ctx):
    edp = await copy_and_compute_asset(ctx, path_data_test_headerless_csv)
    assert len(edp.archiveDatasets) == 0
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "col000"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "col001"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "ISO8601"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "col002"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "ISO8601"
    assert edp.structuredDatasets[0].numericColumns[0].name == "col003"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "col004"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_xlsx(path_data_test_xlsx, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_xlsx)
    assert len(edp.archiveDatasets) == 0
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "uuid"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "einfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "NATIVE"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "ausfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "NATIVE"
    assert edp.structuredDatasets[0].numericColumns[0].name == "aufenthalt"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "parkhaus"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_xls(path_data_test_xls, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_xls)
    assert len(edp.archiveDatasets) == 0
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50
    assert edp.structuredDatasets[0].stringColumns[0].name == "uuid"
    assert edp.structuredDatasets[0].datetimeColumns[0].name == "einfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[0].format == "NATIVE"
    assert edp.structuredDatasets[0].datetimeColumns[1].name == "ausfahrt"
    assert edp.structuredDatasets[0].datetimeColumns[1].format == "NATIVE"
    assert edp.structuredDatasets[0].numericColumns[0].name == "aufenthalt"
    assert edp.structuredDatasets[0].numericColumns[0].dataType == "UInt32"
    assert edp.structuredDatasets[0].numericColumns[1].name == "parkhaus"
    assert edp.structuredDatasets[0].numericColumns[1].dataType == "UInt8"


@mark.asyncio
async def test_analyse_zip(path_data_test_zip, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_zip)

    assert len(edp.archiveDatasets) == 1
    archive = edp.archiveDatasets[0]
    assert archive.algorithm == "zip"

    assert len(edp.structuredDatasets) == 1
    structured = edp.structuredDatasets[0]
    assert structured.columnCount == 5
    assert structured.rowCount == 50


@mark.asyncio
async def test_analyse_multiassets_zip(path_data_test_multiassets_zip, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_multiassets_zip)

    assert len(edp.datasetTree) == 6
    ds_root_node = edp.datasetTree[0]
    assert ds_root_node.parent is None
    assert ds_root_node.name == "test_multiassets_zip"
    assert ds_root_node.fileProperties.name == "test_multiassets.zip"
    assert ds_root_node.fileProperties.fileType == "zip"

    assert len(edp.archiveDatasets) == 2
    for archive in edp.archiveDatasets:
        assert archive.algorithm == "zip"

    assert len(edp.structuredDatasets) == 4
    assert edp.structuredDatasets[0].columnCount == 5
    assert edp.structuredDatasets[0].rowCount == 50


@mark.asyncio
async def test_analyse_png(ctx, path_data_test_png):
    edp = await copy_and_compute_asset(ctx, path_data_test_png)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "PNG"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.PALETTED
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert 90 <= edp.imageDatasets[0].dpi.x <= 100
    assert 90 <= edp.imageDatasets[0].dpi.y <= 100
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_jpg(ctx, path_data_test_jpg):
    edp = await copy_and_compute_asset(ctx, path_data_test_jpg)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "JPEG"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.RGB
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert 90 <= edp.imageDatasets[0].dpi.x <= 100
    assert 90 <= edp.imageDatasets[0].dpi.y <= 100
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_jpeg(ctx, path_data_test_jpeg):
    edp = await copy_and_compute_asset(ctx, path_data_test_jpeg)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "JPEG"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.GRAYSCALE
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert 90 <= edp.imageDatasets[0].dpi.x <= 100
    assert 90 <= edp.imageDatasets[0].dpi.y <= 100
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_gif(ctx, path_data_test_gif):
    edp = await copy_and_compute_asset(ctx, path_data_test_gif)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "GIF"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.PALETTED
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert edp.imageDatasets[0].dpi is None
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_bmp(ctx, path_data_test_bmp):
    edp = await copy_and_compute_asset(ctx, path_data_test_bmp)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "BMP"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.PALETTED
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert 90 <= edp.imageDatasets[0].dpi.x <= 100
    assert 90 <= edp.imageDatasets[0].dpi.y <= 100
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_tiff(ctx, path_data_test_tiff):
    edp = await copy_and_compute_asset(ctx, path_data_test_tiff)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "TIFF"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.GRAYSCALE
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert 90 <= edp.imageDatasets[0].dpi.x <= 100
    assert 90 <= edp.imageDatasets[0].dpi.y <= 100
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_tif(ctx, path_data_test_tif):
    edp = await copy_and_compute_asset(ctx, path_data_test_tif)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "TIFF"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.RGB
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert 90 <= edp.imageDatasets[0].dpi.x <= 100
    assert 90 <= edp.imageDatasets[0].dpi.y <= 100
    _assert_image_pixel_metrics(edp.imageDatasets[0])


@mark.asyncio
async def test_analyse_webp(ctx, path_data_test_webp):
    edp = await copy_and_compute_asset(ctx, path_data_test_webp)
    assert len(edp.archiveDatasets) == 0
    assert edp.imageDatasets[0].codec == "WEBP"
    assert edp.imageDatasets[0].colorMode == ImageColorMode.RGB
    assert edp.imageDatasets[0].resolution == Resolution(width=600, height=400)
    assert edp.imageDatasets[0].dpi is None
    _assert_image_pixel_metrics(edp.imageDatasets[0])


def _assert_image_pixel_metrics(image_dataset):
    expected_brightness = 2.8
    expected_blurriness = 660
    expected_sharpness = 3.3
    expected_brisque = 100
    expected_ela_score = 13.6

    assert abs(image_dataset.brightness - expected_brightness) < 0.1
    assert expected_blurriness * 0.9 <= image_dataset.blurriness <= expected_blurriness * 1.1
    assert expected_sharpness * 0.9 <= image_dataset.sharpness <= expected_sharpness * 1.1
    assert expected_brisque * 0.9 <= image_dataset.brisque <= expected_brisque * 1.1
    assert 0.8 <= image_dataset.noise <= 12
    assert not image_dataset.lowContrast

    if image_dataset.codec in ("JPG", "JPEG"):
        assert abs(image_dataset.elaScore - expected_ela_score) < 0.1
    else:
        assert image_dataset.elaScore is None


# Videos


@mark.asyncio
async def test_analyse_mp4(ctx, path_data_test_mp4):
    edp = await copy_and_compute_asset(ctx, path_data_test_mp4)
    assert len(edp.videoDatasets) == 1
    assert len(edp.audioDatasets) == 2
    video_dataset = edp.videoDatasets[0]
    assert edp.datasetTree[0].parent is None
    assert edp.datasetTree[0].dataset.reference == "#/videoDatasets/0"
    assert edp.datasetTree[1].parent.reference == "#/datasetTree/0"
    assert edp.datasetTree[1].dataset.reference == "#/audioDatasets/0"
    assert edp.datasetTree[2].parent.reference == "#/datasetTree/0"
    assert edp.datasetTree[2].dataset.reference == "#/audioDatasets/1"
    assert len(edp.archiveDatasets) == 0
    assert video_dataset.codec == "h264"
    assert video_dataset.resolution == Resolution(width=1280, height=720)
    assert abs(video_dataset.fps - 30) < 0.1
    assert abs(video_dataset.duration - 30.0) < 0.1
    assert video_dataset.pixelFormat == VideoPixelFormat.YUV420P
    assert abs(edp.audioDatasets[0].duration - 25.5) < 0.1
    assert abs(edp.audioDatasets[1].duration - 8.5) < 0.1


@mark.asyncio
async def test_analyse_avi(ctx, path_data_test_avi):
    edp = await copy_and_compute_asset(ctx, path_data_test_avi)
    assert len(edp.videoDatasets) == 1
    assert len(edp.audioDatasets) == 2
    video_dataset = edp.videoDatasets[0]
    assert len(edp.archiveDatasets) == 0
    assert video_dataset.codec == "mpeg4"
    assert video_dataset.resolution == Resolution(width=1280, height=720)
    assert abs(video_dataset.fps - 30) < 0.1
    assert abs(video_dataset.duration - 30.0) < 0.1
    assert video_dataset.pixelFormat == VideoPixelFormat.YUV420P
    assert abs(edp.audioDatasets[0].duration - 25.5) < 0.1
    assert abs(edp.audioDatasets[1].duration - 8.5) < 0.1


@mark.asyncio
async def test_analyse_mkv(ctx, path_data_test_mkv):
    edp = await copy_and_compute_asset(ctx, path_data_test_mkv)
    assert len(edp.videoDatasets) == 1
    assert len(edp.audioDatasets) == 2
    video_dataset = edp.videoDatasets[0]
    assert len(edp.archiveDatasets) == 0
    assert video_dataset.codec == "h264"
    assert video_dataset.resolution == Resolution(width=1280, height=720)
    assert abs(video_dataset.fps - 30) < 0.1
    assert abs(video_dataset.duration - 30.0) < 0.1
    assert video_dataset.pixelFormat == VideoPixelFormat.YUV420P
    assert abs(edp.audioDatasets[0].duration - 30.0) < 0.1
    assert abs(edp.audioDatasets[1].duration - 30.0) < 0.1


@mark.asyncio
async def test_analyse_mov(ctx, path_data_test_mov):
    edp = await copy_and_compute_asset(ctx, path_data_test_mov)
    assert len(edp.videoDatasets) == 1
    assert len(edp.audioDatasets) == 2
    video_dataset = edp.videoDatasets[0]
    assert len(edp.archiveDatasets) == 0
    assert video_dataset.codec == "h264"
    assert video_dataset.resolution == Resolution(width=1280, height=720)
    assert abs(video_dataset.fps - 30) < 0.1
    assert abs(video_dataset.duration - 30.0) < 0.1
    assert video_dataset.pixelFormat == VideoPixelFormat.YUV420P
    assert abs(edp.audioDatasets[0].duration - 25.5) < 0.1
    assert abs(edp.audioDatasets[1].duration - 8.5) < 0.1


@mark.asyncio
async def test_analyse_flv(ctx, path_data_test_flv):
    edp = await copy_and_compute_asset(ctx, path_data_test_flv)
    assert len(edp.videoDatasets) == 1
    assert len(edp.audioDatasets) == 1
    video_dataset = edp.videoDatasets[0]
    assert len(edp.archiveDatasets) == 0
    assert video_dataset.codec == "flv1"
    assert video_dataset.resolution == Resolution(width=1280, height=720)
    assert abs(video_dataset.fps - 30) < 0.1
    assert abs(video_dataset.duration - 30.0) < 0.1
    assert video_dataset.pixelFormat == VideoPixelFormat.YUV420P
    assert abs(edp.audioDatasets[0].duration - 30.0) < 0.1


@mark.asyncio
async def test_analyse_wmv(ctx, path_data_test_wmv):
    edp = await copy_and_compute_asset(ctx, path_data_test_wmv)
    assert len(edp.videoDatasets) == 1
    assert len(edp.audioDatasets) == 2
    video_dataset = edp.videoDatasets[0]
    assert len(edp.archiveDatasets) == 0
    assert video_dataset.codec == "wmv2"
    assert video_dataset.resolution == Resolution(width=1280, height=720)
    assert abs(video_dataset.fps - 30) < 0.1
    assert abs(video_dataset.duration - 30.0) < 0.1
    assert video_dataset.pixelFormat == VideoPixelFormat.YUV420P
    assert abs(edp.audioDatasets[0].duration - 30.0) < 0.1
    assert abs(edp.audioDatasets[1].duration - 30.0) < 0.1


# Audio files


@mark.asyncio
async def test_analyse_wav(ctx, path_data_test_wav):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_wav,
        exp_codec="pcm_s16le",
        exp_sample_rate=44100,
        exp_bit_rate=1411200,
        exp_bits_per_sample=16,
    )


@mark.asyncio
async def test_analyse_aac(ctx, path_data_test_aac):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_aac,
        exp_codec="aac",
        exp_sample_rate=44100,
        exp_bit_rate=126325,
        exp_bits_per_sample=None,
    )


@mark.asyncio
async def test_analyse_flac(ctx, path_data_test_flac):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_flac,
        exp_codec="flac",
        exp_sample_rate=44100,
        exp_bit_rate=None,
        exp_bits_per_sample=None,
    )


@mark.asyncio
async def test_analyse_m4a(ctx, path_data_test_m4a):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_m4a,
        exp_codec="aac",
        exp_sample_rate=48000,
        exp_bit_rate=192029,
        exp_bits_per_sample=None,
    )


@mark.asyncio
async def test_analyse_mp3(ctx, path_data_test_mp3):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_mp3,
        exp_codec="mp3",
        exp_sample_rate=48000,
        exp_bit_rate=206986,
        exp_bits_per_sample=None,
    )


@mark.asyncio
async def test_analyse_ogg(ctx, path_data_test_ogg):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_ogg,
        exp_codec="vorbis",
        exp_sample_rate=44100,
        exp_bit_rate=160000,
        exp_bits_per_sample=None,
    )


@mark.asyncio
async def test_analyse_opus(ctx, path_data_test_opus):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_opus,
        exp_codec="opus",
        exp_sample_rate=48000,
        exp_bit_rate=None,
        exp_bits_per_sample=None,
    )


@mark.asyncio
async def test_analyse_wma(ctx, path_data_test_wma):
    await analyse_audio(
        ctx=ctx,
        path=path_data_test_wma,
        exp_codec="wmav2",
        exp_sample_rate=44100,
        exp_bit_rate=192000,
        exp_bits_per_sample=None,
    )


async def analyse_audio(
    ctx: TaskContext,
    path: Path,
    exp_codec: str,
    exp_sample_rate: int,
    exp_bit_rate: Optional[int],
    exp_bits_per_sample: Optional[int],
):
    edp = await copy_and_compute_asset(ctx, path)
    assert len(edp.videoDatasets) == 0
    assert len(edp.audioDatasets) == 1
    audio_dataset: AudioDataSet = edp.audioDatasets[0]
    assert audio_dataset.channels == 2
    assert audio_dataset.duration and abs(audio_dataset.duration - 8.5) < 0.5
    assert audio_dataset.spectrogram.suffix == ".png"
    assert audio_dataset.codec == exp_codec
    assert audio_dataset.sampleRate == exp_sample_rate
    assert audio_dataset.bitRate == exp_bit_rate
    assert audio_dataset.bitsPerSample == exp_bits_per_sample


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_analyse_pdf(path_data_test_pdf, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_pdf)

    assert len(edp.documentDatasets) == 1
    doc_dataset = edp.documentDatasets[0]
    assert doc_dataset.docType == "PDF-1.7"
    assert doc_dataset.title == "Vornamen von Neugeborenen in der Stadt Aachen 2021"
    assert doc_dataset.subject == "Vornamen-Statistik"
    assert doc_dataset.toolchain == "Microsoft® Word für Microsoft 365"
    assert doc_dataset.keywords == ["Vornamen", "Neugeborene", "Aachen", "2021"]
    assert doc_dataset.numPages == 18
    assert doc_dataset.numImages == 2
    assert doc_dataset.modified == ModificationState.unmodified

    assert len(edp.imageDatasets) == 2
    assert len(edp.unstructuredTextDatasets) == 1
    assert len(edp.structuredDatasets) == 0


async def test_unstructured_text_without_table(path_unstructured_text_only_txt, compute_asset_fn):
    edp = await compute_asset_fn(path_unstructured_text_only_txt)
    assert len(edp.unstructuredTextDatasets) == 1
    unstructured_dataset = edp.unstructuredTextDatasets[0]
    assert len(unstructured_dataset.languages) == 2
    assert "deu" in unstructured_dataset.languages
    assert "eng" in unstructured_dataset.languages
    assert unstructured_dataset.wordCount == 12
    assert unstructured_dataset.lineCount == 3


async def test_unstructured_text_with_table(path_unstructured_text_with_table, compute_asset_fn):
    edp = await compute_asset_fn(path_unstructured_text_with_table)
    assert len(edp.structuredDatasets) == 1
    assert len(edp.unstructuredTextDatasets) == 1
    structured_dataset = edp.structuredDatasets[0]
    assert structured_dataset.columnCount == 3
    assert structured_dataset.rowCount == 2
    headers = [column.name for column in structured_dataset.all_columns]
    assert "id" in headers
    assert "name" in headers
    assert "width" in headers
    assert structured_dataset.all_columns
    assert structured_dataset.numericColumnCount == 2
    assert structured_dataset.stringColumnCount == 1
    assert structured_dataset.datetimeColumnCount == 0
    unstructured_dataset = edp.unstructuredTextDatasets[0]
    assert len(unstructured_dataset.embeddedTables) == 1
    assert len(unstructured_dataset.languages) == 1
    assert "eng" in unstructured_dataset.languages
    assert unstructured_dataset.wordCount == 20
    assert unstructured_dataset.lineCount == 2


@mark.asyncio
async def test_analyse_semi_structured_json(path_data_test_json, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_json)

    assert len(edp.semiStructuredDatasets) == 1
    assert edp.semiStructuredDatasets[0].jsonSchema is not None

    assert len(edp.structuredDatasets) == 3
    table1_ds = edp.structuredDatasets[0]
    table1_headers = [col.name for col in table1_ds.all_columns]
    assert set(table1_headers) == {"id", "details", "extras", "type", "name", "ppu", "topping", "batters.batter"}
    assert table1_ds.rowCount == 2
    assert table1_ds.columnCount == 8
    assert table1_ds.numericColumnCount == 2
    assert table1_ds.stringColumnCount == 6
    assert table1_ds.datetimeColumnCount == 0
    assert table1_ds.numericColumns[1].name == "ppu"
    assert table1_ds.numericColumns[1].dataType == "Float32"
    assert table1_ds.numericColumns[1].numberUnique == 2
    assert abs(table1_ds.numericColumns[1].min - 0.55) < 0.01
    assert abs(table1_ds.numericColumns[1].max - 0.70) < 0.01
    assert abs(table1_ds.numericColumns[1].mean - 0.625) < 0.001

    table2_ds = edp.structuredDatasets[1]
    table2_headers = [col.name for col in table2_ds.all_columns]
    assert set(table2_headers) == {"id", "type"}
    assert table2_ds.rowCount == 3
    assert table2_ds.columnCount == 2
    assert table2_ds.numericColumnCount == 1
    assert table2_ds.stringColumnCount == 1
    assert table2_ds.datetimeColumnCount == 0
    assert table2_ds.stringColumns[0].name == "type"
    assert table2_ds.stringColumns[0].interpretableCount == 3
    assert table2_ds.stringColumns[0].numberUnique == 2

    table3_ds = edp.structuredDatasets[2]
    table3_headers = [col.name for col in table3_ds.all_columns]
    assert set(table3_headers) == {"id", "type"}
    assert table3_ds.rowCount == 4
    assert table3_ds.columnCount == 2
    assert table3_ds.numericColumnCount == 1
    assert table3_ds.stringColumnCount == 1
    assert table3_ds.datetimeColumnCount == 0
    assert table3_ds.stringColumns[0].name == "type"
    assert table3_ds.stringColumns[0].interpretableCount == 4
    assert table3_ds.stringColumns[0].numberUnique == 3


@mark.asyncio
async def test_analyse_json_without_structured_data(path_data_test_without_structured_data, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_without_structured_data)

    assert len(edp.semiStructuredDatasets) == 1
    assert edp.semiStructuredDatasets[0].jsonSchema is not None

    assert len(edp.structuredDatasets) == 0


@mark.asyncio
async def test_analyse_json_with_normalization(path_data_test_with_normalization, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_with_normalization)

    assert len(edp.semiStructuredDatasets) == 1
    assert edp.semiStructuredDatasets[0].jsonSchema is not None

    assert len(edp.structuredDatasets) == 2
    table1_ds = edp.structuredDatasets[0]
    table1_headers = [col.name for col in table1_ds.all_columns]
    assert set(table1_headers) == {
        "_id",
        "index",
        "guid",
        "isActive",
        "balance",
        "picture",
        "age",
        "eyeColor",
        "name",
        "gender",
        "company",
        "email",
        "phone",
        "address",
        "about",
        "registered",
        "geoLocation.latitude",
        "geoLocation.longitude",
        "tags",
        "friends",
        "greeting",
        "favoriteFruit",
    }
    assert table1_ds.numericColumns[3].name == "geoLocation.latitude"
    assert table1_ds.numericColumns[3].interpretableCount == 15
    assert table1_ds.numericColumns[3].numberUnique == 15
    assert abs(table1_ds.numericColumns[3].min - -80.34) < 0.01
    assert abs(table1_ds.numericColumns[3].max - 77.75) < 0.01
    assert abs(table1_ds.numericColumns[3].mean - -1.25) < 0.01

    assert table1_ds.numericColumns[4].name == "geoLocation.longitude"
    assert table1_ds.numericColumns[4].interpretableCount == 15
    assert table1_ds.numericColumns[4].numberUnique == 15
    assert abs(table1_ds.numericColumns[4].min - -134.89) < 0.01
    assert abs(table1_ds.numericColumns[4].max - 143.67) < 0.01
    assert abs(table1_ds.numericColumns[4].mean - -14.18) < 0.01

    table2_ds = edp.structuredDatasets[1]
    table2_headers = [col.name for col in table2_ds.all_columns]
    assert set(table2_headers) == {"id", "name"}


async def test_analyse_docx(path_data_test_docx, compute_asset_fn):
    edp = await compute_asset_fn(path_data_test_docx)

    assert len(edp.documentDatasets) == 1
    doc_ds = edp.documentDatasets[0]
    assert doc_ds.title == "Vornamen von Neugeborenen in der Stadt Aachen 2021"
    assert doc_ds.keywords == ["Vornamen", "Neugeborene", "Aachen", "2021"]
    assert doc_ds.numImages == 2
    assert doc_ds.numPages is None  # Can't determine pages of DOCX yet!

    assert len(edp.imageDatasets) == 2
    image1_ds = edp.imageDatasets[0]
    assert image1_ds.codec == "JPEG"
    image2_ds = edp.imageDatasets[1]
    assert image2_ds.codec == "PNG"

    assert len(edp.structuredDatasets) == 1
    table_ds = edp.structuredDatasets[0]
    headers = [col.name for col in table_ds.all_columns]
    assert set(headers) == {"anzahl", "vorname", "geschlecht", "position"}
    assert table_ds.rowCount == 388
    assert table_ds.columnCount == 4

    assert len(edp.unstructuredTextDatasets) == 1
    text_ds = edp.unstructuredTextDatasets[0]
    assert text_ds.lineCount == 3
    assert text_ds.wordCount == 30


@mark.asyncio
async def test_raise_on_only_unknown_datasets(analyse_asset_fn, tmp_path):
    file_path = tmp_path / "unsupported.unsupported"
    with open(file_path, "wt", encoding="utf-8") as file:
        file.write("This type is not supported")
    with raises((RuntimeWarning, RuntimeError)):
        await analyse_asset_fn(file_path)


async def copy_and_analyse_asset(ctx: TaskContext, asset_path: Path, user_provided_data):
    copy_asset_to_ctx_input_dir(asset_path, ctx)
    return await analyse_asset(ctx, user_provided_data)


async def copy_and_compute_asset(ctx: TaskContext, asset_path: Path):
    copy_asset_to_ctx_input_dir(asset_path, ctx)
    return await _compute_asset(ctx)


def _assert_pickle_temporal_consistencies(
    temporal_consistencies: List[TemporalConsistency],
):
    microseconds_consistency = temporal_consistencies[0]
    assert microseconds_consistency.timeScale == "microseconds"
    assert microseconds_consistency.stable is False
    assert microseconds_consistency.differentAbundancies == 50
    assert microseconds_consistency.numberOfGaps == 49

    milliseconds_consistency = temporal_consistencies[1]
    assert milliseconds_consistency.timeScale == "milliseconds"
    assert milliseconds_consistency.stable is False
    assert milliseconds_consistency.differentAbundancies == 50
    assert milliseconds_consistency.numberOfGaps == 49

    seconds_consistency = temporal_consistencies[2]
    assert seconds_consistency.timeScale == "seconds"
    assert seconds_consistency.stable is False
    assert seconds_consistency.differentAbundancies == 50
    assert seconds_consistency.numberOfGaps == 48

    minutes_consistency = temporal_consistencies[3]
    assert minutes_consistency.timeScale == "minutes"
    assert minutes_consistency.stable is False
    assert minutes_consistency.differentAbundancies == 47
    assert minutes_consistency.numberOfGaps == 44

    hours_consistency = temporal_consistencies[4]
    assert hours_consistency.timeScale == "hours"
    assert hours_consistency.stable is False
    assert hours_consistency.differentAbundancies == 12
    assert hours_consistency.numberOfGaps == 3

    days_consistency = temporal_consistencies[5]
    assert days_consistency.timeScale == "days"
    assert days_consistency.stable is True
    assert days_consistency.differentAbundancies == 1
    assert days_consistency.numberOfGaps == 0

    weeks_consistency = temporal_consistencies[6]
    assert weeks_consistency.timeScale == "weeks"
    assert weeks_consistency.stable is True
    assert weeks_consistency.differentAbundancies == 1
    assert weeks_consistency.numberOfGaps == 0

    month_consistency = temporal_consistencies[7]
    assert month_consistency.timeScale == "months"
    assert month_consistency.stable is True
    assert month_consistency.differentAbundancies == 1
    assert month_consistency.numberOfGaps == 0

    year_consistency = temporal_consistencies[8]
    assert year_consistency.timeScale == "years"
    assert year_consistency.stable is True
    assert year_consistency.differentAbundancies == 1
    assert year_consistency.numberOfGaps == 0


def read_edp_file(json_file: Path):
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = file.read()
    return ExtendedDatasetProfile.model_validate_json(json_data)
