from extended_dataset_profile import Resolution, VideoDataSet, VideoPixelFormat

from edps.importers import lookup_importer
from tests.conftest import copy_asset_to_ctx_input_dir


async def test_import_mp4(path_data_test_mp4, ctx):
    path_in_ctx = copy_asset_to_ctx_input_dir(path_data_test_mp4, ctx)
    video_importer = lookup_importer("mp4")
    await ctx.exec("dataset", video_importer, path_in_ctx)
    dataset = next(ctx.collect_datasets())
    assert isinstance(dataset, VideoDataSet)
    assert dataset.codec == "h264"
    assert dataset.resolution == Resolution(width=1280, height=720)
    assert dataset.fps and abs(dataset.fps - 30) < 0.1
    assert dataset.duration and abs(dataset.duration - 30.0) < 0.1
    assert dataset.pixelFormat == VideoPixelFormat.YUV420P
