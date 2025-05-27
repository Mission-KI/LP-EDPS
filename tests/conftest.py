import os
import pickle
import shutil
from io import FileIO
from logging import getLogger
from pathlib import Path
from typing import Awaitable, Callable

from extended_dataset_profile.models.v1.edp import ExtendedDatasetProfile
from pytest import fixture

from edps.analyzers.structured.importer import csv_import_dataframe
from edps.service import analyse_asset
from edps.taskcontext import TaskContext
from edps.taskcontextimpl import TaskContextImpl
from edps.types import Config, UserProvidedEdpData

TESTS_ROOT_PATH = Path(__file__).parent.absolute()


@fixture
def path_work(tmp_path):
    """This is the path to the working directory. Change this to the following code, to review the results in a directory:

    Example:
        path = TESTS_ROOT_PATH / "work"
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        path.mkdir()
        return path
    """
    return tmp_path


@fixture
def path_data_test_csv():
    return TESTS_ROOT_PATH / "data/test.csv"


@fixture(scope="session")
def path_data_test_counts_csv():
    return TESTS_ROOT_PATH / "data/test_counts.csv"


@fixture
def path_data_bast_csv():
    return TESTS_ROOT_PATH / "data/bast.csv"


@fixture
def path_data_test_extra_headers_csv():
    return TESTS_ROOT_PATH / "data/test_extra_headers.csv"


@fixture
def path_data_test_missing_headers_csv():
    return TESTS_ROOT_PATH / "data/test_missing_headers.csv"


@fixture
def path_data_test_headerless_csv():
    return TESTS_ROOT_PATH / "data/test_headerless.csv"


@fixture
def path_data_test_xls():
    return TESTS_ROOT_PATH / "data/test.xls"


@fixture
def path_data_german_decimal_comma_csv():
    return TESTS_ROOT_PATH / "data/german_decimal_comma.csv"


@fixture
def path_data_hamburg_csv():
    return TESTS_ROOT_PATH / "data/hamburg.csv"


@fixture
def path_data_test_xlsx():
    return TESTS_ROOT_PATH / "data/test.xlsx"


@fixture
def path_data_test_zip():
    return TESTS_ROOT_PATH / "data/test.zip"


@fixture
def path_data_test_pdf():
    return TESTS_ROOT_PATH / "data/test.pdf"


@fixture
def path_data_test_docx():
    return TESTS_ROOT_PATH / "data/test.docx"


@fixture
def path_data_test_multiassets_zip():
    return TESTS_ROOT_PATH / "data/test_multiassets.zip"


# images


@fixture
def path_data_test_png():
    return TESTS_ROOT_PATH / "data/test.png"


@fixture
def path_data_test_jpg():
    return TESTS_ROOT_PATH / "data/test.jpg"


@fixture
def path_data_test_jpeg():
    return TESTS_ROOT_PATH / "data/test.jpeg"


@fixture
def path_data_test_gif():
    return TESTS_ROOT_PATH / "data/test.gif"


@fixture
def path_data_test_bmp():
    return TESTS_ROOT_PATH / "data/test.bmp"


@fixture
def path_data_test_tiff():
    return TESTS_ROOT_PATH / "data/test.tiff"


@fixture
def path_data_test_tif():
    return TESTS_ROOT_PATH / "data/test.tif"


@fixture
def path_data_test_webp():
    return TESTS_ROOT_PATH / "data/test.webp"


# videos


@fixture
def path_data_test_mp4():
    return TESTS_ROOT_PATH / "data/test.mp4"


@fixture
def path_data_test_avi():
    return TESTS_ROOT_PATH / "data/test.avi"


@fixture
def path_data_test_mkv():
    return TESTS_ROOT_PATH / "data/test.mkv"


@fixture
def path_data_test_mov():
    return TESTS_ROOT_PATH / "data/test.mov"


@fixture
def path_data_test_flv():
    return TESTS_ROOT_PATH / "data/test.flv"


@fixture
def path_data_test_wmv():
    return TESTS_ROOT_PATH / "data/test.wmv"


# audio files


@fixture
def path_data_test_wav():
    return TESTS_ROOT_PATH / "data/test.wav"


@fixture
def path_data_test_aac():
    return TESTS_ROOT_PATH / "data/test.aac"


@fixture
def path_data_test_flac():
    return TESTS_ROOT_PATH / "data/test.flac"


@fixture
def path_data_test_m4a():
    return TESTS_ROOT_PATH / "data/test.m4a"


@fixture
def path_data_test_mp3():
    return TESTS_ROOT_PATH / "data/test.mp3"


@fixture
def path_data_test_ogg():
    return TESTS_ROOT_PATH / "data/test.ogg"


@fixture
def path_data_test_opus():
    return TESTS_ROOT_PATH / "data/test.opus"


@fixture
def path_data_test_wma():
    return TESTS_ROOT_PATH / "data/test.wma"


@fixture
def path_data_test_with_text():
    return TESTS_ROOT_PATH / "data/test_with_text.png"


@fixture
def path_data_pontusx_algocustomdata():
    return TESTS_ROOT_PATH / "data/pontusx/algoCustomData.json"


@fixture
def path_data_pontusx_ddo():
    return TESTS_ROOT_PATH / "data/pontusx/ddo.json"


@fixture
def path_unstructured_text_only_txt():
    return TESTS_ROOT_PATH / "data/unstructured_text_only.txt"


@fixture
def path_unstructured_text_with_table():
    return TESTS_ROOT_PATH / "data/unstructured_text_with_table.txt"


@fixture
def path_data_test_json():
    return TESTS_ROOT_PATH / "data/test.json"


@fixture
def path_data_test_without_structured_data():
    return TESTS_ROOT_PATH / "data/test_without_structured_data.json"


@fixture
def path_data_test_with_normalization():
    return TESTS_ROOT_PATH / "data/test_with_normalization.json"


ASSET_FILES = [
    "data/bast.csv",
    "data/test.csv",
    "data/test.json",
    "data/test.docx",
    "data/test.pdf",
    "data/test.jpg",
    "data/test.mp4",
    "data/test.mp3",
    "data/test.m4a",
    "data/unstructured_text_with_table.txt",
    "data/test_multiassets.zip",
]


# "asset_path" iterates through multiple assets
@fixture(params=ASSET_FILES)
def asset_path(request):
    return TESTS_ROOT_PATH / request.param


@fixture
def path_language_deu_wiki_llm_txt():
    return TESTS_ROOT_PATH / "data/language/deu_wiki_llm.txt"


@fixture
def path_language_deu_eng_wiki_llm_txt():
    return TESTS_ROOT_PATH / "data/language/deu_eng_wiki_llm.txt"


@fixture(scope="session")
def path_user_provided_data():
    return TESTS_ROOT_PATH / "data/user_provided_data.json"


@fixture(scope="session")
def user_provided_data(path_user_provided_data):
    return UserProvidedEdpData.model_validate_json(path_user_provided_data.read_bytes())


@fixture(scope="session")
def config_data():
    return Config()


@fixture(scope="session")
def logger():
    return getLogger("edps.test")


@fixture
def ctx(path_work, config_data, logger) -> TaskContext:
    logger.info("Creating test TaskContext with working directory '%s'", path_work.as_posix())
    return TaskContextImpl(config_data, logger, path_work)


def copy_asset_to_ctx_input_dir(asset_path: Path, ctx: TaskContext):
    shutil.rmtree(ctx.input_path, ignore_errors=True)
    ctx.input_path.mkdir()
    dest_path = ctx.input_path / asset_path.name
    shutil.copy(asset_path, dest_path)
    return dest_path


@fixture
async def path_data_test_pickle(ctx, path_data_test_csv, tmp_path):
    # Freshly pickle the dataframe from the CSV
    dataframe = await csv_import_dataframe(ctx, path_data_test_csv)
    pickle_path = tmp_path / "test.pickle"
    with open(pickle_path, "wb") as file:  # Use "wb" mode to write in binary
        pickle.dump(dataframe, file)
    return pickle_path


@fixture(scope="session")
def null_dev():
    with FileIO(os.devnull, mode="w") as dev:
        yield dev


@fixture(scope="session")
def analyse_asset_fn(null_dev, user_provided_data, logger) -> Callable[[Path], Awaitable[ExtendedDatasetProfile]]:
    return lambda path: analyse_asset(path, null_dev, user_provided_data, logger=logger)
