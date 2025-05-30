import shutil
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

from pontusx.args import get_args
from pontusx.service import run_service

DID = "did230948"


_logger = getLogger(__file__)


async def test_cli(path_data_test_csv, path_data_pontusx_ddo, path_data_pontusx_algocustomdata, monkeypatch):
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        _logger.info("Preparing Pontux-X container data dir: %s", temp_dir)
        # Create dirs
        (temp_dir_path / "ddos").mkdir()
        (temp_dir_path / f"inputs/{DID}").mkdir(parents=True)
        (temp_dir_path / "outputs").mkdir()
        # Copy DDO to DATA/ddos/DID (without file-extension)
        shutil.copy(path_data_pontusx_ddo, temp_dir_path / f"ddos/{DID}")
        # Copy algoCustomData.json to DATA/inputs/algoCustomData.json
        shutil.copy(
            path_data_pontusx_algocustomdata,
            temp_dir_path / "inputs/algoCustomData.json",
        )
        # Copy test.csv to DATA/inputs/0 (without file-extension)
        shutil.copy(
            path_data_test_csv,
            temp_dir_path / f"inputs/{DID}/0",
        )

        monkeypatch.setenv("BASEDIR", temp_dir)
        monkeypatch.setenv("DIDS", f'["{DID}"]')
        args = get_args()
        await run_service(_logger, args)

        # Expect a ZIP file
        files = [file.name for file in (temp_dir_path / "outputs").iterdir() if file.is_file()]
        _logger.info("Got output files: %s", files)
        assert len(files) == 1
        assert files[0].endswith(".zip")
