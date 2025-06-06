import sys

from edps.__main__ import command_line_interface


def test_cli(monkeypatch, path_data_test_csv, path_user_provided_data, path_work):
    with monkeypatch.context() as context:
        mocked_cli_args = [
            sys.argv[0],
            "--input_file",
            str(path_data_test_csv),
            "--user_provided_data",
            str(path_user_provided_data),
            "--output_path",
            str(path_work / "edps.zip"),
        ]
        context.setattr(sys, "argv", mocked_cli_args)
        command_line_interface()
