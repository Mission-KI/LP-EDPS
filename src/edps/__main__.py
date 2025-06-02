from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, CliApp

from edps.service import analyse_asset_to_zip
from edps.types import Config, UserProvidedEdpData


class CommandLineInterface(BaseSettings, cli_parse_args=True, cli_prog_name="edps_cli"):
    input_file: Path = Field(description="Asset or file to analyse")
    user_provided_data: Path = Field(
        description="The user provided portion of the EDP fields. Those are the ones, that can not be inferred by computing the file."
    )
    config_path: Optional[Path] = Field(
        default=None, description="Path to a JSON configuration of this services functions."
    )
    output_path: Path = Field(default=Path("./"), description="File or directory to write the resulting EDP to.")

    async def cli_cmd(self):
        if self.output_path.is_dir():
            self.output_path /= f"{self.input_file.with_suffix('').name}_edp.zip"

        if self.config_path is None:
            config = Config()
        else:
            config = Config.model_validate_json(self.config_path.read_bytes())

        user_provided_data = UserProvidedEdpData.model_validate_json(self.user_provided_data.read_bytes())

        await analyse_asset_to_zip(
            input_file=self.input_file,
            user_data=user_provided_data,
            zip_output=self.output_path,
            config=config,
        )


def command_line_interface():
    CliApp.run(CommandLineInterface)


if __name__ == "__main__":
    command_line_interface()
