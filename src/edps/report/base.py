from abc import ABC, abstractmethod
from io import BufferedIOBase
from pathlib import Path

from extended_dataset_profile import ExtendedDatasetProfile
from pydantic.dataclasses import dataclass

from ..taskcontext import TaskContext


@dataclass
class ReportInput:
    edp: ExtendedDatasetProfile


class ReportGenerator(ABC):
    @abstractmethod
    async def generate(self, ctx: TaskContext, input: ReportInput, base_dir: Path, output_buffer: BufferedIOBase):
        """Generates a report from the given input and writes it to output_buffer.
        Files references in the EDP (plots) are resolved relative to base_dir."""
