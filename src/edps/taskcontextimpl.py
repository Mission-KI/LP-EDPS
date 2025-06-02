import asyncio
import warnings
from contextlib import contextmanager
from logging import Logger
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Awaitable, Callable, Concatenate, Iterator, Optional, Unpack, cast

from extended_dataset_profile import FileProperties

from edps.file import build_real_sub_path, determine_file_type, sanitize_path
from edps.importers import lookup_importer, lookup_unsupported_type_message
from edps.taskcontext import TaskContext
from edps.types import Config, DataSet, is_dataset


class TaskContextImpl(TaskContext):
    """A context provides a logger and supports executing sub-tasks."""

    def __init__(self, config: Config, logger: Logger, base_path: Path, name_parts: list[str] = []):
        self._config = config
        self._logger = logger
        self._name_parts = name_parts
        self._children: list[TaskContext] = []
        self._dataset: Optional[DataSet] = None
        self._file_properties: Optional[FileProperties] = None

        self._base_path = base_path.resolve()
        if not base_path.is_dir():
            raise FileNotFoundError(f"Path '{base_path}' doesn't exist or is not a directory!")
        self._input_path = base_path / "input"
        self._input_path.mkdir(exist_ok=True)
        self._output_path = base_path / "output"
        self._output_path.mkdir(exist_ok=True)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def logger(self) -> Logger:
        """Return logger for this context."""
        return self._logger

    @property
    def input_path(self) -> Path:
        """Return path for input files (common to all TaskContexts in hierarchy)."""
        return self._input_path

    @property
    def output_path(self) -> Path:
        """Return path for output files (common to all TaskContexts in hierarchy)."""
        return self._output_path

    def create_working_dir(self, name: str) -> Path:
        """Create a working directory specified by this TaskContext and the provided name."""
        working_path = self._base_path / "work" / "_".join(self._name_parts) / name
        working_path.mkdir(parents=True)
        return working_path

    @property
    def children(self) -> list["TaskContext"]:
        return self._children

    @property
    def name_parts(self) -> list[str]:
        return self._name_parts

    @property
    def qualified_path(self) -> PurePosixPath:
        return PurePosixPath("/".join(self._name_parts))

    def build_output_reference(self, final_part: str) -> PurePosixPath:
        parent_ref = "/".join(self._name_parts)
        ref = parent_ref + "/" + final_part if parent_ref else final_part
        ref = ref.replace(".", "_")
        return PurePosixPath(ref)

    def prepare_output_path(self, name: str):
        save_path = build_real_sub_path(self.output_path, sanitize_path(path=name))
        if save_path.exists():
            message = (
                f'The path "{save_path}" already exists, will overwrite! This is most likely an implementation error.'
            )
            warnings.warn(message, RuntimeWarning)
            self.logger.warning(message)
            save_path.unlink()
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path

    def relative_path(self, path: Path) -> Path:
        return path.resolve().relative_to(self._base_path)

    @property
    def dataset_name(self) -> str:
        if not self.name_parts:
            raise RuntimeError("Can not give dataset name for root TaskContext")
        return self.name_parts[-1]

    @property
    def dataset(self) -> Optional[DataSet]:
        return self._dataset

    @property
    def file_properties(self) -> Optional[FileProperties]:
        return self._file_properties

    def collect_datasets(self) -> Iterator[DataSet]:
        if own_dataset := self.dataset:
            yield own_dataset

        for child in self.children:
            yield from child.collect_datasets()

    async def exec[**P, R_DS: DataSet, *R_Ts](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R_DS] | Awaitable[tuple[R_DS, Unpack[R_Ts]]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Execute a subtask, store the dataset and swallow the results. Catch and log any errors."""

        try:
            await self.exec_with_result(dataset_name, task_fn, *args, **kwargs)
        except Exception as exception:
            self.logger.error("Error in sub context, continuing anyways...", exc_info=exception)
            warnings.warn(f"Error in sub context: {exception}")
            return

    async def exec_with_result[**P, R](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Execute a subtask, store the dataset and return the results."""

        # Short sleep allows the task to be cancelled
        await asyncio.sleep(0.001)

        # Replace dot with underscore. Keep slash as they are needed in archives.
        dataset_name = sanitize_path(dataset_name.replace(".", "_"))
        child_context = self._prepare_sub_context(dataset_name)

        try:
            result = await task_fn(child_context, *args, **kwargs)
        except Exception as exception:
            child_context.logger.error("Error", exc_info=exception)
            raise exception

        if is_dataset(result):
            child_context._put_dataset(cast(DataSet, result))
            return child_context, result
        elif isinstance(result, tuple) and is_dataset(result[0]):
            child_context._put_dataset(cast(DataSet, result[0]))
            return child_context, *result
        else:
            child_context.logger.error("Task function didn't return a dataset.")
            raise RuntimeError("Task function didn't return a dataset.")

    async def import_file(self, path: Path, dataset_name: Optional[str] = None) -> None:
        """Import file if supported. Store the dataset. Catch and log any errors."""
        if path.is_dir():
            async with asyncio.TaskGroup() as group:
                for sub_file in path.iterdir():
                    sub_dataset_name = dataset_name + "/" + sub_file.name if dataset_name else sub_file.name
                    group.create_task(self.import_file(sub_file, sub_dataset_name))
            return

        try:
            await self.import_file_with_result(path, dataset_name)
        except Exception as exception:
            self.logger.error("Error while importing file. continuing anyways...", exc_info=exception)
            warnings.warn(f"Error while importing file: {exception}")

    async def import_file_with_result(self, path: Path, dataset_name: Optional[str] = None) -> DataSet:
        """Import file if supported. Store and return the dataset."""
        if dataset_name is None:
            dataset_name = path.name

        if path.is_dir():
            raise RuntimeError("Can not run the import_file_with_results() on directories. Use import_file instead!")

        file_type = determine_file_type(path)
        file_properties = FileProperties(name=path.name, fileType=file_type, size=path.stat().st_size)
        importer = lookup_importer(file_type)
        if importer is None:
            message = lookup_unsupported_type_message(file_type)
            raise NotImplementedError(message)

        context, dataset = await self.exec_with_result(
            dataset_name,
            importer,
            path,
        )
        context._file_properties = file_properties
        return cast(DataSet, dataset)

    def _prepare_sub_context(self, dataset_name: str) -> "TaskContextImpl":
        child_name_parts = self.name_parts + [dataset_name]
        new_logger = self.logger.getChild(dataset_name)
        sub_context = TaskContextImpl(self._config, new_logger, self._base_path, child_name_parts)
        self.children.append(sub_context)
        return sub_context

    def _put_dataset(self, dataset: DataSet):
        if self._dataset:
            raise RuntimeError(
                "There is already a dataset in this context. You need to put exactly one dataset into this dataset context!"
            )
        self._dataset = dataset


@contextmanager
def create_temporary_task_context(config: Config, logger: Logger):
    with TemporaryDirectory() as work_dir:
        yield TaskContextImpl(config=config, logger=logger, base_path=Path(work_dir))
