from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Concatenate, Iterator, Optional, Tuple, Unpack, overload

from extended_dataset_profile import FileProperties

from edps.types import Config, DataSet


class TaskContext(ABC):
    """Interface. A context provides a logger and supports executing sub-tasks."""

    @property
    @abstractmethod
    def config(self) -> Config:
        """Return the configuration of this service run."""

    @property
    @abstractmethod
    def logger(self) -> Logger:
        """Return logger for this context."""

    @property
    @abstractmethod
    def input_path(self) -> Path:
        """Return path for input files (common to all TaskContexts in hierarchy)."""

    @property
    @abstractmethod
    def output_path(self) -> Path:
        """Return path for output files (common to all TaskContexts in hierarchy)."""

    @abstractmethod
    def create_working_dir(self, name: str) -> Path:
        """Create a working directory specified by this TaskContext and the provided name.
        This allows multiple working directories per TaskContext.
        Trying to create the same working directory twice (same TaskContext and name) will raise an error."""

    @property
    @abstractmethod
    def children(self) -> list["TaskContext"]:
        """Return all child TaskContexts."""

    @property
    @abstractmethod
    def name_parts(self) -> list[str]:
        """Return all name parts coming from the TaskContext hierarchy."""

    @property
    @abstractmethod
    def qualified_path(self) -> PurePosixPath:
        """Return a qualified path containing of all the name parts."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the last name part identifying the dataset."""

    @property
    @abstractmethod
    def dataset(self) -> Optional[DataSet]:
        """Return the DataSet attached to the TaskContext."""

    @property
    @abstractmethod
    def file_properties(self) -> Optional[FileProperties]:
        """Return the file properties, if this instance had import file executed."""

    @abstractmethod
    def build_output_reference(self, final_part: str) -> PurePosixPath:
        """Build a new output reference that can be used for a file name in the output path consisting of all the name parts."""

    @abstractmethod
    def relative_path(self, path: Path) -> Path:
        """Convert the input path to a path relative to the TaskContext base path."""

    @abstractmethod
    def collect_datasets(self) -> Iterator[DataSet]:
        """Return the datasets of this TaskContext and all its children using depth-first traversal."""

    @abstractmethod
    async def exec[**P, R_DS: DataSet, *R_Ts](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R_DS] | Awaitable[tuple[R_DS, Unpack[R_Ts]]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Execute a subtask swallowing the results of the task function. The returned dataset is stored in the TaskContext.
        Contrary to exec_with_result() any occuring errors are automatically caught and logged."""

    @overload
    async def exec_with_result[**P, R_DS: DataSet, *R_Ts](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[Tuple[R_DS, Unpack[R_Ts]]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Tuple["TaskContext", R_DS, Unpack[R_Ts]]:
        """[Overload] Execute a subtask returning a tuple of DataSet and other information. The returned dataset is also stored in the TaskContext."""

    @overload
    async def exec_with_result[**P, R_DS: DataSet](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R_DS]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Tuple["TaskContext", R_DS]:
        """[Overload] Execute a subtask returning just a DataSet. The returned dataset is also stored in the TaskContext."""

    @abstractmethod
    async def exec_with_result[**P, R](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Any:
        """Execute a subtask returning the results of the task function (which must at least contain a DataSet).
        This creates a sub-context. The returned dataset is also stored in the TaskContext.
        Contrary to method exec() any errors occuring while trying to execute the subtask must be handled by the caller."""

    @abstractmethod
    async def import_file(self, path: Path, dataset_name: Optional[str] = None) -> None:
        """Import and analyze the file if it's a supported type. The dataset is stored in the TaskContext.
        Contrary to method import_file_with_result() any occuring errors are automatically caught and logged."""

    @abstractmethod
    async def import_file_with_result(self, path: Path, dataset_name: Optional[str] = None) -> DataSet:
        """Import and analyze the file if it's a supported type. The dataset is stored in the TaskContext and additionally returned.
        Contrary to method import_file() any occuring errors must be handled by the caller."""
