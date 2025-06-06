import shutil
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import UUID

from edps.taskcontext import REPORT_FILENAME
from edps.types import Config, UserProvidedEdpData
from jobapi.types import JobData, JobState, JobView


class Job(ABC):
    """Internal Job class. There are different implementations depending on persistence."""

    @property
    @abstractmethod
    def job_id(self) -> UUID: ...

    @property
    @abstractmethod
    def state(self) -> JobState: ...

    @property
    @abstractmethod
    def state_detail(self) -> Optional[str]: ...

    def set_state(self, state: JobState, detail: Optional[str] = None) -> None:
        if state in [JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED]:
            # Finalize job
            shutil.rmtree(self.input_data_dir, ignore_errors=True)
        self._set_state_impl(state, detail)

    @abstractmethod
    def _set_state_impl(self, state: JobState, detail: Optional[str]) -> None: ...

    @property
    @abstractmethod
    def asset_id(self) -> str: ...

    @property
    @abstractmethod
    def asset_version(self) -> Optional[str]: ...

    @property
    @abstractmethod
    def started(self) -> Optional[datetime]: ...

    @started.setter
    @abstractmethod
    def started(self, started: Optional[datetime]): ...

    @property
    @abstractmethod
    def finished(self) -> Optional[datetime]: ...

    @finished.setter
    @abstractmethod
    def finished(self, finished: Optional[datetime]): ...

    @property
    @abstractmethod
    def job_data(self) -> JobData: ...

    @property
    @abstractmethod
    def job_base_dir(self) -> Path: ...

    def to_job_view(self) -> JobView:
        return JobView(job_id=self.job_id, state=self.state, state_detail=self.state_detail)

    @property
    def input_data_dir(self) -> Path:
        return self.job_base_dir / "input"

    @property
    def zip_archive(self) -> Path:
        return self.job_base_dir / "result.zip"

    @property
    def report_file(self) -> Path:
        return self.job_base_dir / REPORT_FILENAME

    @property
    def log_file(self) -> Path:
        return self.job_base_dir / "job.log"

    @property
    def configuration(self) -> Config:
        return self.job_data.config

    @property
    def user_provided_edp_data(self) -> UserProvidedEdpData:
        return self.job_data.user_provided_edp_data


class JobSession(ABC):
    """Abstract JobSession for creating and finding jobs. It is instantiated by the JobRepository."""

    @abstractmethod
    async def create_job(self, job_id: UUID, job_data: JobData, job_base_dir: Path) -> Job:
        """Create a new job."""
        ...

    @abstractmethod
    async def get_job(self, job_id: UUID) -> Job:
        """Find a job by ID."""
        ...


class JobRepository(ABC):
    """Abstract JobRepository for creating JobSessions."""

    @abstractmethod
    @asynccontextmanager
    def new_session(self) -> AsyncIterator[JobSession]:
        """Start a new session in a context. When the context is closed, it gets automatically committed."""
        ...


class JobRepositoryFactory(ABC):
    """Abstract JobRepositoryFactory for creating JobRepositories."""

    @abstractmethod
    def create(self) -> JobRepository:
        """Create a new JobRepository."""
        ...
