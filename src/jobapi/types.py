from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field

from edps.types import Config, UserProvidedEdpData


class JobData(BaseModel):
    """Data required for running a job."""

    user_provided_edp_data: UserProvidedEdpData = Field(
        description="EDP data that can not be computed automatically and must be supplied by the caller."
    )
    config: Config = Field(
        default_factory=lambda: Config(), description="Configuration of the analysis steps. These are optional."
    )


class JobState(str, Enum):
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    CANCELLED = "CANCELLED"
    DELETED = "DELETED"


END_STATES = set([JobState.FAILED, JobState.CANCELLED, JobState.COMPLETED, JobState.DELETED])


class JobView(BaseModel):
    job_id: UUID = Field(description="Job ID", frozen=True)
    state: JobState = Field(
        description="""Job state:\n
- WAITING_FOR_DATA: job is waiting for data upload\n
- QUEUED: job is queued for analyzation\n
- PROCESSING: job is processing data\n
- COMPLETED: job has completed successfully\n
- FAILED: job has failed\n
- CANCELLATION_REQUESTED: job cancellation requested by user\n
- CANCELLED: job has been canceled\n
- DELETED: job has been deleted
""",
        default=JobState.WAITING_FOR_DATA,
    )
    state_detail: str | None = Field(description="State details", default=None)
