import asyncio
from typing import List
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient, Response

from jobapi.__main__ import init_fastapi
from jobapi.config import AppConfig
from jobapi.types import JobData, JobState, JobView


@pytest.fixture
def app(path_work):
    return init_fastapi(AppConfig(working_dir=path_work))


@pytest.fixture
async def test_client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.mark.slow
async def test_api(test_client: AsyncClient, user_provided_data, path_data_test_csv):
    job_data = JobData(user_provided_edp_data=user_provided_data)
    response = await test_client.post("/v1/dataspace/analysisjob", content=job_data.model_dump_json(by_alias=True))
    assert response.status_code == 200, response.text
    job = JobView.model_validate(response.json())
    assert job.state == JobState.WAITING_FOR_DATA

    asset_path = path_data_test_csv
    files = {"upload_file": (asset_path.name, asset_path.read_bytes())}
    response = await test_client.post(f"/v1/dataspace/analysisjob/{job.job_id}/data", files=files)
    job = extract_job_view(response)
    assert job.state in [JobState.QUEUED, JobState.PROCESSING]

    # Wait until the processing is completed
    job = await _wait_until_state_is_not(test_client, job, [JobState.QUEUED, JobState.PROCESSING])
    assert job.state == JobState.COMPLETED

    # Check for valid result
    response = await test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/result")
    assert response.status_code == 200, response.text
    # Check for report
    response = await test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/report")
    assert response.status_code == 200, response.text
    # Check for log
    response = await test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/log")
    assert response.status_code == 200, response.text


@pytest.mark.slow
async def test_api_client_error(test_client: AsyncClient):
    random_uuid = uuid4()
    response = await test_client.get(f"/v1/dataspace/analysisjob/{random_uuid}/result")
    assert response.status_code == 400
    assert response.json()["detail"] is not None


@pytest.mark.slow
async def test_api_cancel_waiting_for_data(test_client: AsyncClient, user_provided_data, path_data_test_csv):
    job_data = JobData(user_provided_edp_data=user_provided_data)
    response = await test_client.post("/v1/dataspace/analysisjob", content=job_data.model_dump_json(by_alias=True))
    assert response.status_code == 200, response.text
    job = JobView.model_validate(response.json())
    assert job.state == JobState.WAITING_FOR_DATA

    # Try canceling before the job has been queued
    response = await test_client.post(f"/v1/dataspace/analysisjob/{job.job_id}/cancel")
    assert response.status_code == 204, response.text


@pytest.mark.slow
async def test_api_cancel_running(test_client: AsyncClient, user_provided_data, path_data_test_csv):
    job_data = JobData(user_provided_edp_data=user_provided_data)
    response = await test_client.post("/v1/dataspace/analysisjob", content=job_data.model_dump_json(by_alias=True))
    assert response.status_code == 200, response.text
    job = JobView.model_validate(response.json())
    assert job.state == JobState.WAITING_FOR_DATA

    asset_path = path_data_test_csv
    files = {"upload_file": (asset_path.name, asset_path.read_bytes())}

    await test_client.post(f"/v1/dataspace/analysisjob/{job.job_id}/data", files=files)
    await asyncio.sleep(1)

    response = await test_client.post(f"/v1/dataspace/analysisjob/{job.job_id}/cancel")
    assert response.status_code == 204, response.text
    job = await _wait_until_state_is_not(test_client, job, [JobState.CANCELLATION_REQUESTED])

    # Check for status
    response = await test_client.get(f"/v1/dataspace/analysisjob/{job.job_id}/status")
    job = extract_job_view(response)
    assert job.state == JobState.CANCELLED


async def _wait_until_state_is_not(client: AsyncClient, job: JobView, blocking_states: List[JobState]):
    response = await client.get(f"/v1/dataspace/analysisjob/{job.job_id}/status")
    job = extract_job_view(response)

    while job.state in blocking_states:
        response = await client.get(f"/v1/dataspace/analysisjob/{job.job_id}/status")
        job = extract_job_view(response)
        await asyncio.sleep(1.0)

    return job


def extract_job_view(response: Response) -> JobView:
    assert response.status_code == 200, response.text
    return JobView.model_validate(response.json())
