import json
import logging
from argparse import ArgumentParser
from importlib.metadata import version as get_version
from logging import Logger, getLogger
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from uvicorn import run as server_run

from jobapi import get_job_api_router
from jobapi.config import AppConfig, get_app_config


def config_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")


def main():
    """We start and configure the server programmatically as an alternative to the FastAPI CLI."""

    config_logging()
    app_config = get_app_config()
    logger = getLogger(__name__)
    app = init_fastapi(app_config)

    logger.info("Starting server..")
    # To bind to all network interfaces add host="0.0.0.0".
    server_run(app, port=8000, host=app_config.host)
    logger.info("Shutting down.")


def init_fastapi(app_config: AppConfig):
    job_api_router = get_job_api_router(app_config)
    app = FastAPI(title="EDPS Rest API", version=get_version("edps"))
    _add_exception_handler(app, getLogger(__name__))
    app.include_router(job_api_router, prefix="/v1/dataspace")
    return app


def _add_exception_handler(app: FastAPI, logger: Logger):
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(f"{request.method} {request.url.path} -> HTTPException: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )


def cmdline_export_openapi_schema():
    def _get_args():
        parser = ArgumentParser()
        parser.add_argument(
            "-o", "--output", type=Path, required=True, help="PurePosixPath to output the OpenAPI schema to"
        )
        return parser.parse_args()

    args = _get_args()
    app_config = get_app_config()
    app = init_fastapi(app_config)
    schema = app.openapi()

    with open(args.output, "wt") as file:
        json.dump(schema, file, indent=4)


if __name__ == "__main__":
    main()
