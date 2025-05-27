import asyncio
import logging
from logging import getLogger

from pontusx.args import get_args
from pontusx.service import run_service


def config_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")


async def amain():
    config_logging()
    logger = getLogger("edps.pontusx")
    args = get_args()
    logger.debug("Pontus-X CLI got these arguments: %s", args)
    logger.info("Processing asset with DID='%s'", args.did)
    await run_service(logger, args)
    logger.info("Done")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
