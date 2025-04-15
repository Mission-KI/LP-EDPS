import asyncio
from pathlib import Path
from pickle import load

from extended_dataset_profile import StructuredDataSet
from pandas import DataFrame

from edps.analyzers.structured import PandasAnalyzer
from edps.taskcontext import TaskContext


async def pickle_importer(ctx: TaskContext, path: Path) -> StructuredDataSet:
    ctx.logger.info("Reading object from %s", ctx.relative_path(path))
    with open(path, "rb") as opened_file:
        data_object = await asyncio.to_thread(load, opened_file)
    if not isinstance(data_object, DataFrame):
        raise NotImplementedError(f'Type "{type(data_object)}" not yet supported')
    ctx.logger.info("Finished reading object, object recognized as pandas data frame")
    return await PandasAnalyzer(data_object).analyze(ctx)
