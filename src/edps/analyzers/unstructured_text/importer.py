from io import StringIO
from pathlib import Path

from clevercsv.encoding import get_encoding
from extended_dataset_profile import UnstructuredTextDataSet

from edps.analyzers.unstructured_text import Analyzer as UnstructuredTextAnalyzer
from edps.taskcontext import TaskContext


async def unstructured_text_importer(ctx: TaskContext, path: Path) -> UnstructuredTextDataSet:
    ctx.logger.info("Analyzing unstructured text '%s'", ctx.relative_path(path))
    encoding = get_encoding(path)
    if encoding is None:
        raise RuntimeError("Could not determine encoding of %s", ctx.relative_path(path))
    with open(path, "rt", encoding=encoding) as file_io:
        return await UnstructuredTextAnalyzer(ctx, file_io).analyze()


async def unstructured_text_importer_from_string(ctx: TaskContext, content: str) -> UnstructuredTextDataSet:
    ctx.logger.info("Analyzing unstructured text '%s'", ctx.qualified_path)
    with StringIO() as text_io:
        text_io.write(content)
        text_io.seek(0)
        return await UnstructuredTextAnalyzer(ctx, text_io).analyze()
