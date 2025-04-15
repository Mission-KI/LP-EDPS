import json
from pathlib import Path

from clevercsv.encoding import get_encoding
from extended_dataset_profile import SemiStructuredDataSet

from edps.analyzers.semi_structured import JsonAnalyzer
from edps.taskcontext import TaskContext


async def json_importer(ctx: TaskContext, path: Path) -> SemiStructuredDataSet:
    ctx.logger.info("Importing '%s' as JSON", ctx.relative_path(path))

    encoding = get_encoding(path)
    if encoding is None:
        raise RuntimeError("Could not determine encoding of %s", ctx.relative_path(path))

    with open(path, "rt", encoding=encoding) as opened_file:
        json_data = json.load(opened_file)

    if not isinstance(json_data, (list, dict)):
        raise ValueError("Expected JSON root to be a list or dict, got " + type(json_data).__name__)

    return await JsonAnalyzer(json_data).analyze(ctx)
