from pathlib import Path

from extended_dataset_profile import ArchiveDataSet

from edps.compression import DECOMPRESSION_ALGORITHMS
from edps.file import calculate_size, determine_file_type
from edps.taskcontext import TaskContext


async def archive_importer(ctx: TaskContext, path: Path) -> ArchiveDataSet:
    file_type = determine_file_type(path)
    decompression = DECOMPRESSION_ALGORITHMS.get(file_type)
    if decompression is None:
        raise NotImplementedError(f'Archive type "{file_type}" not supported.')

    extraction_path = ctx.create_working_dir(path.name)
    await decompression.extract(path, extraction_path)
    archive_dataset = ArchiveDataSet(
        algorithm=file_type,
        extractedSize=calculate_size(extraction_path),
    )

    await ctx.import_file(extraction_path, dataset_name=None)
    return archive_dataset
