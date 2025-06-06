import asyncio
import hashlib
import shutil
from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path
from typing import IO, Dict, Iterator, List, Optional, Tuple, Type, cast
from warnings import warn

import easyocr
import static_ffmpeg
from extended_dataset_profile import (
    CURRENT_VERSION,
    ArchiveDataSet,
    AudioDataSet,
    DatasetTreeNode,
    DataSetType,
    DocumentDataSet,
    ExtendedDatasetProfile,
    ImageDataSet,
    JsonReference,
    SemiStructuredDataSet,
    StructuredDataSet,
    TemporalCover,
    UnstructuredTextDataSet,
    VideoDataSet,
)
from pandas import DataFrame
from pydantic import BaseModel

import edps
from edps.analyzers.structured import determine_periodicity
from edps.compression import DECOMPRESSION_ALGORITHMS
from edps.compression.zip import ZipAlgorithm
from edps.file import calculate_size
from edps.importers import get_importable_types
from edps.report import PdfReportGenerator, ReportInput
from edps.taskcontext import TaskContext
from edps.taskcontextimpl import create_temporary_task_context
from edps.types import AugmentedColumn, ComputedEdpData, Config, DataSet, UserProvidedEdpData

TEXT_ENCODING = "utf-8"


class UserInputError(RuntimeError):
    pass


def dump_service_info():
    logger = getLogger(__name__)
    logger.info("The following data types are supported: [%s]", get_importable_types())
    implemented_decompressions = [key for key, value in DECOMPRESSION_ALGORITHMS.items() if value is not None]
    logger.info("The following compressions are supported: [%s]", ", ".join(implemented_decompressions))


class AnalyzeResult:
    def __init__(self, task_context: TaskContext, edp: ExtendedDatasetProfile):
        self._task_context = task_context
        self.edp = edp

    @property
    def output_path(self) -> Path:
        return self._task_context.output_path

    async def write_edp_to_output(self) -> Path:
        main_ref = self.edp.assetRefs[0]
        json_name = main_ref.assetId + ("_" + main_ref.assetVersion if main_ref.assetVersion else "") + ".json"
        save_path = self._task_context.prepare_output_path(json_name)
        relative_save_path = Path(save_path.relative_to(self._task_context.output_path))
        with open(save_path, "wt", encoding=TEXT_ENCODING) as io_wrapper:
            json: str = self.edp.model_dump_json(by_alias=True)
            await asyncio.to_thread(io_wrapper.write, json)
        self._task_context.logger.debug('Generated EDP file "%s"', relative_save_path)
        return relative_save_path

    async def write_pdf_report_to_output(self) -> Path:
        input = ReportInput(edp=self.edp)
        with self._task_context.report_file_path.open("wb") as file_io:
            await PdfReportGenerator().generate(self._task_context, input, file_io)
        return self._task_context.report_file_path

    async def compress_output_to(self, zip_output: Path | IO[bytes]) -> None:
        await ZipAlgorithm().compress(self._task_context.output_path, zip_output)


async def analyse_asset_to_zip(
    input_file: Path,
    zip_output: Path | IO[bytes],
    user_data: UserProvidedEdpData,
    config: Optional[Config] = None,
    logger: Optional[Logger] = None,
) -> ExtendedDatasetProfile:
    """
    Let the service analyse the assets in ctx.input_path.
    The result (EDP JSON, plots and report) is written to ctx.output_path.

    Parameters
    ----------
    ctx : TaskContext
        Gives access to the appropriate logger and output_context and allows executing sub-tasks.
    config_data : Config
        The meta and config information about the asset supplied by the data space. These can not get calculated and must be supplied
        by the user.

    Returns
    -------
    Path
        File path to the generated EDP JSON (relative to ctx.output_path).
    """
    if config is None:
        config = Config()

    if logger is None:
        logger = getLogger("edps")

    with create_temporary_task_context(config=config, logger=logger) as task_context:
        result = await analyze_asset(input_file=input_file, task_context=task_context, user_data=user_data)
        await result.write_edp_to_output()
        await result.write_pdf_report_to_output()
        await result.compress_output_to(zip_output)
        return result.edp


async def analyze_asset(input_file: Path, task_context: TaskContext, user_data: UserProvidedEdpData) -> AnalyzeResult:
    shutil.copy(input_file, task_context.input_path)
    computed_data = await _compute_asset(task_context)
    edp = ExtendedDatasetProfile(**_as_dict(computed_data), **_as_dict(user_data))
    return AnalyzeResult(task_context=task_context, edp=edp)


async def _compute_asset(ctx: TaskContext) -> ComputedEdpData:
    input_files = [path for path in ctx.input_path.iterdir() if path.is_file()]
    number_input_files = len(input_files)
    if number_input_files != 1:
        raise UserInputError(f"Expected exactly one input file in input_path. Got {number_input_files}.")
    file = input_files[0]

    await ctx.import_file_with_result(file, file.name)
    computed_edp_data = _create_computed_edp_data(ctx, file)
    if _has_temporal_columns(computed_edp_data):
        computed_edp_data.temporalCover = _get_overall_temporal_cover(computed_edp_data)
    computed_edp_data.periodicity = _get_overall_temporal_consistency(computed_edp_data)
    return computed_edp_data


def _create_computed_edp_data(ctx: TaskContext, path: Path) -> ComputedEdpData:
    generated_by = f"EDP Service @ {edps.__version__}"
    edp = ComputedEdpData(
        schemaVersion=CURRENT_VERSION,
        volume=calculate_size(path),
        generatedBy=generated_by,
        assetSha256Hash=compute_sha256(path),
    )
    augmenter = _Augmenter(ctx, ctx.config.augmentedColumns)
    _insert_datasets_into_edp(ctx, edp, augmenter, None)
    augmenter.warn_unapplied_augmentations()
    return edp


def _insert_datasets_into_edp(
    ctx: TaskContext,
    edp: ComputedEdpData,
    augmenter: "_Augmenter",
    parent_node_reference: Optional[JsonReference],
):
    childrens_parent_node_reference: Optional[JsonReference]
    if ctx.dataset:
        augmenter.apply(ctx.dataset, ctx.dataset_name)
        list_name, dataset_type = _DATASET_TYPE_TABLE[type(ctx.dataset)]
        edp.dataTypes.add(dataset_type)
        dataset_list: List = getattr(edp, list_name)
        index = len(dataset_list)
        dataset_list.append(ctx.dataset)
        reference = _make_json_reference(f"#/{list_name}/{index}")
        edp.datasetTree.append(
            DatasetTreeNode(
                dataset=reference,
                datasetType=dataset_type,
                parent=parent_node_reference,
                name=ctx.dataset_name,
                fileProperties=ctx.file_properties,
            )
        )
        childrens_parent_node_reference = _make_json_reference(f"#/datasetTree/{len(edp.datasetTree) - 1}")
    else:
        childrens_parent_node_reference = parent_node_reference

    for child in ctx.children:
        _insert_datasets_into_edp(child, edp, augmenter, childrens_parent_node_reference)


def _has_temporal_columns(edp: ComputedEdpData) -> bool:
    for structured in edp.structuredDatasets:
        if len(structured.datetimeColumns) > 0:
            return True

    return False


def _get_overall_temporal_cover(edp: ComputedEdpData) -> TemporalCover:
    earliest = min(
        column.temporalCover.earliest for structured in edp.structuredDatasets for column in structured.datetimeColumns
    )
    latest = max(
        column.temporalCover.latest for structured in edp.structuredDatasets for column in structured.datetimeColumns
    )
    return TemporalCover(earliest=earliest, latest=latest)


def _get_overall_temporal_consistency(edp: ComputedEdpData) -> Optional[str]:
    all_temporal_consistencies = list(_iterate_all_temporal_consistencies(edp))
    if len(all_temporal_consistencies) == 0:
        return None
    sum_temporal_consistencies = sum(all_temporal_consistencies[1:], all_temporal_consistencies[0])
    return cast(
        Optional[str],
        determine_periodicity(
            sum_temporal_consistencies["numberOfGaps"], sum_temporal_consistencies["differentAbundancies"]
        ),
    )


def _iterate_all_temporal_consistencies(edp: ComputedEdpData) -> Iterator[DataFrame]:
    for dataset in edp.structuredDatasets:
        for row in dataset.datetimeColumns:
            if len(row.temporalConsistencies) == 0:
                continue
            dataframe = DataFrame(
                index=[item.timeScale for item in row.temporalConsistencies],
            )
            dataframe["differentAbundancies"] = [item.differentAbundancies for item in row.temporalConsistencies]
            dataframe["numberOfGaps"] = [item.numberOfGaps for item in row.temporalConsistencies]
            yield dataframe


def _as_dict(model: BaseModel):
    field_keys = type(model).model_fields.keys()
    return {key: model.__dict__[key] for key in field_keys}


@dataclass
class _AugmentationStep:
    augmented_column: AugmentedColumn
    applied: bool = False


class _Augmenter:
    def __init__(self, ctx: TaskContext, augmentations: List[AugmentedColumn]):
        self._ctx = ctx
        self._augmentations = [_AugmentationStep(augmented_column=augmentation) for augmentation in augmentations]

    def apply(self, dataset: DataSet, dataset_name: str):
        if not isinstance(dataset, StructuredDataSet):
            return

        for step in self._augmentations:
            step.applied |= self._apply_single_augmentation(dataset, step.augmented_column, dataset_name)

    def _apply_single_augmentation(
        self, dataset: StructuredDataSet, augmented_column: AugmentedColumn, dataset_name: str
    ) -> bool:
        if augmented_column.datasetName is not None and augmented_column.datasetName != dataset_name:
            return False

        try:
            dataset.get_columns_dict()[augmented_column.name].augmentation = augmented_column.augmentation
            return True
        except KeyError:
            if augmented_column.datasetName is not None:
                message = (
                    f'Augmented column "{augmented_column.name}" is not known in file "{augmented_column.datasetName}'
                )
                self._ctx.logger.warning(message)
                warn(message)
                return True
            return False

    def warn_unapplied_augmentations(self) -> None:
        non_applied_augmentations = [step.augmented_column for step in self._augmentations if not step.applied]
        for augmented_column in non_applied_augmentations:
            if augmented_column.datasetName is None:
                message = f'No column "{augmented_column.name}" found in any dataset!'
                warn(message)
                self._ctx.logger.warning(message)
            else:
                message = f'"{augmented_column}" is not a known structured dataset!"'
                warn(message)
                self._ctx.logger.warning(message)


def _make_json_reference(text: str) -> JsonReference:
    return JsonReference(reference=text)


_DATASET_TYPE_TABLE: Dict[Type[DataSet], Tuple[str, DataSetType]] = {
    ArchiveDataSet: ("archiveDatasets", DataSetType.archive),
    StructuredDataSet: ("structuredDatasets", DataSetType.structured),
    SemiStructuredDataSet: ("semiStructuredDatasets", DataSetType.semiStructured),
    UnstructuredTextDataSet: ("unstructuredTextDatasets", DataSetType.unstructuredText),
    ImageDataSet: ("imageDatasets", DataSetType.image),
    VideoDataSet: ("videoDatasets", DataSetType.video),
    AudioDataSet: ("audioDatasets", DataSetType.audio),
    DocumentDataSet: ("documentDatasets", DataSetType.documents),
}


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_artifacts():
    """
    Downloads all artifacts needed for the service execution.
    Therefore the service will not have to do any downloads after this to
    function. This is especially useful when running in isolated environments.

    This is optional for most installations. The required artifacts will
    be lazy loaded if this function was not called.
    """
    static_ffmpeg.add_paths(weak=True)
    easyocr.Reader(["en", "de"], gpu=False, download_enabled=True, verbose=False)
