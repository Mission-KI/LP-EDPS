import hashlib
import warnings
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Optional, Tuple, cast
from warnings import warn

import easyocr
import static_ffmpeg
from extended_dataset_profile.models.v0.edp import (
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
from edps.file import calculate_size
from edps.filewriter import write_edp
from edps.importers import get_importable_types
from edps.report import PdfReportGenerator, ReportInput
from edps.taskcontext import TaskContext
from edps.types import AugmentedColumn, ComputedEdpData, DataSet, UserProvidedEdpData

REPORT_FILENAME = "report.pdf"


class UserInputError(RuntimeError):
    pass


class Service:
    def __init__(self):
        _logger = getLogger(__name__)
        _logger.info("Initializing")

        _logger.info("The following data types are supported: [%s]", get_importable_types())
        implemented_decompressions = [key for key, value in DECOMPRESSION_ALGORITHMS.items() if value is not None]
        _logger.info("The following compressions are supported: [%s]", ", ".join(implemented_decompressions))

    async def analyse_asset(self, ctx: TaskContext, user_data: UserProvidedEdpData) -> Path:
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
        computed_data = await self._compute_asset(ctx)
        edp = ExtendedDatasetProfile(**_as_dict(computed_data), **_as_dict(user_data))
        main_ref = user_data.assetRefs[0]
        json_name = main_ref.assetId + ("_" + main_ref.assetVersion if main_ref.assetVersion else "")
        json_path = await write_edp(ctx, PurePosixPath(json_name), edp)
        await self._generate_report(ctx, edp)
        return json_path

    async def _compute_asset(self, ctx: TaskContext) -> ComputedEdpData:
        input_files = [path for path in ctx.input_path.iterdir() if path.is_file()]
        number_input_files = len(input_files)
        if number_input_files != 1:
            raise UserInputError(f"Expected exactly one input file in input_path. Got {number_input_files}.")
        file = input_files[0]

        await ctx.import_file_with_result(file, file.name)
        computed_edp_data = self._create_computed_edp_data(ctx, file)
        if self._has_temporal_columns(computed_edp_data):
            computed_edp_data.temporalCover = self._get_overall_temporal_cover(computed_edp_data)
        computed_edp_data.periodicity = self._get_overall_temporal_consistency(computed_edp_data)
        return computed_edp_data

    def _create_computed_edp_data(self, ctx: TaskContext, path: Path) -> ComputedEdpData:
        generated_by = f"EDP Service @ {edps.__version__}"
        edp = ComputedEdpData(
            volume=calculate_size(path), generatedBy=generated_by, assetSha256Hash=compute_sha256(path)
        )
        augmenter = _Augmenter(ctx, ctx.config.augmentedColumns)
        self._insert_datasets_into_edp(ctx, edp, augmenter, None)
        augmenter.warn_unapplied_augmentations()
        return edp

    def _insert_datasets_into_edp(
        self,
        ctx: TaskContext,
        edp: ComputedEdpData,
        augmenter: "_Augmenter",
        parent_node_reference: Optional[JsonReference],
    ):
        childrens_parent_node_reference: Optional[JsonReference]
        if ctx.dataset:
            augmenter.apply(ctx.dataset, ctx.dataset_name)
            list_name, index = self._insert_dataset_into_edp(edp, ctx.dataset)
            reference = _make_json_reference(f"#/{list_name}/{index}")
            edp.datasetTree.append(
                DatasetTreeNode(
                    dataset=reference,
                    parent=parent_node_reference,
                    name=ctx.dataset_name,
                    fileProperties=ctx.file_properties,
                )
            )
            childrens_parent_node_reference = _make_json_reference(f"#/datasetTree/{len(edp.datasetTree) - 1}")
        else:
            childrens_parent_node_reference = parent_node_reference

        for child in ctx.children:
            self._insert_datasets_into_edp(child, edp, augmenter, childrens_parent_node_reference)

    def _insert_dataset_into_edp(self, edp: ComputedEdpData, dataset: DataSet) -> Tuple[str, int]:
        if isinstance(dataset, ArchiveDataSet):
            edp.archiveDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.archive)
            return "archiveDatasets", (len(edp.archiveDatasets) - 1)
        elif isinstance(dataset, StructuredDataSet):
            edp.structuredDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.structured)
            return "structuredDatasets", (len(edp.structuredDatasets) - 1)
        elif isinstance(dataset, SemiStructuredDataSet):
            edp.semiStructuredDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.semiStructured)
            return "semiStructuredDatasets", (len(edp.semiStructuredDatasets) - 1)
        elif isinstance(dataset, UnstructuredTextDataSet):
            edp.unstructuredTextDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.unstructuredText)
            return "unstructuredTextDatasets", (len(edp.unstructuredTextDatasets) - 1)
        elif isinstance(dataset, ImageDataSet):
            edp.imageDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.image)
            return "imageDatasets", (len(edp.imageDatasets) - 1)
        elif isinstance(dataset, VideoDataSet):
            edp.videoDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.video)
            return "videoDatasets", (len(edp.videoDatasets) - 1)
        elif isinstance(dataset, AudioDataSet):
            edp.audioDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.audio)
            return "audioDatasets", (len(edp.audioDatasets) - 1)
        elif isinstance(dataset, DocumentDataSet):
            edp.documentDatasets.append(dataset)
            edp.dataTypes.add(DataSetType.documents)
            return "documentDatasets", (len(edp.documentDatasets) - 1)

        raise NotImplementedError(f'Did not expect dataset type "{type(dataset)}"')

    def _has_temporal_columns(self, edp: ComputedEdpData) -> bool:
        for structured in edp.structuredDatasets:
            if len(structured.datetimeColumns) > 0:
                return True

        return False

    def _get_overall_temporal_cover(self, edp: ComputedEdpData) -> TemporalCover:
        earliest = min(
            column.temporalCover.earliest
            for structured in edp.structuredDatasets
            for column in structured.datetimeColumns
        )
        latest = max(
            column.temporalCover.latest
            for structured in edp.structuredDatasets
            for column in structured.datetimeColumns
        )
        return TemporalCover(earliest=earliest, latest=latest)

    def _get_overall_temporal_consistency(self, edp: ComputedEdpData) -> Optional[str]:
        all_temporal_consistencies = list(self._iterate_all_temporal_consistencies(edp))
        if len(all_temporal_consistencies) == 0:
            return None
        sum_temporal_consistencies = sum(all_temporal_consistencies[1:], all_temporal_consistencies[0])
        return cast(
            Optional[str],
            determine_periodicity(
                sum_temporal_consistencies["numberOfGaps"], sum_temporal_consistencies["differentAbundancies"]
            ),
        )

    def _iterate_all_temporal_consistencies(self, edp: ComputedEdpData) -> Iterator[DataFrame]:
        for dataset in edp.structuredDatasets:
            for row in dataset.datetimeColumns:
                dataframe = DataFrame(
                    index=[item.timeScale for item in row.temporalConsistencies],
                )
                dataframe["differentAbundancies"] = [item.differentAbundancies for item in row.temporalConsistencies]
                dataframe["numberOfGaps"] = [item.numberOfGaps for item in row.temporalConsistencies]
                yield dataframe

    async def _generate_report(self, ctx: TaskContext, edp: ExtendedDatasetProfile):
        try:
            input = ReportInput(edp=edp)
            with get_report_path(ctx).open("wb") as file_io:
                await PdfReportGenerator().generate(ctx, input, ctx.output_path, file_io)
        except Exception as exception:
            ctx.logger.warning("Error generating the report, continuing anyways..", exc_info=exception)
            warnings.warn(f"Error generating the report. Error: {exception}")


def _as_dict(model: BaseModel):
    field_keys = model.model_fields.keys()
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


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_report_path(ctx: TaskContext) -> Path:
    return ctx.output_path / REPORT_FILENAME


def download_artifacts():
    """
    Downloads all artifacts needed for the service execution.
    """
    static_ffmpeg.add_paths(weak=True)
    easyocr.Reader(["en", "de"], gpu=False, download_enabled=True)
