import html
from datetime import datetime, timedelta
from typing import Any, List, Optional, Set, Union, get_args

import extended_dataset_profile
from extended_dataset_profile.version import Version
from pydantic import AnyUrl, BaseModel, Field, model_validator


def _get_edp_field(name: str) -> Any:
    """Function to avoid mypy from complaining about putting FieldInfo's into fields."""
    return extended_dataset_profile.ExtendedDatasetProfile.model_fields[name]


class UserProvidedEdpData(BaseModel):
    """
    All fields of the extended dataset profile which must be supplied by the caller.
    """

    name: str = _get_edp_field("name")
    assetRefs: List[extended_dataset_profile.AssetReference] = _get_edp_field("assetRefs")
    dataCategory: str | None = _get_edp_field("dataCategory")
    assetProcessingStatus: extended_dataset_profile.AssetProcessingStatus | None = _get_edp_field(
        "assetProcessingStatus"
    )
    description: str | None = _get_edp_field("description")
    tags: List[str] = _get_edp_field("tags")
    dataSubCategory: str | None = _get_edp_field("dataSubCategory")
    assetTypeInfo: str | None = _get_edp_field("assetTypeInfo")
    transferTypeFlag: extended_dataset_profile.AssetTransferType | None = _get_edp_field("transferTypeFlag")
    transferTypeFrequency: extended_dataset_profile.AssetUpdatePeriod | None = _get_edp_field("transferTypeFrequency")
    growthFlag: extended_dataset_profile.AssetGrowthRate | None = _get_edp_field("growthFlag")
    immutabilityFlag: extended_dataset_profile.AssetImmutability | None = _get_edp_field("immutabilityFlag")
    allowedForAiTraining: bool | None = _get_edp_field("allowedForAiTraining")
    nda: str | None = _get_edp_field("nda")
    dpa: str | None = _get_edp_field("dpa")
    dataLog: str | None = _get_edp_field("dataLog")
    freely_available: bool = _get_edp_field("freely_available")

    @model_validator(mode="before")
    def escape_all_string_fields(cls, data: Any) -> Any:
        return recursively_escape_strings(data)


class ComputedEdpData(BaseModel):
    """All fields of the extended dataset profile that get calculated by this service."""

    generatedBy: str = _get_edp_field("generatedBy")
    dataTypes: Set[extended_dataset_profile.DataSetType] = _get_edp_field("dataTypes")
    assetSha256Hash: str = _get_edp_field("assetSha256Hash")
    archiveDatasets: List[extended_dataset_profile.ArchiveDataSet] = _get_edp_field("archiveDatasets")
    structuredDatasets: List[extended_dataset_profile.StructuredDataSet] = _get_edp_field("structuredDatasets")
    semiStructuredDatasets: List[extended_dataset_profile.SemiStructuredDataSet] = _get_edp_field(
        "semiStructuredDatasets"
    )
    unstructuredTextDatasets: List[extended_dataset_profile.UnstructuredTextDataSet] = _get_edp_field(
        "unstructuredTextDatasets"
    )
    imageDatasets: List[extended_dataset_profile.ImageDataSet] = _get_edp_field("imageDatasets")
    schemaVersion: Version = _get_edp_field("schemaVersion")
    volume: int = _get_edp_field("volume")
    videoDatasets: List[extended_dataset_profile.VideoDataSet] = _get_edp_field("videoDatasets")
    audioDatasets: List[extended_dataset_profile.AudioDataSet] = _get_edp_field("audioDatasets")
    temporalCover: extended_dataset_profile.TemporalCover | None = _get_edp_field("temporalCover")
    periodicity: str | None = _get_edp_field("periodicity")
    documentDatasets: List[extended_dataset_profile.DocumentDataSet] = _get_edp_field("documentDatasets")
    datasetTree: List[extended_dataset_profile.DatasetTreeNode] = _get_edp_field("datasetTree")


class AugmentedColumn(BaseModel):
    """
    Stores information about a column having been modified before running the service.
    This information will be attached to each column on service execution.
    """

    name: str = Field(description="Name of the augmented column")
    datasetName: Optional[str] = Field(
        default=None,
        description="Name of the dataset this column was added to. If datasetName is None, EDPS will assume that the augmented column contained in all structured datasets.",
    )
    augmentation: extended_dataset_profile.Augmentation = Field(description="Augmentation information")


class DistributionConfig(BaseModel):
    """Configuration parameters specific to the distribution analysis."""

    minimum_number_numeric_values: int = Field(
        default=16, description="Minimum number of interpretable values to run numeric distribution analysis"
    )
    minimum_number_unique_string: int = Field(
        default=4, description="Minimum number of unique values to run string distribution analysis"
    )
    timeout: timedelta = Field(default=timedelta(minutes=1), description="Timeout to use for the distribution fitting.")
    max_samples: int = Field(
        default=int(1e6), description="Maximum number of values to use for determining the distribution of values."
    )


class SeasonalityConfig(BaseModel):
    """Configuration for the seasonality analysis step."""

    target_samples: int = Field(
        default=500, description="Number of samples to resample to before running seasonality analysis."
    )
    trend_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for evaluating a normalized trend as increasing or decreasing.",
    )


class StructuredConfig(BaseModel):
    """Configurations for the structured data analysis"""

    distribution: DistributionConfig = Field(
        default_factory=DistributionConfig,
        description="Configuration parameters specific to the distribution analysis.",
    )
    seasonality: SeasonalityConfig = Field(
        default_factory=SeasonalityConfig, description="Configurations specific to the seasonality analysis."
    )


class UnstructuredTextConfig(BaseModel):
    """
    Configuration for the unstructured text analysis.
    """

    minimum_sentence_length: int = Field(
        default=14,
        description="Minimum number of words in sentence for language detection. Shorter sentences will be skipped.",
    )
    language_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence to count a language as detected."
    )


class AudioConfig(BaseModel):
    """
    Configuration for the audio analysis.
    """

    lowest_frequency: int = Field(
        default=20, description="Lowest frequency which should be considered in the FFT analysis."
    )
    dynamic_range_min: float = Field(
        default=1e-1, description="Lower cutoff for dynamic range, default is -10 dB (1e-1)."
    )


class Config(BaseModel):
    """
    Extended dataset profile service configuration

    This configuration contains all customizable variables for the analysis of assets.
    All analyzer configurations are collected here.
    """

    augmentedColumns: List[AugmentedColumn] = Field(default_factory=list, description="List of augmented columns")
    structured_config: StructuredConfig = Field(
        default_factory=StructuredConfig, description="Configurations for the structured data analysis"
    )
    unstructured_text_config: UnstructuredTextConfig = Field(
        default_factory=UnstructuredTextConfig,
        description="Configuration for the unstructured text analysis.",
    )
    audio_config: AudioConfig = Field(default_factory=AudioConfig, description="Configuration for audio analysis.")


def recursively_escape_strings(data: Any) -> Any:
    if data is None or isinstance(data, (bool, datetime, AnyUrl)):
        return data
    elif isinstance(data, str):
        return html.escape(data)
    elif isinstance(data, list):
        return [recursively_escape_strings(item) for item in data]
    elif isinstance(data, dict):
        return {k: recursively_escape_strings(v) for k, v in data.items()}
    elif isinstance(data, BaseModel):
        as_dict = data.model_dump()
        escaped_dict = recursively_escape_strings(as_dict)
        return data.__class__(**escaped_dict)
    else:
        raise NotImplementedError(f"Type {type(data)} not supported")


DataSet = Union[
    extended_dataset_profile.ArchiveDataSet,
    extended_dataset_profile.StructuredDataSet,
    extended_dataset_profile.SemiStructuredDataSet,
    extended_dataset_profile.UnstructuredTextDataSet,
    extended_dataset_profile.ImageDataSet,
    extended_dataset_profile.VideoDataSet,
    extended_dataset_profile.AudioDataSet,
    extended_dataset_profile.DocumentDataSet,
]


def is_dataset(value: Any) -> bool:
    dataset_types = get_args(DataSet)
    return any(isinstance(value, dataset_type) for dataset_type in dataset_types)
