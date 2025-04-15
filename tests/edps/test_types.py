from datetime import datetime, timezone

from extended_dataset_profile import AssetReference, DataSpace, ExtendedDatasetProfile, License, Publisher
from pydantic import BaseModel, HttpUrl

from edps.types import ComputedEdpData, UserProvidedEdpData


class Combined(ComputedEdpData, UserProvidedEdpData):
    pass


def test_user_provided_edp_data_subset_of_edp():
    _assert_subset_of(UserProvidedEdpData, ExtendedDatasetProfile)


def test_computed_edp_data_subset_of_edp():
    _assert_subset_of(UserProvidedEdpData, ExtendedDatasetProfile)


def test_user_provided_and_computed_combines_to_edp():
    _assert_subset_of(Combined, ExtendedDatasetProfile)
    _assert_subset_of(ExtendedDatasetProfile, Combined)


def test_recursively_escape_strings():
    user_provided_data = UserProvidedEdpData(
        assetRefs=[
            AssetReference(
                assetId="my-dataset-id",
                dataSpace=DataSpace(name="Hello <script>alert('XSS');</script>", url="https://beebucket.ai/en/"),
                assetUrl=HttpUrl("https://beebucket.ai/en/"),
                assetVersion="2.3.1",
                publisher=Publisher(name="beebucket"),
                publishDate=datetime(year=1995, month=10, day=10, hour=10, tzinfo=timezone.utc),
                license=License(url="https://opensource.org/license/mit"),
            )
        ],
        name="Hello <script>alert('XSS');</script>",
        dataCategory="TestDataCategory",
        description="Our very first test edp",
        tags=["test", "Hello <script>alert('XSS');</script>"],
        freely_available=True,
    )
    expected_escaped = "Hello &lt;script&gt;alert(&#x27;XSS&#x27;);&lt;/script&gt;"

    assert user_provided_data.name == expected_escaped
    assert user_provided_data.assetRefs is not None
    assert user_provided_data.assetRefs[0].dataSpace.name == expected_escaped
    assert user_provided_data.tags[1] == expected_escaped


def _assert_subset_of(potential_subset: type[BaseModel], superset: type[BaseModel]) -> None:
    assert len(potential_subset.model_fields) > 0, "Subset does not have any fields"
    for name, field_info in potential_subset.model_fields.items():
        superset_field = superset.model_fields.get(name)
        assert superset_field is not None, f"{name} not part of {superset}"
        assert field_info.annotation == superset_field.annotation, (
            f"Type of {name} not the same. Is {field_info.annotation}, should be {superset_field.annotation}"
        )
