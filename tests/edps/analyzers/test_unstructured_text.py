from extended_dataset_profile import StructuredDataSet, UnstructuredTextDataSet, WordFrequency
from pytest import mark, raises

from edps.analyzers.unstructured_text.chunk import Chunk
from edps.analyzers.unstructured_text.importer import unstructured_text_importer
from edps.analyzers.unstructured_text.language import (
    calculate_language_confidences,
    detect_word_cloud,
    extract_languages,
)
from edps.taskcontext import TaskContext
from tests.conftest import copy_asset_to_ctx_input_dir

_ENCODING = "utf-8"


def test_chunk_init():
    Chunk(0, 1)
    Chunk(0, 0)
    with raises(RuntimeError):
        Chunk(10, 1)


def test_chunk_overlap():
    base_chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    inside_chunk = Chunk(start_position_inclusive=9, end_position_exclusive=10)
    assert base_chunk.overlaps(inside_chunk)
    before_chunk = Chunk(start_position_inclusive=-10, end_position_exclusive=0)
    assert not base_chunk.overlaps(before_chunk)
    after_chunk = Chunk(start_position_inclusive=16, end_position_exclusive=17)
    assert not base_chunk.overlaps(after_chunk)


def test_position_setters():
    chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    with raises(RuntimeError):
        chunk.start_line_inclusive = 17
    with raises(RuntimeError):
        chunk.end_line_exclusive = -1
    chunk.start_line_inclusive = 1
    chunk.end_line_exclusive = 2


def test_chunk_subtract_non_overlapping():
    base_chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    non_contact_chunk = Chunk(start_position_inclusive=16, end_position_exclusive=17)
    result_chunks = list(base_chunk - non_contact_chunk)
    assert len(result_chunks) == 1
    assert result_chunks[0].start_line_inclusive == 0
    assert result_chunks[0].end_line_exclusive == 16
    # Check that base_chunk is unmodified.
    result_chunks[0].start_line_inclusive = 10
    assert base_chunk.start_line_inclusive == 0


def test_chunk_subtract_beginning():
    chunk = Chunk(start_position_inclusive=0, end_position_exclusive=16)
    subtract_chunk = Chunk(start_position_inclusive=0, end_position_exclusive=8)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 1
    assert result_chunks[0].start_line_inclusive == 8
    assert result_chunks[0].end_line_exclusive == 16


def test_chunk_subtract_end():
    chunk = Chunk(0, 16)
    subtract_chunk = Chunk(8, 16)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 1
    assert result_chunks[0].start_line_inclusive == 0
    assert result_chunks[0].end_line_exclusive == 8


def test_chunk_subtract_all():
    chunk = Chunk(0, 16)
    subtract_chunk = Chunk(0, 16)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 0


def test_chunk_subtract_middle():
    chunk = Chunk(0, 16)
    subtract_chunk = Chunk(4, 12)
    result_chunks = list(chunk - subtract_chunk)
    assert len(result_chunks) == 2
    assert result_chunks[0].start_line_inclusive == 0
    assert result_chunks[0].end_line_exclusive == 4
    assert result_chunks[1].start_line_inclusive == 12
    assert result_chunks[1].end_line_exclusive == 16


def test_deu_language_detection(ctx, path_language_deu_wiki_llm_txt):
    text = path_language_deu_wiki_llm_txt.read_text(encoding=_ENCODING)
    confidences = calculate_language_confidences(ctx, text)
    languages = extract_languages(ctx, confidences=confidences)
    assert languages == set(["deu", "eng"])


def test_deu_and_eng_language_detection(ctx, path_language_deu_eng_wiki_llm_txt):
    text = path_language_deu_eng_wiki_llm_txt.read_text(encoding=_ENCODING)
    confidences = calculate_language_confidences(ctx, text)
    languages = extract_languages(ctx, confidences=confidences)
    assert languages == set(["deu", "eng"])


def test_deu_word_cloud_detection(ctx, path_language_deu_wiki_llm_txt):
    confidences = calculate_language_confidences(ctx, path_language_deu_wiki_llm_txt.read_text(encoding=_ENCODING))
    word_cloud = list(detect_word_cloud(ctx, confidences=confidences))
    expected = [
        WordFrequency(word="Sprachmodelle", count=10),
        WordFrequency(word="Jahr", count=6),
        WordFrequency(word="Informationen", count=5),
        WordFrequency(word="Parameter", count=5),
        WordFrequency(word="Modalität", count=4),
        WordFrequency(word="ChatGPT", count=3),
        WordFrequency(word="Fähigkeiten", count=3),
        WordFrequency(word="LLMs", count=3),
        WordFrequency(word="OpenAI", count=3),
        WordFrequency(word="Sprachmodell", count=3),
    ]
    assert len(word_cloud) == 10
    for entry in expected:
        assert entry in word_cloud


def test_deu_and_eng_word_cloud_detection(ctx, path_language_deu_eng_wiki_llm_txt):
    text = path_language_deu_eng_wiki_llm_txt.read_text(encoding=_ENCODING)
    confidences = calculate_language_confidences(ctx, text)
    word_cloud = list(detect_word_cloud(ctx, confidences=confidences))
    expected = [
        WordFrequency(word="Sprachmodelle", count=10),
        WordFrequency(word="learning", count=7),
        WordFrequency(word="Jahr", count=6),
        WordFrequency(word="language", count=6),
        WordFrequency(word="LLMs", count=4),
        WordFrequency(word="Informationen", count=5),
        WordFrequency(word="Parameter", count=5),
        WordFrequency(word="network", count=5),
        WordFrequency(word="model", count=5),
        WordFrequency(word="Modalität", count=4),
    ]
    assert len(word_cloud) == 10
    for entry in expected:
        assert entry in word_cloud


@mark.asyncio
async def test_unstructured_text_only(ctx, path_unstructured_text_only_txt):
    path_in_ctx = copy_asset_to_ctx_input_dir(path_unstructured_text_only_txt, ctx)
    await ctx.exec("dataset", unstructured_text_importer, path_in_ctx)
    datasets = list(ctx.collect_datasets())
    assert len(datasets) == 1
    assert any(isinstance(dataset, UnstructuredTextDataSet) for dataset in datasets)


@mark.asyncio
async def test_unstructured_text_with_table(ctx: TaskContext, path_unstructured_text_with_table):
    path_in_ctx = copy_asset_to_ctx_input_dir(path_unstructured_text_with_table, ctx)
    await ctx.exec("dataset", unstructured_text_importer, path_in_ctx)
    datasets = list(ctx.collect_datasets())
    assert len(datasets) == 2
    assert any(isinstance(dataset, UnstructuredTextDataSet) for dataset in datasets)
    assert any(isinstance(dataset, StructuredDataSet) for dataset in datasets)
    structured_dataset = next(dataset for dataset in datasets if isinstance(dataset, StructuredDataSet))
    assert structured_dataset.columnCount == 3
    assert structured_dataset.rowCount == 2
    headers = [column.name for column in structured_dataset.all_columns]
    assert "id" in headers
    assert "name" in headers
    assert "width" in headers
    assert structured_dataset.all_columns
    assert structured_dataset.numericColumnCount == 2
    assert structured_dataset.stringColumnCount == 1
    assert structured_dataset.datetimeColumnCount == 0
    unstructured_dataset = next(dataset for dataset in datasets if isinstance(dataset, UnstructuredTextDataSet))
    assert len(unstructured_dataset.embeddedTables) == 1
    assert unstructured_dataset.wordCount == 20
    assert unstructured_dataset.lineCount == 2
