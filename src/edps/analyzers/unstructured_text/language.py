import re
import warnings
from collections import Counter
from typing import Iterator, List, Optional, Set, Union, cast

import spacy
from extended_dataset_profile import WordFrequency
from lingua import Language, LanguageDetector, LanguageDetectorBuilder
from pandas import DataFrame, RangeIndex, Series
from spacy.tokens import Token
from spacy.tokens.doc import Doc

from edps.taskcontext import TaskContext

_TEXT_BODY_KEY = "text_body"
_DETECTED_LANGUAGE = "best-language"

_SANITIZER_REGEXS = {re.compile(r"\n"): " ", re.compile(r"(\s\s+)"): " "}


class MissingSpacyModelWarning(UserWarning):
    """
    Custom warning indicating that no spaCy model was found for the requested language.
    As a result, word cloud detection cannot be performed for that language.
    """


def _detect_language_confidences_per_sentence(detector: LanguageDetector, text: str) -> Series:
    results = detector.compute_language_confidence_values(text)
    confidence = Series({result.language: result.value for result in results}, name="confidence", dtype=float)
    return confidence


def _sanitize_sentence(text: str) -> str:
    for regex, replacement in _SANITIZER_REGEXS.items():
        text = regex.sub(replacement, text)
    return text


def _split_sentences(text: str) -> Iterator[str]:
    doc: Doc = spacy.blank("en")(text)
    sentencizer = spacy.pipeline.Sentencizer()
    doc = sentencizer(doc)
    for sentence in doc.sents:
        yield _sanitize_sentence(str(sentence))


def calculate_language_confidences(ctx: TaskContext, text: str) -> DataFrame:
    config = ctx.config.unstructured_text_config
    sentences = list(_split_sentences(text))
    if len(sentences) == 0:
        # Prevent accidentally filtering out all text.
        sentences = [_sanitize_sentence(text)]

    language_detector = LanguageDetectorBuilder.from_languages(*LANGUAGE_MODEL_MAP.keys()).build()
    columns: List[Union[Language, str]] = [
        result.language for result in language_detector.compute_language_confidence_values("")
    ]
    columns += [_TEXT_BODY_KEY]
    confidences = DataFrame(columns=columns, index=RangeIndex(0, 0))

    for index, sentence in enumerate(sentences):
        current_sentence_scores = _detect_language_confidences_per_sentence(language_detector, sentence)
        confidences.loc[index] = current_sentence_scores
        confidences[_TEXT_BODY_KEY] = confidences[_TEXT_BODY_KEY].astype(object)
        confidences.loc[index, _TEXT_BODY_KEY] = sentence

    detected_language = confidences.drop(columns=_TEXT_BODY_KEY).idxmax(axis=1)
    best_values = confidences.drop(columns=_TEXT_BODY_KEY).max(axis=1)
    detected_language[best_values < config.language_confidence_threshold] = None
    detected_language[confidences[_TEXT_BODY_KEY].str.len() < config.minimum_sentence_length] = None
    confidences[_DETECTED_LANGUAGE] = detected_language
    return confidences


def extract_languages(ctx: TaskContext, confidences: DataFrame) -> Set[str]:
    # Weight confidence by sentence length!
    languages: Set[Language] = set((language for language in confidences[_DETECTED_LANGUAGE] if language))
    ctx.logger.info("Detected Languages: %s", [language.name for language in languages])
    return {language.iso_code_639_3.name.lower() for language in languages}


def detect_word_cloud(ctx: TaskContext, confidences: DataFrame, top_n: int = 10) -> Iterator[WordFrequency]:
    aggregated_counts: Counter[str] = Counter()
    text_by_lang = confidences.groupby(_DETECTED_LANGUAGE)[_TEXT_BODY_KEY].apply(". ".join)

    for lang, text in text_by_lang.items():
        if lang is None:
            continue

        nlp = _load_spacy_model(ctx, cast(Language, lang))
        if nlp is None:
            continue

        doc = nlp(text)
        word_counts = Counter(
            _normalize_token(token) for token in doc if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 3
        )
        aggregated_counts.update(word_counts)

    for word, count in aggregated_counts.most_common(top_n):
        yield WordFrequency(word=word, count=count)


LANGUAGE_MODEL_MAP = {
    Language.GERMAN: "de_core_news_lg",
    Language.ENGLISH: "en_core_web_lg",
    Language.FRENCH: "fr_core_news_lg",
}


def _load_spacy_model(ctx: TaskContext, lang: Language) -> Optional[spacy.language.Language]:
    model_name = LANGUAGE_MODEL_MAP.get(lang)
    if not model_name:
        message = f"Can not detect word cloud for text. No spaCy model found for language '{lang}'."
        ctx.logger.warning(message)
        warnings.warn(message, MissingSpacyModelWarning)
        return None

    nlp = spacy.load(model_name)
    nlp.max_length = 5000000
    return nlp


def _normalize_token(token: Token, lemma_tags: Set[str] = {"NNS", "NNPS"}):
    """
    Normalize a spaCy token by converting plural forms to singular.

    Default tags:
    - "NNS": plural common noun (e.g., "cars")
    - "NNPS": plural proper noun (e.g., "Smiths")
    """
    return token.lemma_ if token.tag_ in lemma_tags else token.text
