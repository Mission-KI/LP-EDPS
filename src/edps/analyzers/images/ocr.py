import warnings
from typing import List

import easyocr
import numpy as np
from pandas import DataFrame
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OCRBox:
    x: float
    y_top: float
    y_bottom: float
    text: str
    confidence: float


class OCR:
    def __init__(self, languages: List[str]):
        """
        Initializes the OCR engine with the specified languages.

        :param languages: List of language codes (e.g. ['en']).
        """
        self._reader = easyocr.Reader(lang_list=languages, gpu=False, download_enabled=True, verbose=False)

    def read(self, img: np.ndarray) -> DataFrame:
        """
        Performs OCR on the given image and groups recognized text into lines.
        The results are returned as a pandas DataFrame where the index is the line number
        and the columns include 'detected text' and 'confidence'.

        :param img: The image to process.
        :return: A pandas DataFrame containing the OCR results.
        """
        with warnings.catch_warnings(record=False):
            # EasyOCR always tells pytorch to pin memory, even if there is no accelerator
            # device selected. If there is no accelerator device, this will lead to a warning
            # from pytorch.
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.",
            )
            raw_results = self._reader.readtext(img)
        if not raw_results:
            return DataFrame(columns=["detected text", "confidence"])

        ocr_boxes: List[OCRBox] = []
        for coords, text, confidence in raw_results:
            ocr_boxes.append(
                OCRBox(
                    x=coords[0][0],
                    y_top=coords[0][1],
                    y_bottom=coords[2][1],
                    text=text,
                    confidence=confidence,
                )
            )

        ocr_boxes.sort(key=lambda box: box.y_top)
        grouped_rows = _group_results_into_rows(ocr_boxes)
        return _convert_to_dataframe(grouped_rows)


def _group_results_into_rows(ocr_boxes: List[OCRBox]) -> List[List[OCRBox]]:
    if not ocr_boxes:
        return []

    rows: List[List[OCRBox]] = []
    current_row = [ocr_boxes[0]]
    current_row_top = ocr_boxes[0].y_top
    current_row_bottom = ocr_boxes[0].y_bottom

    for box in ocr_boxes[1:]:
        if current_row_top <= box.y_top <= current_row_bottom or current_row_top <= box.y_bottom <= current_row_bottom:
            current_row.append(box)
            current_row_top = min(current_row_top, box.y_top)
            current_row_bottom = max(current_row_bottom, box.y_bottom)
        else:
            current_row.sort(key=lambda b: b.x)
            rows.append(current_row)
            current_row = [box]
            current_row_top = box.y_top
            current_row_bottom = box.y_bottom

    if current_row:
        current_row.sort(key=lambda b: b.x)
        rows.append(current_row)

    return rows


def _convert_to_dataframe(rows: List[List[OCRBox]]) -> DataFrame:
    records = []
    for line_number, row in enumerate(rows, start=1):
        row_text = " ".join(item.text for item in row)
        avg_confidence = sum(item.confidence for item in row) / len(row)
        records.append({"line number": line_number, "detected text": row_text, "confidence": avg_confidence})

    df = DataFrame(records)
    df.set_index("line number", inplace=True)
    return df
