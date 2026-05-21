from __future__ import annotations

from pathlib import Path
from typing import Any, List

from src.ocr_pipeline import OCRPipeline, extract_document

_default_pipeline: OCRPipeline | None = None


def get_pipeline() -> OCRPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = OCRPipeline(prefer_paddle=False)
    return _default_pipeline


def preprocess_image(img):
    return get_pipeline().preprocess_image(img)


def text_in_pdf(filepath: str) -> bool:
    return get_pipeline()._pdf_has_text(Path(filepath))


def load_digital_pdf(filepath: str) -> list:
    return [page.to_dict() for page in get_pipeline()._load_digital_pdf(Path(filepath))]


def load_scanned_pdf(filepath: str) -> list:
    return [page.to_dict() for page in get_pipeline()._load_scanned_pdf(Path(filepath))]


def load_image(filepath: str) -> list:
    return [get_pipeline()._load_image(Path(filepath)).to_dict()]


def load_document(filepath: str, **kwargs: Any) -> List[dict]:
    if kwargs:
        return extract_document(filepath, **kwargs)
    return get_pipeline().extract(filepath)
