from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.ocr_pipeline import OCRPipeline


@dataclass
class LayoutField:
    label: str
    text: str
    bbox: List[float]
    page_no: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LayoutModel:
    def __init__(self, checkpoint: Optional[str] = None) -> None:
        self.checkpoint = checkpoint
        self.ocr = OCRPipeline()

    def extract_fields(self, document_path: str) -> List[Dict[str, Any]]:
        pages = self.ocr.extract(document_path)
        fields: List[Dict[str, Any]] = []
        for page in pages:
            for region in page.get("bbox_list", []):
                label = self._label(region.get("text", ""))
                if label:
                    fields.append(
                        LayoutField(
                            label=label,
                            text=region.get("text", ""),
                            bbox=region.get("bbox", []),
                            page_no=page.get("page_no", 1),
                            confidence=float(region.get("conf", 0.5)),
                        ).to_dict()
                    )
        return fields

    def _label(self, text: str) -> Optional[str]:
        lower = text.lower().strip()
        if not lower:
            return None
        if re.search(r"\b(total|amount|grand total|net payable)\b", lower):
            return "total"
        if re.search(r"\b(invoice|bill|receipt)\b", lower):
            return "document_type"
        if re.search(r"\b(date|dated)\b|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", lower):
            return "date"
        if re.search(r"\b(gstin|pan|aadhaar|account)\b", lower):
            return "identifier"
        return None


def extract_layout(document_path: str, checkpoint: Optional[str] = None) -> List[Dict[str, Any]]:
    return LayoutModel(checkpoint=checkpoint).extract_fields(document_path)
