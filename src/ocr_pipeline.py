from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Sequence

import cv2
import fitz
import numpy as np
import pdfplumber

from src.invoice_extractor import extract_invoice_fields

try:
    import pytesseract
except Exception:  # pragma: no cover - optional runtime dependency
    pytesseract = None
else:
    tesseract_cmd = which("tesseract") or "/opt/homebrew/bin/tesseract"
    if Path(tesseract_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

PaddleOCR = None


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}


@dataclass
class OCRRegion:
    text: str
    bbox: List[float]
    conf: float = 1.0
    engine: str = "unknown"
    label: Optional[str] = None


@dataclass
class OCRPage:
    text: str
    page_no: int
    bbox_list: List[Dict[str, Any]]
    source_file: str
    width: Optional[float] = None
    height: Optional[float] = None
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OCRPipeline:
    def __init__(
        self,
        paddle_lang: str = "en",
        tesseract_lang: str = "eng",
        prefer_paddle: bool = True,
        pdf_text_threshold: int = 40,
        render_scale: float = 2.0,
    ) -> None:
        self.paddle_lang = paddle_lang
        self.tesseract_lang = tesseract_lang
        self.prefer_paddle = prefer_paddle
        self.pdf_text_threshold = pdf_text_threshold
        self.render_scale = render_scale
        self._paddle = None

    def extract(self, filepath: str | Path) -> List[Dict[str, Any]]:
        path = Path(filepath)
        ext = path.suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            return [self._load_image(path).to_dict()]
        if ext in SUPPORTED_PDF_EXTS:
            if self._pdf_has_text(path):
                return [page.to_dict() for page in self._load_digital_pdf(path)]
            return [page.to_dict() for page in self._load_scanned_pdf(path)]
        raise ValueError(f"Unsupported file type: {ext}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Could not read image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = self._deskew(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=15)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(gray < 250))
        if len(coords) < 100:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.2:
            return gray
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _pdf_has_text(self, path: Path) -> bool:
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages[:3]:
                    if len((page.extract_text() or "").strip()) >= self.pdf_text_threshold:
                        return True
        except Exception:
            return False
        return False

    def _load_digital_pdf(self, path: Path) -> List[OCRPage]:
        pages: List[OCRPage] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                words = page.extract_words() or []
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        cells = [str(cell).strip() for cell in row if cell]
                        if cells:
                            text += "\n" + " | ".join(cells)
                bbox_list = [
                    OCRRegion(
                        text=w.get("text", ""),
                        bbox=[float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])],
                        engine="pdfplumber",
                    ).__dict__
                    for w in words
                    if w.get("text")
                ]
                ocr_page = OCRPage(
                    text=text.strip(),
                    page_no=page_no,
                    bbox_list=bbox_list,
                    source_file=path.name,
                    width=float(page.width),
                    height=float(page.height),
                    extraction_method="pdfplumber",
                )
                self._add_invoice_metadata(ocr_page)
                pages.append(ocr_page)
        return pages

    def _load_scanned_pdf(self, path: Path) -> List[OCRPage]:
        pages: List[OCRPage] = []
        doc = fitz.open(str(path))
        try:
            for page_no, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=fitz.Matrix(self.render_scale, self.render_scale), alpha=False)
                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                processed = self.preprocess_image(image)
                ocr_page = self._ocr_image(processed, page_no, path.name, page.rect.width, page.rect.height)
                self._add_invoice_metadata(ocr_page, image)
                ocr_page.extraction_method = "paddleocr" if self.prefer_paddle else "tesseract"
                pages.append(ocr_page)
        finally:
            doc.close()
        return pages

    def _load_image(self, path: Path) -> OCRPage:
        image = cv2.imread(str(path))
        processed = self.preprocess_image(image)
        h, w = processed.shape[:2]
        page = self._ocr_image(processed, 1, path.name, float(w), float(h))
        self._add_invoice_metadata(page, image)
        return page

    def _add_invoice_metadata(self, page: OCRPage, image: Optional[np.ndarray] = None) -> None:
        invoice_fields = extract_invoice_fields(image=image, text=page.text, lang=self.tesseract_lang)
        page.metadata = page.metadata or {}
        if invoice_fields:
            page.metadata["invoice_fields"] = invoice_fields
            field_text = self._invoice_fields_text(invoice_fields)
            if field_text and field_text not in page.text:
                page.text = f"{page.text}\n{field_text}".strip()

    def _ocr_image(self, image: np.ndarray, page_no: int, source_file: str, width: float, height: float) -> OCRPage:
        if self.prefer_paddle:
            try:
                page = self._paddle_ocr_image(image, page_no, source_file, width, height)
                if page.text.strip():
                    return page
            except Exception as exc:
                print(f"[OCRPipeline] PaddleOCR unavailable, falling back to Tesseract: {exc}")
        return self._tesseract_ocr_image(image, page_no, source_file, width, height)

    def _get_paddle(self):
        global PaddleOCR
        if PaddleOCR is None:
            from paddleocr import PaddleOCR as _PaddleOCR
            PaddleOCR = _PaddleOCR
        if self._paddle is None:
            self._paddle = PaddleOCR(lang=self.paddle_lang)
        return self._paddle

    def _paddle_ocr_image(self, image: np.ndarray, page_no: int, source_file: str, width: float, height: float) -> OCRPage:
        ocr = self._get_paddle()
        result = ocr.predict(image)
        regions: List[Dict[str, Any]] = []
        texts: List[str] = []
        if result:
            first = result[0]
            rec_texts = first.get("rec_texts", []) if isinstance(first, dict) else []
            rec_scores = first.get("rec_scores", []) if isinstance(first, dict) else []
            rec_boxes = first.get("rec_boxes", []) if isinstance(first, dict) else []
            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                bbox = self._normalize_box(box)
                if text:
                    texts.append(text)
                    regions.append(OCRRegion(text=text, bbox=bbox, conf=float(score), engine="paddleocr").__dict__)
        return OCRPage(" ".join(texts).strip(), page_no, regions, source_file, width, height, "paddleocr")

    def _tesseract_ocr_image(self, image: np.ndarray, page_no: int, source_file: str, width: float, height: float) -> OCRPage:
        if pytesseract is None:
            return OCRPage("", page_no, [], source_file, width, height, "unavailable")
        try:
            data = pytesseract.image_to_data(image, lang=self.tesseract_lang, output_type=pytesseract.Output.DICT)
        except Exception as exc:
            print(f"[OCRPipeline] Tesseract unavailable: {exc}")
            return OCRPage("", page_no, [], source_file, width, height, "unavailable")
        texts: List[str] = []
        regions: List[Dict[str, Any]] = []
        for i, text in enumerate(data.get("text", [])):
            text = (text or "").strip()
            if not text:
                continue
            conf = self._safe_float(data.get("conf", [0])[i], default=0.0)
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            texts.append(text)
            regions.append(OCRRegion(text=text, bbox=[x, y, x + w, y + h], conf=max(conf, 0.0) / 100.0, engine="tesseract").__dict__)
        return OCRPage(" ".join(texts).strip(), page_no, regions, source_file, width, height, "tesseract")

    @staticmethod
    def _invoice_fields_text(fields: Dict[str, Any]) -> str:
        labels = {
            "invoice_number": "Invoice Number",
            "date": "Invoice Date",
            "due_date": "Due Date",
            "subtotal": "Sub Total",
            "total": "Total",
            "amount_paid": "Amount Paid",
            "amount_due": "Amount Due",
            "total_in_words": "Total in Words",
        }
        lines = [f"{label}: {fields[key]}" for key, label in labels.items() if fields.get(key)]
        return "\n".join(lines)

    @staticmethod
    def _normalize_box(box: Any) -> List[float]:
        arr = np.array(box).astype(float)
        if arr.ndim == 1 and arr.size == 4:
            return [float(v) for v in arr.tolist()]
        arr = arr.reshape(-1, 2)
        x0, y0 = arr.min(axis=0)
        x1, y1 = arr.max(axis=0)
        return [float(x0), float(y0), float(x1), float(y1)]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default


def extract_document(filepath: str | Path, **kwargs: Any) -> List[Dict[str, Any]]:
    return OCRPipeline(**kwargs).extract(filepath)
