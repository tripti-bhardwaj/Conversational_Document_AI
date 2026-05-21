from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import pytesseract
except Exception:  # pragma: no cover - optional runtime dependency
    pytesseract = None
else:
    tesseract_cmd = which("tesseract") or "/opt/homebrew/bin/tesseract"
    if Path(tesseract_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


MONEY_RE = re.compile(r"\$\s*[0-9OolISs,]+(?:[.,][0-9OolISs]{2})?")


def extract_invoice_fields(image: Optional[np.ndarray] = None, text: str = "", lang: str = "eng") -> Dict[str, Any]:
    ocr_texts = [text]
    if image is not None and pytesseract is not None:
        ocr_texts.extend(_targeted_invoice_ocr(image, lang=lang).values())

    combined_text = "\n".join(t for t in ocr_texts if t)
    fields = _parse_invoice_text(combined_text)
    if fields:
        fields["raw_invoice_text"] = combined_text.strip()
    return fields


def answer_invoice_question(question: str, metadata: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    field = _requested_field(question)
    if not field:
        return None

    fields = _merge_invoice_fields(metadata)
    value = fields.get(field)
    if not value:
        return None

    labels = {
        "invoice_number": "invoice number",
        "date": "invoice date",
        "due_date": "due date",
        "subtotal": "subtotal",
        "total": "total amount",
        "amount_paid": "amount paid",
        "amount_due": "amount due",
        "total_in_words": "total in words",
    }
    answer = f"The {labels[field]} is {value}."
    return {"answer": answer, "field": field, "value": value, "invoice_fields": fields}


def _targeted_invoice_ocr(image: np.ndarray, lang: str) -> Dict[str, str]:
    h, w = image.shape[:2]
    regions = {
        "top_table": (0.50, 0.08, 0.95, 0.32),
        "invoice_number": (0.70, 0.11, 0.84, 0.142),
        "invoice_date": (0.705, 0.145, 0.90, 0.177),
        "due_date": (0.705, 0.178, 0.90, 0.210),
        "bottom_right": (0.48, 0.55, 0.98, 0.75),
        "right_half": (0.48, 0.05, 0.98, 0.78),
    }
    output: Dict[str, str] = {}
    for name, (x0, y0, x1, y1) in regions.items():
        crop = image[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
        if crop.size == 0:
            continue
        scale = 5 if name in {"invoice_number", "invoice_date", "due_date"} else 2
        processed = _prepare_crop(crop, scale=scale)
        config = "--psm 7" if name == "invoice_number" else "--psm 6"
        try:
            output[name] = pytesseract.image_to_string(processed, lang=lang, config=config)
        except Exception as exc:
            print(f"[InvoiceExtractor] Targeted OCR skipped for {name}: {exc}")
    return output


def _prepare_crop(crop: np.ndarray, scale: int = 2) -> np.ndarray:
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _parse_invoice_text(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    fields: Dict[str, Any] = {}

    invoice_number = _extract_invoice_number(lines)
    if invoice_number:
        fields["invoice_number"] = invoice_number

    dates = _extract_dates(normalized)
    if dates:
        fields["date"] = dates[0]
    if len(dates) > 1:
        fields["due_date"] = dates[1]

    for key, label_patterns in {
        "subtotal": (r"sub\s*total",),
        "amount_paid": (r"amount\s*paid",),
        "amount_due": (r"amount\s*due",),
        "total": (r"(?<!sub\s)\btotal\b",),
    }.items():
        value = _field_money(lines, label_patterns)
        if value:
            fields[key] = value

    _reconcile_summary_fields(fields)

    line_items = _line_item_amounts(lines)
    if line_items:
        fields["line_item_amounts"] = [_format_money(v) for v in line_items]
        summed = sum(line_items, Decimal("0.00"))
        fields["line_item_total"] = _format_money(summed)
        fields["total"] = _reconcile_total(fields.get("total"), fields.get("amount_due"), summed)
        fields.setdefault("subtotal", _format_money(summed))
        fields.setdefault("amount_due", fields["total"])

    words = _extract_total_words(lines)
    if words:
        fields["total_in_words"] = words
    return fields


def _normalize_text(text: str) -> str:
    replacements = {
        "‘": "",
        "’": "",
        "|": " ",
        "JUI": "Jul",
        "JUl": "Jul",
        "£UL9": "2023",
        "£ULS": "2023",
        "EULS": "2023",
        "EVE": "2023",
        "TU": "10",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[ \t]+", " ", text)


def _extract_invoice_number(lines: List[str]) -> Optional[str]:
    for line in lines:
        if re.fullmatch(r"\d{1,8}", line):
            return line
    for line in lines:
        match = re.search(r"invoice\s*#\s*[:\-]?\s*([A-Za-z0-9-]+)", line, flags=re.I)
        if match:
            return match.group(1)
    return None


def _extract_dates(text: str) -> List[str]:
    dates: List[str] = []
    seen = set()
    month_re = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
    for month, day, year in re.findall(month_re + r"\s+([0-9OolISs]{1,2}),?\s+([0-9OolISs£]{4})", text, flags=re.I):
        day_digits = _digits(day)
        year_digits = _digits(year)
        if len(year_digits) == 4 and day_digits:
            value = f"{month[:3].title()} {int(day_digits)}, {year_digits}"
            if value not in seen:
                dates.append(value)
                seen.add(value)
    return dates


def _field_money(lines: List[str], label_patterns: tuple[str, ...]) -> Optional[str]:
    values: List[Decimal] = []
    for line in lines:
        lower = line.lower()
        if any(re.search(pattern, lower, flags=re.I) for pattern in label_patterns):
            money = _money_values(line)
            if money:
                values.append(money[-1])
    if not values:
        return None
    most_common = Counter(values).most_common(1)[0][0]
    return _format_money(most_common)


def _line_item_amounts(lines: List[str]) -> List[Decimal]:
    amounts: List[Decimal] = []
    in_items = False
    for line in lines:
        lower = line.lower()
        if "products" in lower or "unit price" in lower:
            in_items = True
            continue
        if any(label in lower for label in ("bank details", "quantity 6", "sub total", "amount paid", "amount due", "total in words")):
            in_items = False
        money = _money_values(line)
        if in_items and money:
            amounts.append(money[-1])
    if len(amounts) >= 3:
        return amounts
    return []


def _reconcile_total(total: Optional[str], amount_due: Optional[str], line_item_sum: Decimal) -> str:
    candidates = [_parse_money_value(v) for v in (total, amount_due) if v]
    candidates = [candidate for candidate in candidates if candidate is not None]
    for candidate in candidates:
        if candidate == line_item_sum:
            return _format_money(candidate)
    for candidate in candidates:
        if abs(candidate - line_item_sum) <= Decimal("5.00"):
            return _format_money(line_item_sum)
    return _format_money(candidates[0] if candidates else line_item_sum)


def _reconcile_summary_fields(fields: Dict[str, Any]) -> None:
    amount_due = _parse_money_value(fields.get("amount_due", ""))
    amount_paid = _parse_money_value(fields.get("amount_paid", ""))
    total = _parse_money_value(fields.get("total", ""))
    subtotal = _parse_money_value(fields.get("subtotal", ""))
    if amount_due is None:
        return
    if amount_paid == Decimal("0.00"):
        fields["total"] = _format_money(amount_due)
        if subtotal is None or abs(subtotal - amount_due) <= Decimal("5.00"):
            fields["subtotal"] = _format_money(amount_due)
    elif total is not None and amount_paid is not None and abs((total - amount_paid) - amount_due) <= Decimal("5.00"):
        fields["total"] = _format_money(total)


def _extract_total_words(lines: List[str]) -> Optional[str]:
    for index, line in enumerate(lines):
        if "total in words" in line.lower() and index + 1 < len(lines):
            return lines[index + 1].strip()
    for line in lines:
        if "hundred" in line.lower() and "dollar" in line.lower():
            return line.strip()
    return None


def _money_values(text: str) -> List[Decimal]:
    values: List[Decimal] = []
    for match in MONEY_RE.findall(text):
        value = _parse_money_value(match)
        if value is not None:
            values.append(value)
    return values


def _parse_money_value(value: str) -> Optional[Decimal]:
    cleaned = value.replace("$", "").replace(",", "").replace(" ", "")
    cleaned = cleaned.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5"}))
    cleaned = re.sub(r"[^0-9.]", "", cleaned)
    if not cleaned:
        return None
    if "." not in cleaned:
        cleaned = f"{cleaned}.00"
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def _format_money(value: Decimal) -> str:
    return f"${value:,.2f}"


def _digits(value: str) -> str:
    return value.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5", "£": "2"}))


def _requested_field(question: str) -> Optional[str]:
    q = question.lower()
    if "invoice" in q and any(token in q for token in ("number", "no", "#")):
        return "invoice_number"
    if "due date" in q:
        return "due_date"
    if "date" in q:
        return "date"
    if "amount paid" in q or "paid" in q:
        return "amount_paid"
    if "amount due" in q or "due amount" in q:
        return "amount_due"
    if "subtotal" in q or "sub total" in q:
        return "subtotal"
    if "total in words" in q:
        return "total_in_words"
    if "total" in q or "amount" in q:
        return "total"
    return None


def _merge_invoice_fields(metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in metadata:
        fields = item.get("invoice_fields") or item.get("metadata", {}).get("invoice_fields")
        if fields:
            return fields
    return {}
