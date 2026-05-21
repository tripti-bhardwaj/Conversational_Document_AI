from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fitz


def highlight_pdf(pdf_path, source_chunks, max_highlights=3, output_path: Optional[str] = None):
    doc = fitz.open(str(pdf_path))
    try:
        for chunk in source_chunks[:max_highlights]:
            page_index = int(chunk.get("page_no", 1)) - 1
            if page_index < 0 or page_index >= len(doc):
                continue
            page = doc[page_index]
            for item in chunk.get("bbox_list", [])[:80]:
                bbox = item.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                rect = fitz.Rect([float(v) for v in bbox])
                if rect.is_empty or rect.is_infinite:
                    continue
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.92, 0.0))
                annot.set_opacity(0.35)
                annot.update()
        if output_path:
            doc.save(output_path, garbage=4, deflate=True)
            return output_path
        return doc.tobytes(garbage=4, deflate=True)
    finally:
        doc.close()
