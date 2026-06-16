"""Cheap local text and metadata extraction for common file types."""

from __future__ import annotations

import csv
import email
import html.parser
import io
import mimetypes
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from email import policy
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .text import normalize_text, stable_id


TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".jsonl",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp"}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
CASE_RE = re.compile(
    r"\b(?:case|contract|invoice|req|rfp)(?:\s*(?:#|:|-)\s*|\s+)[A-Z0-9][A-Z0-9-]*\d[A-Z0-9-]*\b|"
    r"\bP\.?\s*O\.?\s*#?\s+[A-Z0-9][A-Z0-9-]*\d[A-Z0-9-]*\b",
    re.I,
)
CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z&.'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-zA-Z&.'-]+|[A-Z]{2,})){1,5}\b"
)
ORG_WORDS = {
    "inc",
    "inc.",
    "llc",
    "l.l.c.",
    "corp",
    "corp.",
    "corporation",
    "company",
    "co.",
    "department",
    "agency",
    "office",
    "university",
    "medical",
    "systems",
    "group",
    "partners",
}


@dataclass
class ExtractedFile:
    text: str
    pages: list[str]
    doc_type: str
    extraction_method: str
    metadata: dict[str, Any]
    warnings: list[str]


class _HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(self.parts)


def detect_doc_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".eml", ".msg"}:
        return "email"
    if suffix in {".csv", ".tsv", ".xls", ".xlsx"}:
        return "spreadsheet"
    if suffix in {".doc", ".docx", ".rtf"}:
        return "word_doc"
    if suffix in {".ppt", ".pptx"}:
        return "presentation"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in TEXT_SUFFIXES:
        return "text"
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("text/"):
        return "text"
    return "unknown"


def extract_file(path: Path) -> ExtractedFile:
    doc_type = detect_doc_type(path)
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    method = "native_text"

    try:
        if doc_type == "pdf":
            text, warnings, method, metadata = _extract_pdf(path)
        elif doc_type == "email":
            text, metadata, warnings = _extract_email(path)
            method = "email_parser"
        elif doc_type == "spreadsheet":
            text, metadata, warnings = _extract_spreadsheet(path)
            method = "spreadsheet_parser"
        elif doc_type == "word_doc":
            text, warnings = _extract_docx(path)
            method = "native_text" if text else "failed"
        elif doc_type == "presentation":
            text, warnings = _extract_pptx(path)
            method = "native_text" if text else "failed"
        elif doc_type == "html":
            text = _extract_html(path)
        elif doc_type == "image":
            text, warnings, ocr_confidence = _ocr_image(path)
            if ocr_confidence is not None:
                metadata["ocr_confidence"] = ocr_confidence
            method = "ocr" if text.strip() else "failed"
        else:
            text = _read_text(path)
            method = "native_text" if text else "failed"
            if not text:
                warnings.append("no text could be decoded")
    except Exception as exc:  # noqa: BLE001 - extraction should record failures, not crash corpus builds.
        text = ""
        method = "failed"
        warnings.append(f"extraction failed: {exc}")

    pages = _split_pages(text)
    text = "\n\f\n".join(pages)
    if not text.strip():
        warnings.append("document has no extracted text")
    return ExtractedFile(
        text=text,
        pages=pages,
        doc_type=doc_type,
        extraction_method=method,
        metadata=metadata,
        warnings=warnings,
    )


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if _looks_binary(raw):
        return ""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def _looks_binary(raw: bytes) -> bool:
    if not raw:
        return False
    if b"\x00" in raw[:4096]:
        return True
    sample = raw[:4096]
    control = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return control / len(sample) > 0.20


def _extract_html(path: Path) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(_read_text(path))
    return parser.text()


def _extract_pdf(path: Path) -> tuple[str, list[str], str, dict[str, Any]]:
    warnings: list[str] = []
    executable = shutil.which("pdftotext")
    if not executable:
        warnings.append("pdftotext is not installed; PDF native text extraction skipped")
    else:
        completed = subprocess.run(
            [executable, "-layout", "-enc", "UTF-8", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            warnings.append(f"pdftotext failed: {stderr or completed.returncode}")
        elif completed.stdout.strip():
            return completed.stdout, warnings, "native_text", {}
        else:
            warnings.append("pdftotext produced no usable text; attempting OCR fallback")

    ocr_text, ocr_warnings, ocr_confidence = _ocr_pdf(path)
    warnings.extend(ocr_warnings)
    if ocr_text.strip():
        metadata = {"ocr_confidence": ocr_confidence} if ocr_confidence is not None else {}
        return ocr_text, warnings, "ocr", metadata
    return "", warnings, "failed", {}


def _ocr_pdf(path: Path) -> tuple[str, list[str], float | None]:
    renderer = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not renderer:
        return "", ["pdftoppm is not installed; PDF OCR rendering skipped"], None
    if not tesseract:
        return "", ["tesseract is not installed; PDF OCR skipped"], None

    warnings: list[str] = []
    pages: list[str] = []
    confidences: list[float] = []
    with tempfile.TemporaryDirectory(prefix="fieldguide-ocr-") as tmp:
        prefix = Path(tmp) / "page"
        rendered = subprocess.run(
            [renderer, "-r", "200", "-png", str(path), str(prefix)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if rendered.returncode != 0:
            stderr = rendered.stderr.strip()
            return "", [f"pdftoppm failed for OCR fallback: {stderr or rendered.returncode}"], None
        images = sorted(Path(tmp).glob("page-*.png"), key=_rendered_page_sort_key)
        if not images:
            return "", ["pdftoppm produced no page images for OCR fallback"], None
        for page_number, image_path in enumerate(images, start=1):
            page_text, page_warnings, page_confidence = _run_tesseract(image_path, page_label=f"page {page_number}")
            warnings.extend(page_warnings)
            if page_confidence is not None:
                confidences.append(page_confidence)
            pages.append(page_text)
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    return "\f".join(pages), warnings, confidence


def _ocr_image(path: Path) -> tuple[str, list[str], float | None]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return "", ["tesseract is not installed; image OCR skipped"], None
    return _run_tesseract(path, page_label="image")


def _run_tesseract(path: Path, *, page_label: str) -> tuple[str, list[str], float | None]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return "", ["tesseract is not installed; OCR skipped"], None
    completed = subprocess.run(
        [tesseract, str(path), "stdout"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    warnings: list[str] = []
    confidence: float | None = None
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        warnings.append(f"tesseract failed on {page_label}: {stderr or completed.returncode}")
    elif completed.stderr.strip():
        # Tesseract often reports resolution estimates on stderr; keep them inspectable.
        warnings.append(f"tesseract note on {page_label}: {completed.stderr.strip()}")
    if completed.returncode == 0 and completed.stdout.strip():
        confidence, confidence_warnings = _tesseract_word_confidence(tesseract, path, page_label=page_label)
        warnings.extend(confidence_warnings)
    return completed.stdout, warnings, confidence


def _tesseract_word_confidence(tesseract: str, path: Path, *, page_label: str) -> tuple[float | None, list[str]]:
    completed = subprocess.run(
        [tesseract, str(path), "stdout", "tsv"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        return None, [f"tesseract confidence extraction failed on {page_label}: {stderr or completed.returncode}"]

    confidences: list[float] = []
    reader = csv.DictReader(io.StringIO(completed.stdout), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        raw_confidence = row.get("conf")
        if not text or raw_confidence is None:
            continue
        try:
            confidence = float(raw_confidence)
        except ValueError:
            continue
        if confidence >= 0:
            confidences.append(confidence)
    if not confidences:
        return None, []
    average = sum(confidences) / len(confidences) / 100
    return round(max(0.0, min(1.0, average)), 4), []


def _rendered_page_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-(\d+)\.png$", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _extract_email(path: Path) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    message = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    headers = {
        "from": str(message.get("from", "")),
        "to": str(message.get("to", "")),
        "cc": str(message.get("cc", "")),
        "subject": str(message.get("subject", "")),
        "message_id": str(message.get("message-id", "")),
    }
    parts = [
        f"From: {headers['from']}",
        f"To: {headers['to']}",
        f"Cc: {headers['cc']}",
        f"Subject: {headers['subject']}",
        "",
    ]
    attachment_names: list[str] = []
    bodies: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            disposition = part.get_content_disposition()
            if disposition == "attachment":
                filename = part.get_filename()
                if filename:
                    attachment_names.append(filename)
                continue
            if part.get_content_type() == "text/plain":
                try:
                    bodies.append(part.get_content())
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"email body part could not be decoded: {exc}")
    else:
        try:
            bodies.append(message.get_content())
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"email body could not be decoded: {exc}")
    if attachment_names:
        parts.append("Attachments: " + ", ".join(attachment_names))
        parts.append("")
    parts.extend(bodies)
    headers["attachments"] = attachment_names
    return "\n".join(parts), headers, warnings


def _extract_docx(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() != ".docx":
        return "", ["legacy Word format is not supported without external conversion"]
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except KeyError:
        return "", ["docx missing word/document.xml"]
    return _text_from_xml(xml, text_tags={"t"}), []


def _extract_pptx(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() != ".pptx":
        return "", ["legacy presentation format is not supported without external conversion"]
    parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(name for name in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml", name))
        for index, name in enumerate(slide_names, start=1):
            text = _text_from_xml(archive.read(name), text_tags={"t"})
            if text.strip():
                parts.append(f"Slide {index}\n{text}")
    return "\n\n".join(parts), []


def _extract_spreadsheet(path: Path) -> tuple[str, dict[str, Any], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows: list[str] = []
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row_number, row in enumerate(csv.reader(handle, delimiter=delimiter), start=1):
                rows.append(f"Row {row_number}: " + " | ".join(cell.strip() for cell in row))
        return "\n".join(rows), {"sheet_names": [path.stem]}, []
    if suffix != ".xlsx":
        return "", {}, ["legacy spreadsheet format is not supported without external conversion"]
    return _extract_xlsx(path)


def _extract_xlsx(path: Path) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    parts: list[str] = []
    sheet_names: list[str] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        workbook_names = _read_workbook_sheet_names(archive)
        sheet_paths = sorted(name for name in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", name))
        for index, sheet_path in enumerate(sheet_paths, start=1):
            sheet_name = workbook_names[index - 1] if index - 1 < len(workbook_names) else f"Sheet{index}"
            sheet_names.append(sheet_name)
            rows = _read_sheet_rows(archive.read(sheet_path), shared_strings)
            parts.append(f"Sheet: {sheet_name}")
            for row_index, cells in rows[:300]:
                projected = " | ".join(f"{ref}: {value}" for ref, value in cells if value)
                if projected:
                    parts.append(f"Row {row_index}: {projected}")
            if len(rows) > 300:
                warnings.append(f"{sheet_name} truncated to first 300 rows for indexing")
    return "\n".join(parts), {"sheet_names": sheet_names}, warnings


def _text_from_xml(xml: bytes, *, text_tags: set[str]) -> str:
    root = ElementTree.fromstring(xml)
    parts: list[str] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in text_tags and element.text:
            parts.append(element.text)
        elif tag in {"p", "tr"}:
            parts.append("\n")
    return normalize_text(" ".join(parts))


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(xml)
    values: list[str] = []
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] != "si":
            continue
        texts = [node.text or "" for node in item.iter() if node.tag.rsplit("}", 1)[-1] == "t"]
        values.append("".join(texts))
    return values


def _read_workbook_sheet_names(archive: zipfile.ZipFile) -> list[str]:
    try:
        xml = archive.read("xl/workbook.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(xml)
    names: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "sheet":
            name = element.attrib.get("name")
            if name:
                names.append(name)
    return names


def _read_sheet_rows(xml: bytes, shared_strings: list[str]) -> list[tuple[int, list[tuple[str, str]]]]:
    root = ElementTree.fromstring(xml)
    rows: list[tuple[int, list[tuple[str, str]]]] = []
    for row in root.iter():
        if row.tag.rsplit("}", 1)[-1] != "row":
            continue
        row_index = int(float(row.attrib.get("r", len(rows) + 1)))
        cells: list[tuple[str, str]] = []
        for cell in row:
            if cell.tag.rsplit("}", 1)[-1] != "c":
                continue
            ref = cell.attrib.get("r", "")
            cell_type = cell.attrib.get("t", "")
            value = ""
            for child in cell:
                if child.tag.rsplit("}", 1)[-1] == "v" and child.text is not None:
                    value = child.text
                    break
            if cell_type == "s" and value.isdigit():
                idx = int(value)
                value = shared_strings[idx] if idx < len(shared_strings) else value
            cells.append((ref, value))
        rows.append((row_index, cells))
    return rows


def _split_pages(text: str, max_page_chars: int = 7000) -> list[str]:
    if not text:
        return [""]
    if "\f" in text:
        pages = [normalize_text(page) for page in text.split("\f")]
        return [page for page in pages if page]
    if len(text) <= max_page_chars:
        return [text]
    pages: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_page_chars)
        if end < len(text):
            boundary = text.rfind("\n\n", start + max_page_chars // 2, end)
            if boundary == -1:
                boundary = text.rfind("\n", start + max_page_chars // 2, end)
            if boundary != -1:
                end = boundary
        pages.append(text[start:end].strip())
        start = end
    return [page for page in pages if page]


def extraction_quality(
    method: str,
    warnings: list[str],
    *,
    text_available: bool | None = None,
    ocr_confidence: float | None = None,
    layout_confidence: float | None = None,
) -> dict[str, Any]:
    if text_available is None:
        text_available = method != "failed"
    return {
        "text_available": text_available,
        "extraction_method": method,
        "ocr_confidence": ocr_confidence,
        "layout_confidence": layout_confidence,
        "table_extraction_quality": "unknown",
        "warnings": warnings,
    }


def extract_entities(text: str, *, doc_id: str, page_number: int | None = None, char_offset: int = 0) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []

    def add_span(start: int, end: int, raw: str, entity_type: str, confidence: float, extractor: str) -> None:
        raw = raw.strip()
        if not raw:
            return
        if not any(char.isalnum() for char in raw):
            return
        start = char_offset + start
        end = char_offset + end
        if any(_spans_overlap(start, end, existing["char_start"], existing["char_end"]) for existing in mentions):
            return
        mentions.append(
            {
                "entity_id": stable_id("E", entity_type, raw.lower(), length=10),
                "text": raw,
                "normalized_text": raw.lower(),
                "type": entity_type,
                "doc_id": doc_id,
                "page_number": page_number,
                "char_start": start,
                "char_end": end,
                "confidence": confidence,
                "extractor": extractor,
            }
        )

    nlp = _load_spacy_ner()
    if nlp is not None and text.strip():
        for ent in nlp(text).ents:
            entity_type = _spacy_entity_type(ent.label_)
            if entity_type:
                add_span(ent.start_char, ent.end_char, ent.text, entity_type, 0.82, "spacy")

    for regex, entity_type, confidence in (
        (EMAIL_RE, "email", 0.98),
        (PHONE_RE, "phone", 0.85),
        (MONEY_RE, "money", 0.95),
        (CASE_RE, "case_number", 0.75),
    ):
        for match in regex.finditer(text):
            add_span(match.start(), match.end(), match.group(0), entity_type, confidence, "regex")

    if nlp is None:
        for match in CAPITALIZED_PHRASE_RE.finditer(text):
            raw = match.group(0).strip()
            lowered = raw.lower().split()
            if raw.lower() in {"from", "subject"}:
                continue
            if len(raw) > 80:
                continue
            entity_type = "organization" if any(word.strip(",") in ORG_WORDS for word in lowered) else "unknown"
            add_span(match.start(), match.end(), raw, entity_type, 0.55, "capitalized_phrase")
    return mentions


@lru_cache(maxsize=1)
def _load_spacy_ner() -> Any | None:
    try:
        import spacy
    except ImportError:
        return None
    for model_name in ("en_core_web_md", "en_core_web_sm", "en_core_web_lg"):
        try:
            return spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        except OSError:
            continue
    return None


def _spacy_entity_type(label: str) -> str | None:
    return {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",
        "LOC": "location",
        "FAC": "location",
        "MONEY": "money",
    }.get(label)


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def file_metadata(path: Path, source_root: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        source_uri = str(path.relative_to(source_root))
    except ValueError:
        source_uri = str(path)
    return {
        "source_uri": source_uri,
        "original_filename": path.name,
        "size_bytes": stat.st_size,
    }
