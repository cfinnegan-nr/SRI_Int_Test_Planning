"""Generate an HTML integration test plan for SRI AML GA.

This module reads the integration diagram image from the project root and
produces a self-contained HTML report with the test strategy and detailed
test cases for integrated data flows across Sensa Risk Intelligence.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import os
import random
import re
import struct
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DIAGRAM_FILENAME = "SRI_Integration_Diagram.png"
I_PHASES_DIAGRAM_FILENAME = "SRI_Integration_I_Phases.png"
I_PHASES_DOCX_FILENAME = "SRI Int Tests IPhases.docx"
REPORT_FILENAME = "sri_int_test_plan.html"
AUTHOR_NAME = "Ciaran Finnegan"
CONFLUENCE_REPORT_TITLE = "SRI AML GA Integration Test Plan Report"
CONFLUENCE_PARENT_PAGE_ID = "949420050"
CONFLUENCE_REPORT_PAGE_ID = "1249018170"
CONFLUENCE_SPACE_KEY = "HARMONY"
CONFLUENCE_ENV_FILE = Path(
    r"C:\Sensa_NR\2025\QA\GenAI\AINative_Env\.env"
)
CONFLUENCE_REPORT_PAGE_URL = (
    "https://netreveal.atlassian.net/wiki/spaces/HARMONY/pages/1249018170/"
    "SRI+AML+GA+Integration+Test+Plan+Report"
)
CONFLUENCE_ARCHITECTURE_PAGE_URL = (
    "https://netreveal.atlassian.net/wiki/spaces/HARMONY/pages/949420050/"
    "Sensa+Risk+Intelligence+-+SRI+AML+GA+Architecture"
)
I_PHASE_IDS = ("I1", "I2", "I3", "I4", "I5", "I6", "I7")
PHASE_FLOW_MAP: Dict[str, Dict[str, str]] = {
    "I1": {
        "title": "Subject updates into Investigation",
        "trace": "Sensa Data -> Kafka Subject Updated Events -> "
        "Sensa Investigation",
        "source": "Sensa Data",
        "target": "Sensa Investigation",
        "event": "Subject Updated Events",
        "data_focus": "subject and account updates",
    },
    "I2": {
        "title": "Batch ingestion triggers AML inference",
        "trace": "Sensa Data -> Data Quality -> AML Inference trigger",
        "source": "Sensa Data",
        "target": "Sensa Detection",
        "event": "AML Inference trigger",
        "data_focus": "batch transaction ingestion",
    },
    "I3": {
        "title": "Detection events generate investigations",
        "trace": "Decisioning -> Kafka Detection Events -> "
        "Sensa Investigation",
        "source": "Sensa Detection",
        "target": "Sensa Investigation",
        "event": "Detection Events",
        "data_focus": "detection event payloads",
    },
    "I4": {
        "title": "Direct transaction feed to AML inference",
        "trace": "MinIO (direct) -> AML Inference (batch)",
        "source": "MinIO direct feed",
        "target": "Sensa Detection",
        "event": "AML Inference batch job",
        "data_focus": "direct transaction batches",
    },
    "I5": {
        "title": "Historical data drives training and inference",
        "trace": "OLAP (Iceberg) -> Sensa Training -> AML Inference",
        "source": "Sensa Data OLAP",
        "target": "Sensa Detection",
        "event": "Model training output",
        "data_focus": "historic transaction data",
    },
    "I6": {
        "title": "Investigation search against OLTP data",
        "trace": "Sensa Investigation -> OLTP (PostgreSQL) "
        "transaction search",
        "source": "Sensa Investigation",
        "target": "OLTP Store",
        "event": "Transaction search",
        "data_focus": "transaction search queries",
    },
    "I7": {
        "title": "Investigation to Sensa Agent assistance",
        "trace": "Sensa Investigation -> Sensa Agent "
        "(Operational Assistants / Investigator Copilot)",
        "source": "Sensa Investigation",
        "target": "Sensa Agent",
        "event": "Assistant interaction",
        "data_focus": "case context and evidence bundles",
    },
}


class PlanGenerationError(RuntimeError):
    """Base exception for plan generation failures."""


class DiagramLoadError(PlanGenerationError):
    """Raised when the integration diagram cannot be loaded."""


class ReportWriteError(PlanGenerationError):
    """Raised when the HTML report cannot be written."""


class RenderError(PlanGenerationError):
    """Raised when the HTML report cannot be rendered."""


class ConfluencePublishError(PlanGenerationError):
    """Raised when the Confluence report cannot be published."""


@dataclass(frozen=True)
class TestStep:
    """A single executable step in a test case."""

    description: str
    expected_result: str
    suggested_data_format: str


@dataclass(frozen=True)
class TestCase:
    """A single integration test case with steps and data guidance."""

    case_id: str
    title: str
    purpose: str
    trace_to_flow: str
    data_notes: List[str]
    steps: List[TestStep]


@dataclass(frozen=True)
class ImageFocus:
    """Deprecated focus metadata retained for legacy plan builders."""

    object_position: str
    scale: float
    height_px: int
    caption: str


@dataclass(frozen=True)
class SectionImage:
    """Image metadata and content for report sections."""

    data_uri: str
    caption: str
    alt_text: str
    hand_drawn: bool
    filename: str
    image_bytes: bytes


@dataclass(frozen=True)
class PlanSection:
    """Content for a single report section."""

    section_id: str
    title: str
    description: List[str]
    section_image: SectionImage
    strategy_sections: List[tuple[str, List[str]]]
    integration_flows: List[str]
    out_of_scope: List[str]
    assumptions: List[str]
    test_cases: List[TestCase]


@dataclass(frozen=True)
class TestPlan:
    """High-level plan with strategy content and test cases."""

    title: str
    author: str
    report_date: str
    objective: str
    sections: List[PlanSection]


@dataclass(frozen=True)
class ConfluenceConfig:
    """Settings required to publish a report to Confluence."""

    base_url: str
    user_email: str
    api_token: str
    space_key: str
    parent_page_id: str
    report_page_id: str


def _normalize_base_url(base_url: str) -> str:
    """Normalize the Confluence base URL for API usage."""
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/wiki"):
        cleaned = cleaned[:-5]
    return cleaned


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file.
    """
    if not env_path.exists():
        LOGGER.info("Confluence env file not found at %s", env_path)
        return

    try:
        contents = env_path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Failed to read env file at %s: %s", env_path, exc)
        return

    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_confluence_config() -> Optional[ConfluenceConfig]:
    """Load Confluence configuration from environment variables.

    Returns:
        ConfluenceConfig if all required values are present, otherwise None.
    """
    load_env_file(CONFLUENCE_ENV_FILE)
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    user_email = os.getenv("CONFLUENCE_USER_EMAIL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not base_url or not user_email or not api_token:
        LOGGER.warning(
            "Confluence publish skipped. Set CONFLUENCE_BASE_URL, "
            "CONFLUENCE_USER_EMAIL, and CONFLUENCE_API_TOKEN to enable."
        )
        return None

    space_key = os.getenv("CONFLUENCE_SPACE_KEY", CONFLUENCE_SPACE_KEY)
    parent_page_id = os.getenv(
        "CONFLUENCE_PARENT_PAGE_ID",
        CONFLUENCE_PARENT_PAGE_ID,
    )
    report_page_id = os.getenv(
        "CONFLUENCE_REPORT_PAGE_ID",
        CONFLUENCE_REPORT_PAGE_ID,
    )

    return ConfluenceConfig(
        base_url=_normalize_base_url(base_url),
        user_email=user_email,
        api_token=api_token,
        space_key=space_key,
        parent_page_id=parent_page_id,
        report_page_id=report_page_id,
    )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a PNG chunk with CRC."""
    length = struct.pack(">I", len(data))
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return length + chunk_type + data + struct.pack(">I", crc & 0xFFFFFFFF)


def _paeth_predictor(a: int, b: int, c: int) -> int:
    """Compute the Paeth predictor for PNG filtering."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


@dataclass
class _PngImage:
    """Simple in-memory RGBA PNG representation."""

    width: int
    height: int
    pixels: bytearray

    @classmethod
    def from_bytes(cls, png_bytes: bytes) -> "_PngImage":
        """Parse a PNG file into an RGBA pixel buffer.

        Supports 8-bit RGBA PNGs with no interlace.
        """
        signature = b"\x89PNG\r\n\x1a\n"
        if not png_bytes.startswith(signature):
            raise DiagramLoadError("Unsupported PNG signature.")

        offset = len(signature)
        width = height = 0
        bit_depth = color_type = None
        idat_data = bytearray()

        while offset < len(png_bytes):
            if offset + 8 > len(png_bytes):
                break
            length = struct.unpack(">I", png_bytes[offset:offset + 4])[0]
            chunk_type = png_bytes[offset + 4:offset + 8]
            chunk_start = offset + 8
            chunk_end = chunk_start + length
            chunk_data = png_bytes[chunk_start:chunk_end]
            offset = chunk_end + 4

            if chunk_type == b"IHDR":
                width, height, bit_depth, color_type, _, _, _ = struct.unpack(
                    ">IIBBBBB",
                    chunk_data,
                )
            elif chunk_type == b"IDAT":
                idat_data.extend(chunk_data)
            elif chunk_type == b"IEND":
                break

        if bit_depth != 8 or color_type not in {2, 6}:
            raise DiagramLoadError(
                "Only 8-bit RGB or RGBA PNGs are supported for diagram processing."
            )

        try:
            decompressed = zlib.decompress(bytes(idat_data))
        except zlib.error as exc:
            raise DiagramLoadError("Failed to decompress PNG data.") from exc

        bytes_per_pixel = 4 if color_type == 6 else 3
        row_bytes = width * bytes_per_pixel
        expected = (row_bytes + 1) * height
        if len(decompressed) < expected:
            raise DiagramLoadError("PNG data is incomplete after decompression.")

        pixels = bytearray(row_bytes * height)
        prior_row = bytearray(row_bytes)
        cursor = 0
        out_offset = 0

        for _ in range(height):
            filter_type = decompressed[cursor]
            cursor += 1
            row_data = decompressed[cursor:cursor + row_bytes]
            cursor += row_bytes
            row = bytearray(row_bytes)

            if filter_type == 0:
                row[:] = row_data
            elif filter_type == 1:
                for i in range(row_bytes):
                    left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                    row[i] = (row_data[i] + left) & 0xFF
            elif filter_type == 2:
                for i in range(row_bytes):
                    row[i] = (row_data[i] + prior_row[i]) & 0xFF
            elif filter_type == 3:
                for i in range(row_bytes):
                    left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                    avg = (left + prior_row[i]) // 2
                    row[i] = (row_data[i] + avg) & 0xFF
            elif filter_type == 4:
                for i in range(row_bytes):
                    left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                    up = prior_row[i]
                    up_left = (
                        prior_row[i - bytes_per_pixel]
                        if i >= bytes_per_pixel
                        else 0
                    )
                    row[i] = (row_data[i] + _paeth_predictor(left, up, up_left)) & 0xFF
            else:
                raise DiagramLoadError(
                    f"Unsupported PNG filter type: {filter_type}"
                )

            pixels[out_offset:out_offset + row_bytes] = row
            out_offset += row_bytes
            prior_row = row

        if color_type == 2:
            rgba_pixels = bytearray(width * height * 4)
            src_index = 0
            for dest_index in range(0, len(rgba_pixels), 4):
                r = pixels[src_index]
                g = pixels[src_index + 1]
                b = pixels[src_index + 2]
                rgba_pixels[dest_index:dest_index + 4] = bytes((r, g, b, 255))
                src_index += 3
            pixels = rgba_pixels

        return cls(width=width, height=height, pixels=pixels)

    def to_bytes(self) -> bytes:
        """Encode the RGBA pixel buffer back to a PNG."""
        bytes_per_pixel = 4
        row_bytes = self.width * bytes_per_pixel
        raw = bytearray()
        for row_idx in range(self.height):
            start = row_idx * row_bytes
            raw.append(0)
            raw.extend(self.pixels[start:start + row_bytes])

        compressed = zlib.compress(bytes(raw), level=6)
        ihdr = struct.pack(
            ">IIBBBBB",
            self.width,
            self.height,
            8,
            6,
            0,
            0,
            0,
        )
        return b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _png_chunk(b"IHDR", ihdr),
                _png_chunk(b"IDAT", compressed),
                _png_chunk(b"IEND", b""),
            ]
        )

    def crop(self, left: int, top: int, width: int, height: int) -> "_PngImage":
        """Crop the image to the requested rectangle."""
        left = max(0, min(left, self.width - 1))
        top = max(0, min(top, self.height - 1))
        width = max(1, min(width, self.width - left))
        height = max(1, min(height, self.height - top))
        row_bytes = self.width * 4
        cropped = bytearray(width * height * 4)

        for row in range(height):
            src_start = (top + row) * row_bytes + left * 4
            src_end = src_start + width * 4
            dest_start = row * width * 4
            cropped[dest_start:dest_start + width * 4] = self.pixels[
                src_start:src_end
            ]

        return _PngImage(width=width, height=height, pixels=cropped)

    def apply_hand_drawn(self, seed: int) -> "_PngImage":
        """Apply a rough hand-drawn effect to the image."""
        rng = random.Random(seed)
        pixels = bytearray(self.pixels)
        total_pixels = self.width * self.height

        for idx in range(0, len(pixels), 4):
            r, g, b, a = pixels[idx:idx + 4]
            if a == 0:
                continue
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            noise = rng.randint(-18, 18)
            r = min(255, max(0, int(gray * 0.55 + r * 0.45) + noise))
            g = min(255, max(0, int(gray * 0.55 + g * 0.45) + noise))
            b = min(255, max(0, int(gray * 0.55 + b * 0.45) + noise))
            pixels[idx:idx + 4] = bytes((r, g, b, a))

        scratch_count = max(4, total_pixels // 25000)
        for _ in range(scratch_count):
            y = rng.randint(0, self.height - 1)
            scratch_strength = rng.randint(12, 36)
            for x in range(self.width):
                idx = (y * self.width + x) * 4
                r, g, b, a = pixels[idx:idx + 4]
                if a == 0:
                    continue
                r = max(0, r - scratch_strength)
                g = max(0, g - scratch_strength)
                b = max(0, b - scratch_strength)
                pixels[idx:idx + 4] = bytes((r, g, b, a))

        border_thickness = max(2, self.width // 250)
        for y in range(self.height):
            for x in range(self.width):
                if (
                    x < border_thickness
                    or x >= self.width - border_thickness
                    or y < border_thickness
                    or y >= self.height - border_thickness
                ):
                    idx = (y * self.width + x) * 4
                    r, g, b, a = pixels[idx:idx + 4]
                    if a == 0:
                        continue
                    r = max(0, r - 30)
                    g = max(0, g - 30)
                    b = max(0, b - 30)
                    pixels[idx:idx + 4] = bytes((r, g, b, a))

        return _PngImage(width=self.width, height=self.height, pixels=pixels)


def _encode_png_data_uri(image_bytes: bytes) -> str:
    """Encode PNG bytes as a data URI for HTML rendering."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _read_docx_paragraphs(docx_path: Path) -> List[str]:
    """Extract paragraph text from a DOCX file."""
    if not docx_path.exists():
        LOGGER.warning("DOCX file not found at %s", docx_path)
        return []

    try:
        with zipfile.ZipFile(docx_path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        LOGGER.error("Failed to read DOCX contents: %s", exc)
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        LOGGER.error("Failed to parse DOCX XML: %s", exc)
        return []

    paragraphs: List[str] = []
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespace):
        parts: List[str] = []
        for text_node in paragraph.findall(".//w:t", namespace):
            if text_node.text:
                parts.append(text_node.text)
        text = " ".join(part.strip() for part in parts).strip()
        if text:
            paragraphs.append(re.sub(r"\s+", " ", text))

    return paragraphs


def _extract_phase_text(docx_path: Path) -> Dict[str, List[str]]:
    """Extract I1-I7 phase descriptions from the DOCX paragraphs."""
    paragraphs = _read_docx_paragraphs(docx_path)
    phase_map = {phase_id: [] for phase_id in I_PHASE_IDS}
    current_phase: Optional[str] = None
    phase_regex = re.compile(r"\bI([1-7])\b", re.IGNORECASE)

    for paragraph in paragraphs:
        match = phase_regex.search(paragraph)
        if match:
            current_phase = f"I{match.group(1)}"
            remainder = phase_regex.sub("", paragraph).strip(":- \t")
            if remainder:
                phase_map[current_phase].append(remainder)
            continue
        if current_phase:
            phase_map[current_phase].append(paragraph)

    return phase_map


def _build_section_images() -> Dict[str, SectionImage]:
    """Create section images (including hand-drawn variants)."""
    try:
        base_diagram_bytes = (PROJECT_ROOT / DIAGRAM_FILENAME).read_bytes()
        i_phases_bytes = (PROJECT_ROOT / I_PHASES_DIAGRAM_FILENAME).read_bytes()
    except OSError as exc:
        LOGGER.error("Failed to load diagram inputs: %s", exc)
        raise DiagramLoadError("Unable to load required diagram inputs.") from exc

    phase_diagram = _PngImage.from_bytes(i_phases_bytes)

    def crop_by_ratio(
        image: _PngImage,
        left_ratio: float,
        top_ratio: float,
        width_ratio: float,
        height_ratio: float,
    ) -> _PngImage:
        left = int(image.width * left_ratio)
        top = int(image.height * top_ratio)
        width = int(image.width * width_ratio)
        height = int(image.height * height_ratio)
        return image.crop(left, top, width, height)

    day_zero_crop = crop_by_ratio(phase_diagram, 0.02, 0.05, 0.96, 0.55)
    day_one_crop = crop_by_ratio(phase_diagram, 0.00, 0.18, 1.00, 0.60)
    day_two_crop = crop_by_ratio(phase_diagram, 0.40, 0.28, 0.58, 0.68)

    day_zero_image = day_zero_crop.apply_hand_drawn(seed=101)
    day_one_image = day_one_crop.apply_hand_drawn(seed=202)
    day_two_image = day_two_crop.apply_hand_drawn(seed=303)

    section_images = {
        "high-level": SectionImage(
            data_uri=_encode_png_data_uri(base_diagram_bytes),
            caption="Architecture focus: end-to-end SRI AML GA data flows.",
            alt_text="SRI AML GA architecture overview",
            hand_drawn=False,
            filename="sri-aml-ga-architecture.png",
            image_bytes=base_diagram_bytes,
        ),
        "day-zero": SectionImage(
            data_uri=_encode_png_data_uri(day_zero_image.to_bytes()),
            caption="Day Zero focus: I1 data-to-investigation readiness.",
            alt_text="Day Zero integration focus diagram",
            hand_drawn=True,
            filename="sri-aml-day-zero-hand-drawn.png",
            image_bytes=day_zero_image.to_bytes(),
        ),
        "day-one": SectionImage(
            data_uri=_encode_png_data_uri(day_one_image.to_bytes()),
            caption="Day 1 focus: I2-I4 detection and investigation flows.",
            alt_text="Day 1 integration focus diagram",
            hand_drawn=True,
            filename="sri-aml-day-one-hand-drawn.png",
            image_bytes=day_one_image.to_bytes(),
        ),
        "day-two": SectionImage(
            data_uri=_encode_png_data_uri(day_two_image.to_bytes()),
            caption="Day 2 focus: I5-I7 end-to-end and agent coverage.",
            alt_text="Day 2 integration focus diagram",
            hand_drawn=True,
            filename="sri-aml-day-two-hand-drawn.png",
            image_bytes=day_two_image.to_bytes(),
        ),
    }

    return section_images


def _summarize_phase_text(phase_id: str, paragraphs: List[str]) -> str:
    """Create a concise summary for an integration phase."""
    if not paragraphs:
        return f"{phase_id} phase description unavailable in source document."
    summary = " ".join(paragraphs[:2]).strip()
    return summary if summary else f"{phase_id} phase description unavailable."


def _build_phase_data_notes(
    phase_id: str,
    phase_summary: str,
    flow_detail: Dict[str, str],
) -> List[str]:
    """Create test data guidance notes for a phase."""
    notes = [
        f"Phase reference: {phase_summary}",
        "Generate synthetic data aligned to the SRI AML GA IFS schema.",
        "Embed correlation IDs and timestamps for traceability.",
    ]
    if phase_id in {"I1", "I2", "I4", "I6"}:
        notes.append(
            "Use deterministic subject, account, and transaction IDs for "
            "reconciliation checks."
        )
    if phase_id in {"I3", "I7"}:
        notes.append(
            "Capture evidence artifacts (event payloads, transcripts, UI "
            "snapshots) for audit trails."
        )
    if phase_id == "I5":
        notes.append(
            "Seed OLAP history with at least one training period for model "
            "version validation."
        )
    return notes


def _build_phase_steps(
    phase_id: str,
    flow_detail: Dict[str, str],
    case_type: str,
) -> List[TestStep]:
    """Generate granular test steps for a phase."""
    if phase_id == "I1":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Prepare a minimal subject update batch with a unique "
                        "run ID and deterministic subject/account identifiers."
                    ),
                    expected_result=(
                        "Batch validates against the SRI AML GA IFS schema "
                        "with all mandatory fields present."
                    ),
                    suggested_data_format=(
                        "CSV/Parquet subject update file; strategy: synthetic "
                        "records with consistent IDs and timestamps."
                    ),
                ),
                TestStep(
                    description=(
                        "Ingest the batch through Sensa Data and confirm Data "
                        "Quality completion."
                    ),
                    expected_result=(
                        "Ingestion completes with a Data Quality pass status."
                    ),
                    suggested_data_format=(
                        "MinIO batch file plus Data Quality report output."
                    ),
                ),
                TestStep(
                    description=(
                        "Verify Subject Updated Events are published to Kafka "
                        "with the run ID."
                    ),
                    expected_result=(
                        "Events match the ingested subjects and include "
                        "correlation identifiers."
                    ),
                    suggested_data_format=(
                        "Kafka JSON payloads with subject_id, run_id, "
                        "timestamp."
                    ),
                ),
                TestStep(
                    description=(
                        "Confirm Sensa Investigation reflects the updated "
                        "subject details."
                    ),
                    expected_result=(
                        "Investigation case context displays the new subject "
                        "attributes."
                    ),
                    suggested_data_format=(
                        "Investigation UI/API snapshot with updated fields."
                    ),
                ),
                TestStep(
                    description="Capture traceability evidence across systems.",
                    expected_result=(
                        "Logs and payloads show consistent run IDs from "
                        "ingestion to investigation."
                    ),
                    suggested_data_format=(
                        "Log extracts from Sensa Data, Kafka, Investigation."
                    ),
                ),
            ]
        return [
            TestStep(
                description=(
                    "Create a subject update batch with missing mandatory "
                    "fields."
                ),
                expected_result="Data Quality flags the batch as failed.",
                suggested_data_format=(
                    "CSV/Parquet with missing subject identifiers or "
                    "mandatory attributes."
                ),
            ),
            TestStep(
                description=(
                    "Confirm no Subject Updated Events are published to Kafka."
                ),
                expected_result="Kafka topics show no new events for the run.",
                suggested_data_format=(
                    "Kafka consumer output filtered by run ID."
                ),
            ),
            TestStep(
                description=(
                    "Verify Investigation does not apply the invalid update."
                ),
                expected_result=(
                    "Subject context remains unchanged for the affected IDs."
                ),
                suggested_data_format="Investigation UI/API snapshot.",
            ),
        ]

    if phase_id == "I2":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Generate a compliant transaction batch with "
                        "risk-relevant attributes and a unique run ID."
                    ),
                    expected_result=(
                        "Batch passes schema validation and is accepted for "
                        "processing."
                    ),
                    suggested_data_format=(
                        "Transaction CSV/Parquet with amounts, timestamps, "
                        "counterparty IDs."
                    ),
                ),
                TestStep(
                    description=(
                        "Ingest the batch via Sensa Data and confirm Data "
                        "Quality completion."
                    ),
                    expected_result="Data Quality reports a pass status.",
                    suggested_data_format="Data Quality summary report output.",
                ),
                TestStep(
                    description=(
                        "Validate the AML Inference trigger fires for the "
                        "batch."
                    ),
                    expected_result="AML Inference job starts for the run ID.",
                    suggested_data_format="Inference job metadata with run ID.",
                ),
                TestStep(
                    description=(
                        "Confirm inference output contains risk scores "
                        "linked to the original transactions."
                    ),
                    expected_result=(
                        "Each transaction has a risk score and traceable ID."
                    ),
                    suggested_data_format=(
                        "Inference output records with transaction IDs."
                    ),
                ),
            ]
        return [
            TestStep(
                description=(
                    "Prepare a transaction batch that violates Data Quality "
                    "rules."
                ),
                expected_result="Data Quality marks the batch as failed.",
                suggested_data_format=(
                    "Transactions missing mandatory fields or invalid types."
                ),
            ),
            TestStep(
                description="Confirm AML Inference is not triggered.",
                expected_result="No inference job is created for the batch.",
                suggested_data_format="Inference scheduler logs.",
            ),
            TestStep(
                description="Validate error reporting and triage evidence.",
                expected_result="Error logs include rule violations and batch ID.",
                suggested_data_format="Data Quality failure report.",
            ),
        ]

    if phase_id == "I3":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Run AML Inference to generate high-risk detection "
                        "events."
                    ),
                    expected_result=(
                        "Decisioning produces Detection Events for high-risk "
                        "transactions."
                    ),
                    suggested_data_format=(
                        "Inference output with risk thresholds exceeded."
                    ),
                ),
                TestStep(
                    description="Consume Detection Events from Kafka.",
                    expected_result="Events are published with valid schema.",
                    suggested_data_format=(
                        "Kafka JSON with event ID, subject ID, risk score."
                    ),
                ),
                TestStep(
                    description="Verify Investigation creates cases per event.",
                    expected_result=(
                        "Investigation cases include event ID and risk data."
                    ),
                    suggested_data_format="Investigation case export.",
                ),
                TestStep(
                    description="Validate event-to-case traceability.",
                    expected_result="Case fields match event payload values.",
                    suggested_data_format="Case comparison checklist.",
                ),
            ]
        return [
            TestStep(
                description="Publish a malformed Detection Event to Kafka.",
                expected_result="Investigation rejects or flags the event.",
                suggested_data_format=(
                    "Kafka payload missing required event fields."
                ),
            ),
            TestStep(
                description="Confirm no invalid case is created.",
                expected_result="Investigation UI shows no new case for the event.",
                suggested_data_format="Investigation activity log.",
            ),
            TestStep(
                description="Review error handling and alerting.",
                expected_result="Validation errors are logged with event ID.",
                suggested_data_format="Investigation error logs.",
            ),
        ]

    if phase_id == "I4":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Prepare a direct MinIO transaction batch for Sensa "
                        "Detection with a run ID."
                    ),
                    expected_result=(
                        "Batch is recognized by the direct ingestion job."
                    ),
                    suggested_data_format=(
                        "Transactions file with timestamps and counterparties."
                    ),
                ),
                TestStep(
                    description="Trigger AML Inference via the direct feed.",
                    expected_result="Inference job starts for the direct batch.",
                    suggested_data_format="Inference job metadata record.",
                ),
                TestStep(
                    description="Validate inference outputs and enrichment.",
                    expected_result=(
                        "Detection outputs include scores and enrichment data."
                    ),
                    suggested_data_format="Inference output with risk scores.",
                ),
            ]
        return [
            TestStep(
                description=(
                    "Upload a direct transaction batch with schema errors."
                ),
                expected_result=(
                    "Direct ingestion rejects the batch and logs validation "
                    "errors."
                ),
                suggested_data_format="Malformed transaction CSV.",
            ),
            TestStep(
                description="Confirm no inference run is triggered.",
                expected_result="Inference scheduler shows no job for the batch.",
                suggested_data_format="Inference scheduler logs.",
            ),
            TestStep(
                description="Capture error evidence for triage.",
                expected_result="Logs capture batch ID and error summary.",
                suggested_data_format="Error log excerpt.",
            ),
        ]

    if phase_id == "I5":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Seed OLAP with historical transactions covering at "
                        "least one training period."
                    ),
                    expected_result=(
                        "OLAP partitions are available for the target period."
                    ),
                    suggested_data_format="Iceberg partitions by date range.",
                ),
                TestStep(
                    description="Initiate a Sensa Training run.",
                    expected_result="Training completes with a new model version.",
                    suggested_data_format="Training job metadata and model ID.",
                ),
                TestStep(
                    description="Run AML Inference using the updated model.",
                    expected_result=(
                        "Inference metadata references the new model version."
                    ),
                    suggested_data_format="Inference output with model version.",
                ),
            ]
        return [
            TestStep(
                description=(
                    "Attempt a training run with insufficient OLAP history."
                ),
                expected_result="Training fails with a data sufficiency error.",
                suggested_data_format="OLAP dataset missing required partitions.",
            ),
            TestStep(
                description=(
                    "Confirm inference continues to use the last stable model."
                ),
                expected_result="Inference metadata shows the prior model version.",
                suggested_data_format="Inference output model metadata.",
            ),
            TestStep(
                description="Capture training failure diagnostics.",
                expected_result="Logs include missing data details and run ID.",
                suggested_data_format="Training error logs.",
            ),
        ]

    if phase_id == "I6":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Ensure OLTP contains transactions from the latest "
                        "batch load."
                    ),
                    expected_result="OLTP rows exist for target transaction IDs.",
                    suggested_data_format="OLTP query output by transaction ID.",
                ),
                TestStep(
                    description=(
                        "Execute Investigation transaction search with the "
                        "known IDs."
                    ),
                    expected_result="Search results return matching transactions.",
                    suggested_data_format="Investigation search criteria payload.",
                ),
                TestStep(
                    description="Validate field-level accuracy.",
                    expected_result="Search results match OLTP values.",
                    suggested_data_format="Comparison checklist for key fields.",
                ),
            ]
        return [
            TestStep(
                description=(
                    "Search for a transaction that is absent from OLTP."
                ),
                expected_result="Investigation returns no results.",
                suggested_data_format="Search criteria with nonexistent ID.",
            ),
            TestStep(
                description="Confirm error handling for missing data.",
                expected_result="UI/API returns a clear no-results message.",
                suggested_data_format="Investigation response payload.",
            ),
        ]

    if phase_id == "I7":
        if case_type == "positive":
            return [
                TestStep(
                    description=(
                        "Open a case with multiple alerts and request "
                        "Operational Assistant guidance."
                    ),
                    expected_result=(
                        "Assistant response references the case context and "
                        "suggested actions."
                    ),
                    suggested_data_format=(
                        "Case summary including alert IDs and subject details."
                    ),
                ),
                TestStep(
                    description=(
                        "Request Investigator Copilot analysis on the same "
                        "case."
                    ),
                    expected_result=(
                        "Copilot summary aligns with evidence and risk "
                        "indicators."
                    ),
                    suggested_data_format="Case timeline and evidence bundle.",
                ),
                TestStep(
                    description="Capture assistant outputs for audit.",
                    expected_result="Transcripts stored with case reference ID.",
                    suggested_data_format="Assistant transcript export.",
                ),
            ]
        return [
            TestStep(
                description=(
                    "Request assistant output with incomplete case context."
                ),
                expected_result=(
                    "Assistant responds with a request for more information "
                    "or flags missing context."
                ),
                suggested_data_format="Case payload missing key fields.",
            ),
            TestStep(
                description="Verify assistant guardrails and logging.",
                expected_result="Logs capture the missing context warning.",
                suggested_data_format="Assistant interaction logs.",
            ),
        ]

    return []


def _build_phase_test_cases(
    day_prefix: str,
    phase_id: str,
    phase_summary: str,
) -> List[TestCase]:
    """Build positive and negative test cases for a phase."""
    flow_detail = PHASE_FLOW_MAP.get(phase_id, {})
    title_prefix = flow_detail.get("title", "Integration flow")
    trace = flow_detail.get("trace", "Integration flow traceability")
    data_notes = _build_phase_data_notes(phase_id, phase_summary, flow_detail)

    positive_case = TestCase(
        case_id=f"{day_prefix}-{phase_id}-01",
        title=f"{title_prefix} - happy path",
        purpose=(
            f"Validate {phase_id} flow: {phase_summary}"
        ),
        trace_to_flow=trace,
        data_notes=data_notes,
        steps=_build_phase_steps(phase_id, flow_detail, "positive"),
    )

    negative_case = TestCase(
        case_id=f"{day_prefix}-{phase_id}-02",
        title=f"{title_prefix} - error handling",
        purpose=(
            f"Validate negative scenarios for {phase_id} flow: {phase_summary}"
        ),
        trace_to_flow=trace,
        data_notes=data_notes,
        steps=_build_phase_steps(phase_id, flow_detail, "negative"),
    )

    return [positive_case, negative_case]


def _build_day_cases(
    day_prefix: str,
    phase_ids: Iterable[str],
    phase_text_map: Dict[str, List[str]],
) -> List[TestCase]:
    """Build all test cases for a specific day."""
    cases: List[TestCase] = []
    for phase_id in phase_ids:
        summary = _summarize_phase_text(phase_id, phase_text_map.get(phase_id, []))
        cases.extend(_build_phase_test_cases(day_prefix, phase_id, summary))
    return cases


def _render_confluence_paragraphs(paragraphs: Iterable[str]) -> str:
    """Render a list of paragraphs for Confluence storage format."""
    return "".join(f"<p>{html.escape(text)}</p>" for text in paragraphs)


def _render_confluence_test_case(test_case: TestCase) -> str:
    """Render a Confluence-friendly block for a test case."""
    steps_html = []
    for step in test_case.steps:
        steps_html.append(
            "<li>"
            f"<p><strong>Step:</strong> {html.escape(step.description)}</p>"
            f"<p><strong>Expected:</strong> "
            f"{html.escape(step.expected_result)}</p>"
            f"<p><strong>Suggested data format:</strong> "
            f"{html.escape(step.suggested_data_format)}</p>"
            "</li>"
        )

    return (
        "<div>"
        f"<h4>{html.escape(test_case.case_id)}: "
        f"{html.escape(test_case.title)}</h4>"
        f"<p><strong>Purpose:</strong> "
        f"{html.escape(test_case.purpose)}</p>"
        f"<p><strong>Traceability:</strong> "
        f"{html.escape(test_case.trace_to_flow)}</p>"
        "<p><strong>Test data notes:</strong></p>"
        f"<ul>{_render_list(test_case.data_notes)}</ul>"
        "<ol>"
        f"{''.join(steps_html)}"
        "</ol>"
        "</div>"
    )


def _render_confluence_section_image(section: PlanSection) -> str:
    """Render a Confluence section image with visual emphasis."""
    image = section.section_image
    filename = html.escape(image.filename, quote=True)
    caption = html.escape(image.caption)
    border_style = (
        "border:4px solid #334e68;"
        "border-left:12px solid #243b53;"
        "padding:14px;"
        "background:#f8fafc;"
        "margin:12px 0 20px 0;"
    )
    return (
        f"<div style=\"{border_style}\">"
        "<ac:image ac:align=\"center\">"
        f"<ri:attachment ri:filename=\"{filename}\" />"
        "</ac:image>"
        f"<p><em>{caption}</em></p>"
        "</div>"
    )


def build_confluence_body(plan: TestPlan) -> str:
    """Build a Confluence storage-format body for the report."""
    sections_html = []
    for section in plan.sections:
        section_parts = [f"<h2>{html.escape(section.title)}</h2>"]
        section_parts.append(_render_confluence_section_image(section))

        if section.description:
            section_parts.append(_render_confluence_paragraphs(section.description))

        for title, bullets in section.strategy_sections:
            section_parts.append(f"<h3>{html.escape(title)}</h3>")
            section_parts.append(f"<ul>{_render_list(bullets)}</ul>")

        if section.integration_flows:
            section_parts.append("<h3>Integration Flows Under Test</h3>")
            section_parts.append(
                f"<ol>{_render_list(section.integration_flows)}</ol>"
            )

        if section.out_of_scope:
            section_parts.append("<h3>Out of Scope</h3>")
            section_parts.append(f"<ul>{_render_list(section.out_of_scope)}</ul>")

        if section.assumptions:
            section_parts.append("<h3>Assumptions</h3>")
            section_parts.append(f"<ul>{_render_list(section.assumptions)}</ul>")

        if section.test_cases:
            section_parts.append("<h3>Test Cases</h3>")
            section_parts.append(
                "".join(
                    _render_confluence_test_case(test_case)
                    for test_case in section.test_cases
                )
            )

        sections_html.append(
            "<div style=\"border:4px solid #334e68;"
            "border-left:12px solid #243b53;"
            "padding:18px;margin-bottom:24px;"
            "background:#f8fafc;\">"
            f"{''.join(section_parts)}"
            "</div>"
        )

    architecture_url = html.escape(
        CONFLUENCE_ARCHITECTURE_PAGE_URL,
        quote=True,
    )

    return (
        f"<h1>{html.escape(plan.title)}</h1>"
        f"<p><strong>Author:</strong> {html.escape(plan.author)}</p>"
        f"<p><strong>Date:</strong> {html.escape(plan.report_date)}</p>"
        f"<p><strong>Objective:</strong> {html.escape(plan.objective)}</p>"
        f"<p><strong>Architecture reference:</strong> "
        f"<a href=\"{architecture_url}\">"
        "SRI AML GA Architecture</a></p>"
        f"{''.join(sections_html)}"
    )


def create_confluence_page(
    config: ConfluenceConfig,
    title: str,
    body: str,
) -> Tuple[str, str]:
    """Create a Confluence page and return its ID and URL.

    Args:
        config: Confluence configuration settings.
        title: Title of the new Confluence page.
        body: Storage-format body for the page.

    Returns:
        Tuple of (page_id, page_url) for the created Confluence page.

    Raises:
        ConfluencePublishError: If the page cannot be created.
    """
    payload = {
        "type": "page",
        "title": title,
        "ancestors": [{"id": config.parent_page_id}],
        "space": {"key": config.space_key},
        "body": {
            "storage": {
                "value": body,
                "representation": "storage",
            }
        },
    }

    request_url = f"{config.base_url}/wiki/rest/api/content"
    encoded_payload = json.dumps(payload).encode("utf-8")
    auth_token = base64.b64encode(
        f"{config.user_email}:{config.api_token}".encode("utf-8")
    ).decode("ascii")
    request = urllib.request.Request(
        request_url,
        data=encoded_payload,
        method="POST",
    )
    request.add_header("Authorization", f"Basic {auth_token}")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        LOGGER.error(
            "Confluence publish failed with status %s: %s",
            exc.code,
            error_body,
        )
        raise ConfluencePublishError(
            "Failed to publish Confluence page (HTTP "
            f"{exc.code}) for user {config.user_email} at {config.base_url}."
        ) from exc
    except urllib.error.URLError as exc:
        LOGGER.error("Confluence publish request failed: %s", exc)
        raise ConfluencePublishError(
            "Failed to connect to Confluence for publishing "
            f"as {config.user_email} at {config.base_url}."
        ) from exc

    try:
        payload_response = json.loads(response_body)
        page_id = payload_response.get("id")
        links = payload_response.get("_links", {})
        web_ui = links.get("webui")
        base_link = links.get("base") or config.base_url
        if not web_ui or not page_id:
            raise KeyError("Required page metadata not found")
        return page_id, f"{base_link}{web_ui}"
    except (json.JSONDecodeError, KeyError) as exc:
        LOGGER.error(
            "Unexpected Confluence response while creating page: %s",
            response_body,
        )
        raise ConfluencePublishError(
            "Failed to parse Confluence publish response for "
            f"{config.user_email} at {config.base_url}."
        ) from exc


def _build_basic_auth_header(config: ConfluenceConfig) -> str:
    """Build the Confluence Basic auth header."""
    token = base64.b64encode(
        f"{config.user_email}:{config.api_token}".encode("utf-8")
    ).decode("ascii")
    return f"Basic {token}"


def _extract_page_id_from_url(page_url: str) -> Optional[str]:
    """Extract a Confluence page ID from a URL."""
    match = re.search(r"/pages/(\d+)", page_url)
    if not match:
        return None
    return match.group(1)


def _get_report_page_url() -> str:
    """Return the configured report page URL."""
    return os.getenv(
        "CONFLUENCE_REPORT_PAGE_URL",
        CONFLUENCE_REPORT_PAGE_URL,
    )


def _get_confluence_page_version(config: ConfluenceConfig, page_id: str) -> int:
    """Fetch the current Confluence page version."""
    request_url = (
        f"{config.base_url}/wiki/rest/api/content/{page_id}?expand=version"
    )
    request = urllib.request.Request(request_url, method="GET")
    request.add_header("Authorization", _build_basic_auth_header(config))
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        LOGGER.error(
            "Confluence version lookup failed with status %s: %s",
            exc.code,
            error_body,
        )
        raise ConfluencePublishError(
            "Failed to fetch Confluence page version (HTTP "
            f"{exc.code}) for page {page_id}, user {config.user_email} "
            f"at {config.base_url}."
        ) from exc
    except urllib.error.URLError as exc:
        LOGGER.error("Confluence version lookup request failed: %s", exc)
        raise ConfluencePublishError(
            "Failed to connect to Confluence for version lookup "
            f"as {config.user_email} at {config.base_url}."
        ) from exc

    try:
        payload = json.loads(response_body)
        return int(payload["version"]["number"])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        LOGGER.error(
            "Unexpected Confluence response while reading version: %s",
            response_body,
        )
        raise ConfluencePublishError(
            "Failed to parse Confluence page version response for page "
            f"{page_id}, user {config.user_email} at {config.base_url}."
        ) from exc


def _validate_report_page_reference(config: ConfluenceConfig) -> None:
    """Validate that the configured page ID matches the report page URL."""
    report_url = _get_report_page_url()
    if report_url != CONFLUENCE_REPORT_PAGE_URL:
        LOGGER.info(
            "Confluence report page URL overridden via environment. "
            "Expected: %s. Actual: %s.",
            CONFLUENCE_REPORT_PAGE_URL,
            report_url,
        )
    extracted_id = _extract_page_id_from_url(report_url)
    if extracted_id is None:
        LOGGER.warning(
            "Confluence report URL does not include a page ID: %s",
            report_url,
        )
        return
    if extracted_id != config.report_page_id:
        raise ConfluencePublishError(
            "Confluence report page mismatch. Page ID configured as "
            f"{config.report_page_id} but URL points to {extracted_id}. "
            f"User: {config.user_email}. Base URL: {config.base_url}. "
            f"Report URL: {report_url}."
        )


def update_confluence_page(
    config: ConfluenceConfig,
    page_id: str,
    title: str,
    body: str,
) -> str:
    """Update an existing Confluence page and return its URL."""
    current_version = _get_confluence_page_version(config, page_id)
    payload = {
        "id": page_id,
        "type": "page",
        "title": title,
        "space": {"key": config.space_key},
        "version": {"number": current_version + 1},
        "body": {
            "storage": {
                "value": body,
                "representation": "storage",
            }
        },
    }

    request_url = f"{config.base_url}/wiki/rest/api/content/{page_id}"
    encoded_payload = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        request_url,
        data=encoded_payload,
        method="PUT",
    )
    request.add_header("Authorization", _build_basic_auth_header(config))
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        LOGGER.error(
            "Confluence update failed with status %s: %s",
            exc.code,
            error_body,
        )
        raise ConfluencePublishError(
            "Failed to update Confluence page (HTTP "
            f"{exc.code}) for user {config.user_email} at {config.base_url}."
        ) from exc
    except urllib.error.URLError as exc:
        LOGGER.error("Confluence update request failed: %s", exc)
        raise ConfluencePublishError(
            "Failed to connect to Confluence for update "
            f"as {config.user_email} at {config.base_url}."
        ) from exc

    try:
        payload_response = json.loads(response_body)
        links = payload_response.get("_links", {})
        web_ui = links.get("webui")
        base_link = links.get("base") or config.base_url
        if not web_ui:
            raise KeyError("webui link not found")
        return f"{base_link}{web_ui}"
    except (json.JSONDecodeError, KeyError) as exc:
        LOGGER.error(
            "Unexpected Confluence response while updating page: %s",
            response_body,
        )
        raise ConfluencePublishError(
            "Failed to parse Confluence update response for "
            f"{config.user_email} at {config.base_url}."
        ) from exc


def delete_confluence_page(config: ConfluenceConfig, page_id: str) -> None:
    """Delete a Confluence page by ID."""
    request_url = f"{config.base_url}/wiki/rest/api/content/{page_id}"
    request = urllib.request.Request(request_url, method="DELETE")
    request.add_header("Authorization", _build_basic_auth_header(config))
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 404:
            LOGGER.warning(
                "Confluence page %s not found for delete. "
                "User: %s. Base URL: %s.",
                page_id,
                config.user_email,
                config.base_url,
            )
            return
        LOGGER.error(
            "Confluence delete failed with status %s: %s",
            exc.code,
            error_body,
        )
        raise ConfluencePublishError(
            "Failed to delete Confluence page (HTTP "
            f"{exc.code}) for user {config.user_email} at {config.base_url}."
        ) from exc
    except urllib.error.URLError as exc:
        LOGGER.error("Confluence delete request failed: %s", exc)
        raise ConfluencePublishError(
            "Failed to connect to Confluence for delete "
            f"as {config.user_email} at {config.base_url}."
        ) from exc


def _upload_confluence_attachment(
    config: ConfluenceConfig,
    page_id: str,
    filename: str,
    content: bytes,
) -> None:
    """Upload or update a Confluence attachment."""
    boundary = f"----SRIFormBoundary{random.randint(100000, 999999)}"
    header = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\n"
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = b"".join([header, content, footer])

    request_url = (
        f"{config.base_url}/wiki/rest/api/content/{page_id}/child/attachment"
    )
    request = urllib.request.Request(
        request_url,
        data=body,
        method="POST",
    )
    request.add_header("Authorization", _build_basic_auth_header(config))
    request.add_header("X-Atlassian-Token", "no-check")
    request.add_header(
        "Content-Type",
        f"multipart/form-data; boundary={boundary}",
    )
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        LOGGER.error(
            "Confluence attachment upload failed with status %s: %s",
            exc.code,
            error_body,
        )
        raise ConfluencePublishError(
            "Failed to upload Confluence attachment (HTTP "
            f"{exc.code}) for user {config.user_email} at {config.base_url}."
        ) from exc
    except urllib.error.URLError as exc:
        LOGGER.error("Confluence attachment upload failed: %s", exc)
        raise ConfluencePublishError(
            "Failed to connect to Confluence for attachment upload "
            f"as {config.user_email} at {config.base_url}."
        ) from exc


def _preflight_confluence_access(config: ConfluenceConfig) -> None:
    """Validate access to the target Confluence page before uploads."""
    _validate_report_page_reference(config)
    try:
        current_version = _get_confluence_page_version(
            config=config,
            page_id=config.report_page_id,
        )
    except ConfluencePublishError as exc:
        report_url = _get_report_page_url()
        raise ConfluencePublishError(
            "Confluence access check failed. Verify the API token belongs to "
            "a licensed user with access to the HARMONY space and the report "
            "page ID configured for this run. "
            f"User: {config.user_email}. Base URL: {config.base_url}. "
            f"Report URL: {report_url}."
        ) from exc

    LOGGER.info(
        "Confluence access check passed for page %s (version %s).",
        config.report_page_id,
        current_version,
    )


def publish_report_to_confluence(plan: TestPlan) -> Optional[str]:
    """Publish the generated report to Confluence when configured."""
    config = load_confluence_config()
    if config is None:
        return None

    _preflight_confluence_access(config)

    body = build_confluence_body(plan)
    delete_confluence_page(config=config, page_id=config.report_page_id)
    new_page_id, new_page_url = create_confluence_page(
        config=config,
        title=CONFLUENCE_REPORT_TITLE,
        body=body,
    )

    for section in plan.sections:
        try:
            _upload_confluence_attachment(
                config=config,
                page_id=new_page_id,
                filename=section.section_image.filename,
                content=section.section_image.image_bytes,
            )
        except ConfluencePublishError:
            LOGGER.warning(
                "Attachment upload failed for %s",
                section.section_image.filename,
            )
            raise

    return new_page_url


def load_diagram_data_uri(diagram_path: Path) -> str:
    """Load the integration diagram and return a data URI string.

    Args:
        diagram_path: Path to the integration diagram image.

    Returns:
        A data URI string suitable for embedding in HTML.

    Raises:
        DiagramLoadError: If the diagram cannot be read.
    """
    try:
        diagram_bytes = diagram_path.read_bytes()
        encoded = base64.b64encode(diagram_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError as exc:
        LOGGER.error(
            "Integration diagram not found at %s",
            diagram_path,
        )
        raise DiagramLoadError(
            f"Integration diagram not found: {diagram_path}"
        ) from exc
    except OSError as exc:
        LOGGER.error(
            "Failed reading integration diagram at %s: %s",
            diagram_path,
            exc,
        )
        raise DiagramLoadError(
            f"Failed to read integration diagram: {diagram_path}"
        ) from exc


def _build_test_plan_legacy() -> TestPlan:
    """Create the test plan using the SRI AML GA architecture context.

    Returns:
        A populated TestPlan object with strategy and test cases.
    """
    try:
        # Always stamp the report with today's local date for traceability.
        report_date = datetime.now().strftime("%B %d, %Y")

        strategy_sections = [
            (
                "Scope and Objectives",
                [
                    "Validate end-to-end batch data ingestion, AML inference, "
                    "and investigation workflows across Sensa Data, Sensa "
                    "Detection, Sensa Investigation, and Sensa Agent.",
                    "Confirm integration points between MinIO feeds, OLAP/OLTP "
                    "stores, Kafka events, and investigation search.",
                    "Demonstrate the AML GA release operates in batch mode "
                    "only and uses MinIO as the sole source feed.",
                ],
            ),
            (
                "Architecture Summary",
                [
                    "MinIO feeds are ingested by Sensa Data (Data Connectors) "
                    "and validated by Data Quality before landing in OLAP "
                    "(Iceberg) and OLTP (PostgreSQL).",
                    "Sensa Detection ingests transactions directly from MinIO "
                    "for AML Inference (batch) and uses OLAP data for training.",
                    "Decisioning Service enriches detection events with OLTP "
                    "data and publishes Detection Events to Kafka.",
                    "Sensa Investigation consumes Subject Updated and "
                    "Detection Events from Kafka and queries OLTP for search.",
                    "Sensa Agent capabilities (Operational Assistants and "
                    "Investigator Copilot) are accessed from Investigation.",
                ],
            ),
            (
                "Test Approach",
                [
                    "Exercise each integration flow with batch payloads "
                    "aligned to the SRI AML GA IFS schema.",
                    "Validate positive flows and error handling "
                    "(data quality failures, missing references).",
                    "Capture evidence from logs, Kafka topics, and UI/API "
                    "responses for traceability.",
                ],
            ),
            (
                "Environments and Test Data",
                [
                    "Use non-production MinIO buckets with controlled batch "
                    "payloads (transactions, subjects, accounts).",
                    "Seed OLTP reference data required for enrichment and "
                    "investigation search.",
                    "Use Kafka topics dedicated to test execution to isolate "
                    "events and ensure clean assertions.",
                ],
            ),
            (
                "Entry and Exit Criteria",
                [
                    "Entry: Batch ingestion endpoints available, Kafka topics "
                    "provisioned, and OLAP/OLTP accessible.",
                    "Exit: All integration test cases executed with expected "
                    "results and defects triaged.",
                ],
            ),
        ]

        integration_flows = [
            "MinIO -> Sensa Data (Data Connectors) -> Data Quality -> OLAP/OLTP",
            "MinIO (direct) -> AML Inference (batch) in Sensa Detection",
            "Data Quality -> AML Inference trigger",
            "OLAP (historic data) -> Sensa Training",
            "AML Inference -> Decisioning Service -> Kafka Detection Events",
            "Sensa Data -> Kafka Subject Updated Events -> Sensa Investigation",
            "Sensa Investigation -> OLTP Transaction Search",
            "Sensa Investigation -> Sensa Agent (Operational Assistants, "
            "Investigator Copilot)",
        ]

        out_of_scope = [
            "Real-time ingestion or streaming processing.",
            "Data ingestion from sources other than MinIO.",
            "Ontology-driven access for AML inference or investigation search.",
            "Entity Resolution, Watchlist Management, and Data Mapping Agent.",
        ]

        assumptions = [
            "SRI AML GA IFS schema is the agreed format for test data.",
            "Kafka topics and schemas are available for Subject Updated and "
            "Detection Events.",
            "OLTP contains required reference data for Decisioning enrichment.",
        ]

        # High-level strategy section drives the overall narrative.
        high_level_section = PlanSection(
            section_id="high-level",
            title="Integrated Test Plan Strategy",
            description=[
                "This section documents the integrated test strategy for the "
                "SRI AML GA release, aligned to the approved architecture.",
                "The focus is on batch-only processing, MinIO-based ingestion, "
                "and the core flows between Sensa Data, Detection, "
                "Investigation, and Agents.",
            ],
            image_focus=ImageFocus(
                object_position="50% 50%",
                scale=1.0,
                height_px=300,
                caption="Architecture focus: end-to-end SRI AML GA data flows.",
            ),
            strategy_sections=strategy_sections,
            integration_flows=integration_flows,
            out_of_scope=out_of_scope,
            assumptions=assumptions,
            test_cases=[],
        )

        # Day Zero: short sanity checks to validate core flow readiness.
        day_zero_cases = [
            TestCase(
                case_id="DZ-01",
                title="Environment and connectivity smoke check",
                purpose=(
                    "Confirm core infrastructure endpoints are reachable "
                    "before executing batch integrations."
                ),
                trace_to_flow="MinIO / Kafka / OLAP / OLTP readiness",
                data_notes=[
                    "Use service account credentials for MinIO, Kafka, and DBs.",
                    "Prepare a unique test run ID for logging correlation.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Authenticate to MinIO and list ingestion buckets "
                            "for Sensa Data and Sensa Detection."
                        ),
                        expected_result=(
                            "MinIO access succeeds and required buckets are "
                            "listed without permission errors."
                        ),
                        suggested_data_format=(
                            "N/A (connectivity check)."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Publish and consume a sample message on the "
                            "Subject Updated and Detection Events Kafka topics."
                        ),
                        expected_result=(
                            "Kafka round-trip succeeds with no schema errors."
                        ),
                        suggested_data_format=(
                            "Minimal JSON event payload with run ID and "
                            "timestamp."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Run a simple OLAP and OLTP query for seeded "
                            "reference data."
                        ),
                        expected_result=(
                            "Queries return expected results and no "
                            "connectivity or permission issues are seen."
                        ),
                        suggested_data_format=(
                            "SQL query output or Iceberg partition metadata."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="DZ-02",
                title="Minimal batch ingestion through Data Quality",
                purpose=(
                    "Verify a single small batch can be ingested, validated, "
                    "and stored in OLAP/OLTP."
                ),
                trace_to_flow="MinIO -> Sensa Data -> Data Quality -> OLAP/OLTP",
                data_notes=[
                    "SRI AML GA IFS schema with 1-3 transactions and subjects.",
                    "Include deterministic IDs for reconciliation.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Upload a minimal batch file to the MinIO "
                            "ingestion bucket."
                        ),
                        expected_result=(
                            "Sensa Data registers an ingestion job and picks "
                            "up the batch without delay."
                        ),
                        suggested_data_format=(
                            "CSV or Parquet file aligned to SRI AML GA IFS."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Review Data Quality results for the batch."
                        ),
                        expected_result=(
                            "Data Quality reports a pass status with no "
                            "mandatory-field errors."
                        ),
                        suggested_data_format=(
                            "Include mandatory fields only, no optional errors."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Confirm OLAP and OLTP record counts match the "
                            "batch payload."
                        ),
                        expected_result=(
                            "Stored counts and key fields match the source."
                        ),
                        suggested_data_format=(
                            "Record-level reconciliation using source IDs."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="DZ-03",
                title="Direct transaction feed to AML Inference (batch)",
                purpose=(
                    "Ensure Sensa Detection can ingest a direct transaction "
                    "batch from MinIO and execute inference."
                ),
                trace_to_flow="MinIO -> AML Inference (Batch)",
                data_notes=[
                    "Transactions must match SRI AML GA IFS schema.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Place a transaction batch file in the Sensa "
                            "Detection MinIO feed location."
                        ),
                        expected_result=(
                            "AML Inference job is triggered for the batch."
                        ),
                        suggested_data_format=(
                            "Transactions file with timestamps, amounts, and "
                            "counterparty identifiers."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Verify inference outputs are produced and tagged "
                            "with the batch run ID."
                        ),
                        expected_result=(
                            "Each transaction receives a risk score or label."
                        ),
                        suggested_data_format=(
                            "Detection output records with transaction IDs and "
                            "risk scores."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="DZ-04",
                title="Decisioning publishes Detection Events to Kafka",
                purpose=(
                    "Validate Decisioning enrichment and Detection Event "
                    "publishing for a minimal inference run."
                ),
                trace_to_flow="Decisioning -> Kafka Detection Events",
                data_notes=[
                    "OLTP reference data must exist for enrichment fields.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Confirm Decisioning retrieves OLTP enrichment "
                            "data for the inference output."
                        ),
                        expected_result=(
                            "Detection events include enriched subject and "
                            "account attributes."
                        ),
                        suggested_data_format=(
                            "OLTP reference rows for subject and account IDs."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Consume the Detection Events topic for the run."
                        ),
                        expected_result=(
                            "Kafka messages are published and schema valid."
                        ),
                        suggested_data_format=(
                            "Detection event JSON with risk score and "
                            "transaction ID."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="DZ-05",
                title="Investigation consumes detection events (sanity)",
                purpose=(
                    "Confirm Sensa Investigation can ingest Detection Events "
                    "from Kafka and create a basic investigation record."
                ),
                trace_to_flow="Kafka Detection Events -> Sensa Investigation",
                data_notes=[
                    "Use a single detection event with required fields.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Publish a valid Detection Event to Kafka."
                        ),
                        expected_result=(
                            "Sensa Investigation consumes the event without "
                            "errors."
                        ),
                        suggested_data_format=(
                            "Detection event JSON with transaction ID, "
                            "risk score, and subject reference."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Locate the created investigation record."
                        ),
                        expected_result=(
                            "An investigation case exists and references the "
                            "detection event ID."
                        ),
                        suggested_data_format=(
                            "Case record showing event ID and initial status."
                        ),
                    ),
                ],
            ),
        ]

        # Day One: deeper integration between Detection, Investigation, Agents.
        day_one_cases = [
            TestCase(
                case_id="D1-01",
                title="Detection event creates investigation case",
                purpose=(
                    "Validate end-to-end flow from inference output to "
                    "investigation case creation."
                ),
                trace_to_flow="Decisioning -> Kafka -> Sensa Investigation",
                data_notes=[
                    "Use a batch with multiple transactions and varied risk.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Run AML Inference for a transaction batch and "
                            "capture Decisioning output."
                        ),
                        expected_result=(
                            "Decisioning emits Detection Events for each "
                            "high-risk transaction."
                        ),
                        suggested_data_format=(
                            "Transactions file with risk-relevant attributes."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Verify Sensa Investigation consumes Detection "
                            "Events and creates cases."
                        ),
                        expected_result=(
                            "Investigation cases are created with correct "
                            "risk scores and transaction references."
                        ),
                        suggested_data_format=(
                            "Detection event JSON with event ID and score."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Confirm case details match Detection Event data."
                        ),
                        expected_result=(
                            "Case fields align with event metadata and subject "
                            "attributes."
                        ),
                        suggested_data_format=(
                            "Case view data exported for verification."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D1-02",
                title="Subject Updated Events refresh investigation context",
                purpose=(
                    "Ensure subject updates from Sensa Data appear in "
                    "investigation views."
                ),
                trace_to_flow="Sensa Data -> Kafka -> Sensa Investigation",
                data_notes=[
                    "Update subject attributes used in investigation display.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Ingest an updated subject record and monitor the "
                            "Subject Updated Events topic."
                        ),
                        expected_result=(
                            "A Subject Updated event is published with the "
                            "updated attributes."
                        ),
                        suggested_data_format=(
                            "Subject update event JSON with subject ID and "
                            "change summary."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Verify investigation case context reflects the "
                            "updated subject details."
                        ),
                        expected_result=(
                            "Subject fields in the case match the updated data."
                        ),
                        suggested_data_format=(
                            "Case context snapshot before and after update."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D1-03",
                title="Investigation transaction search via OLTP",
                purpose=(
                    "Confirm investigation search queries OLTP and returns "
                    "complete transaction details."
                ),
                trace_to_flow="Sensa Investigation -> OLTP Transaction Search",
                data_notes=[
                    "Use transactions ingested in the same test cycle."
                ],
                steps=[
                    TestStep(
                        description=(
                            "Execute a transaction search in Investigation "
                            "using known transaction IDs."
                        ),
                        expected_result=(
                            "Search results return matching transactions with "
                            "full details."
                        ),
                        suggested_data_format=(
                            "Search criteria: transaction ID, date range, "
                            "subject ID."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Compare the search output with OLTP records."
                        ),
                        expected_result=(
                            "Field-level values match the OLTP source data."
                        ),
                        suggested_data_format=(
                            "OLTP query output for the same transaction IDs."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D1-04",
                title="Case lifecycle updates and audit trail",
                purpose=(
                    "Validate case workflow actions are tracked and do not "
                    "break event traceability."
                ),
                trace_to_flow="Sensa Investigation workflow",
                data_notes=[
                    "Use a case created from a detection event in D1-01.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Update the case status through key workflow "
                            "states (e.g., New -> In Review -> Escalated)."
                        ),
                        expected_result=(
                            "Status changes are persisted and visible across "
                            "sessions."
                        ),
                        suggested_data_format=(
                            "Case update payload with status, user, timestamp."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Review the audit trail for status changes and "
                            "linked detection events."
                        ),
                        expected_result=(
                            "Audit trail shows all changes and references the "
                            "event ID."
                        ),
                        suggested_data_format=(
                            "Audit log export with case ID and timestamps."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D1-05",
                title="Operational Assistant guidance in investigations",
                purpose=(
                    "Ensure Operational Assistants provide workflow guidance "
                    "aligned to the investigation context."
                ),
                trace_to_flow="Sensa Investigation -> Sensa Agent",
                data_notes=[
                    "Use a case with multiple alerts and subject details.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Open a case and request Operational Assistant "
                            "guidance for a standard task."
                        ),
                        expected_result=(
                            "Assistant response references the case data and "
                            "suggests next actions."
                        ),
                        suggested_data_format=(
                            "Case summary with alert IDs and subject details."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Record the response and verify it includes the "
                            "correct transaction and subject references."
                        ),
                        expected_result=(
                            "Assistant output aligns with case evidence and "
                            "does not omit key alerts."
                        ),
                        suggested_data_format=(
                            "Assistant response transcript for evidence."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D1-06",
                title="Investigator Copilot analysis on case data",
                purpose=(
                    "Validate Investigator Copilot produces insights grounded "
                    "in the case context."
                ),
                trace_to_flow="Sensa Investigation -> Sensa Agent",
                data_notes=[
                    "Use a case with transaction history and risk indicators.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Request a case summary from Investigator Copilot."
                        ),
                        expected_result=(
                            "Summary reflects case facts and risk indicators "
                            "from the data."
                        ),
                        suggested_data_format=(
                            "Case dataset with transaction timeline."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Compare Copilot response to source case evidence."
                        ),
                        expected_result=(
                            "No conflicting or missing key facts are found."
                        ),
                        suggested_data_format=(
                            "Evidence checklist with expected highlights."
                        ),
                    ),
                ],
            ),
        ]

        # Day Two: full end-to-end integration and robustness coverage.
        day_two_cases = [
            TestCase(
                case_id="D2-01",
                title="End-to-end batch flow across all applications",
                purpose=(
                    "Validate full SRI AML GA integration from ingestion to "
                    "investigation and agent assistance."
                ),
                trace_to_flow="MinIO -> Sensa Data -> Detection -> Kafka -> "
                "Investigation -> Sensa Agent",
                data_notes=[
                    "Use a full SRI AML GA IFS batch with subjects, accounts, "
                    "and transactions.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Upload a full batch to MinIO and confirm Sensa "
                            "Data ingestion and Data Quality completion."
                        ),
                        expected_result=(
                            "Data Quality completes with a pass rate recorded."
                        ),
                        suggested_data_format=(
                            "CSV or Parquet batch with subject, account, "
                            "transaction entities."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Verify AML Inference and Decisioning publish "
                            "Detection Events to Kafka."
                        ),
                        expected_result=(
                            "Detection Events include enrichment fields and "
                            "correlation IDs."
                        ),
                        suggested_data_format=(
                            "Detection event JSON with enriched attributes."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Confirm Investigation cases are created and "
                            "Operational Assistants can respond."
                        ),
                        expected_result=(
                            "Cases exist with alerts, and assistant responses "
                            "match case context."
                        ),
                        suggested_data_format=(
                            "Case export with alert IDs and subject details."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D2-02",
                title="Model training uses OLAP history and updates inference",
                purpose=(
                    "Ensure Sensa Training consumes OLAP history and the "
                    "resulting model is used for inference."
                ),
                trace_to_flow="OLAP -> Sensa Training -> AML Inference",
                data_notes=[
                    "OLAP contains historic data for at least one period.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Initiate a training run for a historic date range."
                        ),
                        expected_result=(
                            "Training job reads OLAP partitions and completes "
                            "with a new model version."
                        ),
                        suggested_data_format=(
                            "OLAP partitions with date-based transaction data."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Run inference and confirm the model version used "
                            "matches the latest training output."
                        ),
                        expected_result=(
                            "Inference metadata references the updated model."
                        ),
                        suggested_data_format=(
                            "Model artifact metadata with version and hash."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D2-03",
                title="Data Quality failure prevents inference trigger",
                purpose=(
                    "Validate failure handling when data quality rules fail."
                ),
                trace_to_flow="Data Quality -> AML Inference trigger",
                data_notes=[
                    "Include invalid records that violate mandatory rules.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Upload a batch with missing mandatory fields."
                        ),
                        expected_result=(
                            "Data Quality marks the batch as failed."
                        ),
                        suggested_data_format=(
                            "Batch file with missing subject identifiers."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Confirm AML Inference is not triggered for the "
                            "failed batch."
                        ),
                        expected_result=(
                            "No inference job is created and errors are logged."
                        ),
                        suggested_data_format=(
                            "Data Quality failure report with error codes."
                        ),
                    ),
                ],
            ),
            TestCase(
                case_id="D2-04",
                title="Cross-system correlation and traceability",
                purpose=(
                    "Ensure a batch run ID is traceable across all systems."
                ),
                trace_to_flow="Batch ID across Data, Detection, Kafka, "
                "Investigation",
                data_notes=[
                    "Use a unique batch ID embedded in file and event metadata.",
                ],
                steps=[
                    TestStep(
                        description=(
                            "Execute a batch with a unique run ID and capture "
                            "Data Quality and inference metadata."
                        ),
                        expected_result=(
                            "Run ID is present in Data Quality, inference, and "
                            "decisioning metadata."
                        ),
                        suggested_data_format=(
                            "Batch filename and metadata containing run ID."
                        ),
                    ),
                    TestStep(
                        description=(
                            "Trace the run ID through Kafka events and "
                            "investigation cases."
                        ),
                        expected_result=(
                            "Detection Events and case records include the "
                            "same run ID."
                        ),
                        suggested_data_format=(
                            "Event payload with correlation ID field."
                        ),
                    ),
                ],
            ),
        ]

        # Order matters: sections are rendered in the report sequence.
        day_zero_section = PlanSection(
            section_id="day-zero",
            title="Day Zero: Sanity Integration Tests",
            description=[
                "Short sanity checks to confirm core batch ingestion and "
                "integration points are operational.",
                "These tests are designed to be fast and unblock Day 1 and "
                "Day 2 execution.",
            ],
            image_focus=ImageFocus(
                object_position="18% 46%",
                scale=1.6,
                height_px=280,
                caption="Architecture focus: ingestion and detection readiness.",
            ),
            strategy_sections=[],
            integration_flows=[],
            out_of_scope=[],
            assumptions=[],
            test_cases=day_zero_cases,
        )

        day_one_section = PlanSection(
            section_id="day-one",
            title="Day 1: Detection to Investigation and Agent Integration",
            description=[
                "Integration testing focused on Sensa Detection, Sensa "
                "Investigation, and Sensa Agent interactions.",
                "Validates Kafka event consumption, investigation workflows, "
                "and AI assistant engagement.",
            ],
            image_focus=ImageFocus(
                object_position="86% 46%",
                scale=1.6,
                height_px=280,
                caption="Architecture focus: Detection, Investigation, Agents.",
            ),
            strategy_sections=[],
            integration_flows=[],
            out_of_scope=[],
            assumptions=[],
            test_cases=day_one_cases,
        )

        day_two_section = PlanSection(
            section_id="day-two",
            title="Day 2: Full End-to-End Integration Tests",
            description=[
                "Comprehensive end-to-end integration coverage across all "
                "Sensa Risk Intelligence applications.",
                "Includes full batch processing, enrichment, investigation, "
                "and agent assistance with volume and error scenarios.",
            ],
            image_focus=ImageFocus(
                object_position="50% 50%",
                scale=1.25,
                height_px=300,
                caption="Architecture focus: full SRI AML GA integration.",
            ),
            strategy_sections=[],
            integration_flows=[],
            out_of_scope=[],
            assumptions=[],
            test_cases=day_two_cases,
        )

        return TestPlan(
            title="Sensa Risk Intelligence AML GA Integration Test Plan",
            author=AUTHOR_NAME,
            report_date=report_date,
            objective=(
                "Provide a structured, repeatable integration test strategy "
                "and execution steps for SRI AML GA data flows."
            ),
            sections=[
                high_level_section,
                day_zero_section,
                day_one_section,
                day_two_section,
            ],
        )
    except Exception as exc:
        LOGGER.exception("Failed to build test plan: %s", exc)
        raise PlanGenerationError("Failed to build test plan data.") from exc


def build_test_plan() -> TestPlan:
    """Create the test plan using the SRI AML GA architecture context.

    Returns:
        A populated TestPlan object with strategy and test cases.
    """
    try:
        # Always stamp the report with today's local date for traceability.
        report_date = datetime.now().strftime("%B %d, %Y")

        # Load images and phase text inputs for richer section content.
        section_images = _build_section_images()
        phase_text_map = _extract_phase_text(
            PROJECT_ROOT / I_PHASES_DOCX_FILENAME
        )
        if not any(phase_text_map.values()):
            LOGGER.warning(
                "No I1-I7 phase text detected in %s.",
                I_PHASES_DOCX_FILENAME,
            )

        strategy_sections = [
            (
                "Scope and Objectives",
                [
                    "Validate end-to-end batch data ingestion, AML inference, "
                    "and investigation workflows across Sensa Data, Sensa "
                    "Detection, Sensa Investigation, and Sensa Agent.",
                    "Confirm integration points between MinIO feeds, OLAP/OLTP "
                    "stores, Kafka events, and investigation search.",
                    "Demonstrate the AML GA release operates in batch mode "
                    "only and uses MinIO as the sole source feed.",
                ],
            ),
            (
                "Architecture Summary",
                [
                    "MinIO feeds are ingested by Sensa Data (Data Connectors) "
                    "and validated by Data Quality before landing in OLAP "
                    "(Iceberg) and OLTP (PostgreSQL).",
                    "Sensa Detection ingests transactions directly from MinIO "
                    "for AML Inference (batch) and uses OLAP data for training.",
                    "Decisioning Service enriches detection events with OLTP "
                    "data and publishes Detection Events to Kafka.",
                    "Sensa Investigation consumes Subject Updated and "
                    "Detection Events from Kafka and queries OLTP for search.",
                    "Sensa Agent capabilities (Operational Assistants and "
                    "Investigator Copilot) are accessed from Investigation.",
                ],
            ),
            (
                "Test Approach",
                [
                    "Exercise each integration flow with batch payloads "
                    "aligned to the SRI AML GA IFS schema.",
                    "Validate positive flows and error handling "
                    "(data quality failures, missing references).",
                    "Capture evidence from logs, Kafka topics, and UI/API "
                    "responses for traceability.",
                ],
            ),
            (
                "Environments and Test Data",
                [
                    "Use non-production MinIO buckets with controlled batch "
                    "payloads (transactions, subjects, accounts).",
                    "Seed OLTP reference data required for enrichment and "
                    "investigation search.",
                    "Use Kafka topics dedicated to test execution to isolate "
                    "events and ensure clean assertions.",
                ],
            ),
            (
                "Entry and Exit Criteria",
                [
                    "Entry: Batch ingestion endpoints available, Kafka topics "
                    "provisioned, and OLAP/OLTP accessible.",
                    "Exit: All integration test cases executed with expected "
                    "results and defects triaged.",
                ],
            ),
        ]

        integration_flows = [
            "MinIO -> Sensa Data (Data Connectors) -> Data Quality -> OLAP/OLTP",
            "MinIO (direct) -> AML Inference (batch) in Sensa Detection",
            "Data Quality -> AML Inference trigger",
            "OLAP (historic data) -> Sensa Training",
            "AML Inference -> Decisioning Service -> Kafka Detection Events",
            "Sensa Data -> Kafka Subject Updated Events -> Sensa Investigation",
            "Sensa Investigation -> OLTP Transaction Search",
            "Sensa Investigation -> Sensa Agent (Operational Assistants, "
            "Investigator Copilot)",
        ]

        out_of_scope = [
            "Real-time ingestion or streaming processing.",
            "Data ingestion from sources other than MinIO.",
            "Ontology-driven access for AML inference or investigation search.",
            "Entity Resolution, Watchlist Management, and Data Mapping Agent.",
        ]

        assumptions = [
            "SRI AML GA IFS schema is the agreed format for test data.",
            "Kafka topics and schemas are available for Subject Updated and "
            "Detection Events.",
            "OLTP contains required reference data for Decisioning enrichment.",
        ]

        # High-level strategy section drives the overall narrative.
        high_level_section = PlanSection(
            section_id="high-level",
            title="Integrated Test Plan Strategy",
            description=[
                "This section documents the integrated test strategy for the "
                "SRI AML GA release, aligned to the approved architecture.",
                "The focus is on batch-only processing, MinIO-based ingestion, "
                "and the core flows between Sensa Data, Detection, "
                "Investigation, and Agents.",
            ],
            section_image=section_images["high-level"],
            strategy_sections=strategy_sections,
            integration_flows=integration_flows,
            out_of_scope=out_of_scope,
            assumptions=assumptions,
            test_cases=[],
        )

        # Day Zero: I1 sanity validations.
        day_zero_cases = _build_day_cases("DZ", ["I1"], phase_text_map)
        day_zero_section = PlanSection(
            section_id="day-zero",
            title="Day Zero: Sanity Integration Tests (I1)",
            description=[
                "Short sanity checks to confirm core batch ingestion and "
                "investigation readiness for I1.",
                "These tests are designed to be fast and unblock Day 1 and "
                "Day 2 execution.",
            ],
            section_image=section_images["day-zero"],
            strategy_sections=[],
            integration_flows=[],
            out_of_scope=[],
            assumptions=[],
            test_cases=day_zero_cases,
        )

        # Day One: I2, I3, I4 integration validations.
        day_one_cases = _build_day_cases(
            "D1",
            ["I2", "I3", "I4"],
            phase_text_map,
        )
        day_one_section = PlanSection(
            section_id="day-one",
            title="Day 1: Integration Tests (I2-I4)",
            description=[
                "Integration testing focused on Sensa Detection, Sensa "
                "Investigation, and the key I2-I4 data flows.",
                "Validates Kafka event consumption, inference triggers, and "
                "investigation workflows.",
            ],
            section_image=section_images["day-one"],
            strategy_sections=[],
            integration_flows=[],
            out_of_scope=[],
            assumptions=[],
            test_cases=day_one_cases,
        )

        # Day Two: I5, I6, I7 end-to-end and agent coverage.
        day_two_cases = _build_day_cases(
            "D2",
            ["I5", "I6", "I7"],
            phase_text_map,
        )
        day_two_section = PlanSection(
            section_id="day-two",
            title="Day 2: End-to-End Integration Tests (I5-I7)",
            description=[
                "Comprehensive end-to-end integration coverage across all "
                "Sensa Risk Intelligence applications.",
                "Includes training, investigation search, and agent-assisted "
                "workflows.",
            ],
            section_image=section_images["day-two"],
            strategy_sections=[],
            integration_flows=[],
            out_of_scope=[],
            assumptions=[],
            test_cases=day_two_cases,
        )

        return TestPlan(
            title="Sensa Risk Intelligence AML GA Integration Test Plan",
            author=AUTHOR_NAME,
            report_date=report_date,
            objective=(
                "Provide a structured, repeatable integration test strategy "
                "and execution steps for SRI AML GA data flows."
            ),
            sections=[
                high_level_section,
                day_zero_section,
                day_one_section,
                day_two_section,
            ],
        )
    except Exception as exc:
        LOGGER.exception("Failed to build test plan: %s", exc)
        raise PlanGenerationError("Failed to build test plan data.") from exc


def _render_list(items: Iterable[str]) -> str:
    """Render a list of items as HTML list entries."""
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def _render_test_case(test_case: TestCase) -> str:
    """Render a single test case as HTML.

    Args:
        test_case: TestCase object to render.

    Returns:
        HTML string for the test case.
    """
    steps_html = []
    for step in test_case.steps:
        steps_html.append(
            "<li>"
            f"<div class=\"step-desc\">"
            f"{html.escape(step.description)}</div>"
            f"<div class=\"step-expected\"><strong>Expected:</strong> "
            f"{html.escape(step.expected_result)}</div>"
            f"<div class=\"step-data\"><strong>Suggested data "
            f"format:</strong> "
            f"{html.escape(step.suggested_data_format)}</div>"
            "</li>"
        )

    return (
        "<article class=\"test-case\">"
        f"<h3>{html.escape(test_case.case_id)}: "
        f"{html.escape(test_case.title)}</h3>"
        f"<p><strong>Purpose:</strong> "
        f"{html.escape(test_case.purpose)}</p>"
        f"<p><strong>Traceability:</strong> "
        f"{html.escape(test_case.trace_to_flow)}</p>"
        "<p><strong>Test data notes:</strong></p>"
        f"<ul>{_render_list(test_case.data_notes)}</ul>"
        "<ol class=\"steps\">"
        f"{''.join(steps_html)}"
        "</ol>"
        "</article>"
    )


def _render_section_figure(section: PlanSection) -> str:
    """Render the diagram figure for a section."""
    image = section.section_image
    hand_drawn_class = " hand-drawn" if image.hand_drawn else ""
    return (
        f"<figure class=\"section-figure{hand_drawn_class}\">"
        "<div class=\"image-frame\">"
        f"<img src=\"{image.data_uri}\" "
        f"alt=\"{html.escape(image.alt_text)}\" "
        "class=\"diagram-image\" />"
        "</div>"
        f"<figcaption>{html.escape(image.caption)}</figcaption>"
        "</figure>"
    )


def _render_section(section: PlanSection) -> str:
    """Render a full HTML section with optional strategy or test cases."""
    try:
        description_html = "".join(
            f"<p>{html.escape(paragraph)}</p>"
            for paragraph in section.description
        )
        strategy_html_parts = []
        for title, bullets in section.strategy_sections:
            strategy_html_parts.append(
                f"<h3>{html.escape(title)}</h3>"
                f"<ul>{_render_list(bullets)}</ul>"
            )

        integration_flows_html = ""
        if section.integration_flows:
            integration_flows_html = (
                "<h3>Integration Flows Under Test</h3>"
                f"<ol>{_render_list(section.integration_flows)}</ol>"
            )

        out_of_scope_html = ""
        if section.out_of_scope:
            out_of_scope_html = (
                "<h3>Out of Scope</h3>"
                f"<ul>{_render_list(section.out_of_scope)}</ul>"
            )

        assumptions_html = ""
        if section.assumptions:
            assumptions_html = (
                "<h3>Assumptions</h3>"
                f"<ul>{_render_list(section.assumptions)}</ul>"
            )

        test_cases_html = ""
        if section.test_cases:
            test_case_items = [_render_test_case(tc) for tc in section.test_cases]
            test_cases_html = "".join(test_case_items)

        return (
            f"<section id=\"{html.escape(section.section_id)}\">"
            f"<h2 class=\"section-title\">{html.escape(section.title)}</h2>"
            f"{_render_section_figure(section)}"
            f"{description_html}"
            f"{''.join(strategy_html_parts)}"
            f"{integration_flows_html}"
            f"{out_of_scope_html}"
            f"{assumptions_html}"
            f"{test_cases_html}"
            "</section>"
        )
    except Exception as exc:
        LOGGER.exception("Failed to render section %s: %s", section.title, exc)
        raise RenderError(
            f"Failed to render section: {section.title}"
        ) from exc


def render_html(
    plan: TestPlan,
    confluence_url: Optional[str] = None,
) -> str:
    """Render the full HTML report for the integration test plan.

    Args:
        plan: TestPlan data containing strategy and test cases.
        confluence_url: Optional Confluence report URL to reference.

    Returns:
        HTML string containing the rendered report.
    """
    try:
        # Render sections first to surface any template errors early.
        section_html = "".join(_render_section(section) for section in plan.sections)
    except RenderError:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to render HTML report: %s", exc)
        raise RenderError("Failed to render HTML report.") from exc

    confluence_link_html = ""
    if confluence_url:
        safe_url = html.escape(confluence_url, quote=True)
        confluence_link_html = (
            f" | <a href=\"{safe_url}\" target=\"_blank\" "
            "rel=\"noopener\">Confluence report</a>"
        )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(plan.title)}</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      color: #1f2933;
      margin: 32px;
      line-height: 1.5;
    }}
    h1 {{
      margin-bottom: 4px;
    }}
    .subtitle {{
      color: #52606d;
      margin-top: 0;
    }}
    section {{
      margin-top: 32px;
      padding: 24px;
      border: 3px solid #334e68;
      border-left-width: 12px;
      background: #f8fafc;
      box-shadow: 0 10px 18px rgba(0, 0, 0, 0.08);
      border-radius: 10px;
    }}
    section:first-of-type {{
      margin-top: 0;
    }}
    .section-title {{
      margin: 0 0 12px 0;
      color: #102a43;
    }}
    .section-figure {{
      margin: 0 0 24px 0;
      padding: 12px;
      border: 2px solid #d9e2ec;
      background: #ffffff;
    }}
    .section-figure.hand-drawn {{
      background: #fff7e6;
      border-style: dashed;
    }}
    .image-frame {{
      position: relative;
      overflow: hidden;
      border: 2px solid #cbd2d9;
      border-radius: 6px;
      background: #ffffff;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }}
    .section-figure.hand-drawn .image-frame {{
      border-style: dashed;
      box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12);
    }}
    .section-figure.hand-drawn .image-frame::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: repeating-linear-gradient(
        -45deg,
        rgba(10, 10, 10, 0.05),
        rgba(10, 10, 10, 0.05) 1px,
        rgba(255, 255, 255, 0.04) 1px,
        rgba(255, 255, 255, 0.04) 3px
      );
      mix-blend-mode: multiply;
      pointer-events: none;
    }}
    .image-frame::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: repeating-linear-gradient(
        45deg,
        rgba(20, 20, 20, 0.05),
        rgba(20, 20, 20, 0.05) 2px,
        rgba(255, 255, 255, 0.03) 2px,
        rgba(255, 255, 255, 0.03) 4px
      );
      mix-blend-mode: multiply;
      pointer-events: none;
    }}
    .diagram-image {{
      width: 100%;
      height: auto;
      display: block;
      filter: grayscale(0.1) contrast(1.2) brightness(1.05) saturate(0.9);
    }}
    .section-figure.hand-drawn {{
      transform: rotate(-0.4deg);
    }}
    .section-figure.hand-drawn .diagram-image {{
      filter: grayscale(0.55) contrast(1.7) brightness(1.2) saturate(0.6)
        sepia(0.15);
    }}
    .test-case {{
      border: 1px solid #d9e2ec;
      padding: 16px;
      margin: 16px 0;
      background: #ffffff;
    }}
    .steps {{
      margin-top: 12px;
    }}
    .step-desc {{
      font-weight: 600;
    }}
    .step-expected,
    .step-data {{
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(plan.title)}</h1>
  <p class="subtitle">
    Author: {html.escape(plan.author)} | Date: {html.escape(plan.report_date)}
    {confluence_link_html}
  </p>
  <p><strong>Objective:</strong> {html.escape(plan.objective)}</p>
  {section_html}
</body>
</html>
""".strip()


def write_report(html_content: str, output_path: Path) -> Path:
    """Write the HTML content to disk.

    Args:
        html_content: Rendered HTML report content.
        output_path: Target path for the HTML report.

    Returns:
        Path to the written report.

    Raises:
        ReportWriteError: If the report cannot be written.
    """
    try:
        output_path.write_text(html_content, encoding="utf-8")
        return output_path
    except OSError as exc:
        LOGGER.error("Failed to write report to %s: %s", output_path, exc)
        raise ReportWriteError(
            f"Failed to write HTML report: {output_path}"
        ) from exc


def main() -> int:
    """Run the plan generation workflow.

    Returns:
        Process exit code (0 for success, 1 for failure).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    LOGGER.info("Starting SRI integration test plan generation.")

    report_path = PROJECT_ROOT / REPORT_FILENAME

    try:
        plan = build_test_plan()
        confluence_url = CONFLUENCE_REPORT_PAGE_URL
        try:
            updated_url = publish_report_to_confluence(plan)
            if updated_url:
                confluence_url = updated_url
        except ConfluencePublishError as exc:
            LOGGER.warning(
                "Confluence publish failed, continuing without link: %s",
                exc,
            )
        except Exception as exc:
            LOGGER.warning(
                "Unexpected Confluence publish failure, continuing: %s",
                exc,
            )
        html_report = render_html(
            plan=plan,
            confluence_url=confluence_url,
        )
        write_report(html_report, report_path)
    except PlanGenerationError as exc:
        LOGGER.exception("Plan generation failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Unexpected error: %s", exc)
        return 1

    LOGGER.info("Report created at: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
