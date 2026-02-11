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
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DIAGRAM_FILENAME = "SRI_Integration_Diagram.png"
REPORT_FILENAME = "sri_int_test_plan.html"
AUTHOR_NAME = "Ciaran Finnegan"
CONFLUENCE_REPORT_TITLE = "SRI AML GA Integration Test Plan Report"
CONFLUENCE_PARENT_PAGE_ID = "949420050"
CONFLUENCE_SPACE_KEY = "HARMONY"
CONFLUENCE_ENV_FILE = Path(
    r"C:\Sensa_NR\2025\QA\GenAI\AINative_Env\.env"
)
CONFLUENCE_ARCHITECTURE_PAGE_URL = (
    "https://netreveal.atlassian.net/wiki/spaces/HARMONY/pages/949420050/"
    "Sensa+Risk+Intelligence+-+SRI+AML+GA+Architecture"
)


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
    """Define how the architecture diagram is focused for a section."""

    object_position: str
    scale: float
    height_px: int
    caption: str


@dataclass(frozen=True)
class PlanSection:
    """Content for a single report section."""

    section_id: str
    title: str
    description: List[str]
    image_focus: ImageFocus
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

    return ConfluenceConfig(
        base_url=_normalize_base_url(base_url),
        user_email=user_email,
        api_token=api_token,
        space_key=space_key,
        parent_page_id=parent_page_id,
    )


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


def build_confluence_body(plan: TestPlan) -> str:
    """Build a Confluence storage-format body for the report."""
    sections_html = []
    for section in plan.sections:
        sections_html.append(f"<h2>{html.escape(section.title)}</h2>")
        if section.description:
            sections_html.append(_render_confluence_paragraphs(section.description))

        for title, bullets in section.strategy_sections:
            sections_html.append(f"<h3>{html.escape(title)}</h3>")
            sections_html.append(f"<ul>{_render_list(bullets)}</ul>")

        if section.integration_flows:
            sections_html.append("<h3>Integration Flows Under Test</h3>")
            sections_html.append(
                f"<ol>{_render_list(section.integration_flows)}</ol>"
            )

        if section.out_of_scope:
            sections_html.append("<h3>Out of Scope</h3>")
            sections_html.append(f"<ul>{_render_list(section.out_of_scope)}</ul>")

        if section.assumptions:
            sections_html.append("<h3>Assumptions</h3>")
            sections_html.append(f"<ul>{_render_list(section.assumptions)}</ul>")

        if section.test_cases:
            sections_html.append("<h3>Test Cases</h3>")
            sections_html.append(
                "".join(
                    _render_confluence_test_case(test_case)
                    for test_case in section.test_cases
                )
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
) -> str:
    """Create a Confluence page and return its URL.

    Args:
        config: Confluence configuration settings.
        title: Title of the new Confluence page.
        body: Storage-format body for the page.

    Returns:
        URL of the created Confluence page.

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
            f"Failed to publish Confluence page (HTTP {exc.code})."
        ) from exc
    except urllib.error.URLError as exc:
        LOGGER.error("Confluence publish request failed: %s", exc)
        raise ConfluencePublishError(
            "Failed to connect to Confluence for publishing."
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
            "Unexpected Confluence response while creating page: %s",
            response_body,
        )
        raise ConfluencePublishError(
            "Failed to parse Confluence publish response."
        ) from exc


def publish_report_to_confluence(plan: TestPlan) -> Optional[str]:
    """Publish the generated report to Confluence when configured."""
    config = load_confluence_config()
    if config is None:
        return None

    body = build_confluence_body(plan)
    return create_confluence_page(
        config=config,
        title=CONFLUENCE_REPORT_TITLE,
        body=body,
    )


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


def build_test_plan() -> TestPlan:
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


def _render_section_figure(section: PlanSection, diagram_data_uri: str) -> str:
    """Render the diagram figure for a section with focus styling."""
    focus = section.image_focus
    style = (
        f"--focus-position: {html.escape(focus.object_position)};"
        f"--focus-scale: {focus.scale};"
        f"--focus-height: {focus.height_px}px;"
    )
    alt_text = f"SRI integration focus: {section.title}"
    hand_drawn_class = ""
    if section.section_id != "high-level":
        hand_drawn_class = " hand-drawn"
    return (
        f"<figure class=\"section-figure{hand_drawn_class}\" style=\"{style}\">"
        "<div class=\"image-frame\">"
        "<div class=\"section-title-overlay\">"
        f"<h2>{html.escape(section.title)}</h2>"
        "</div>"
        f"<img src=\"{diagram_data_uri}\" "
        f"alt=\"{html.escape(alt_text)}\" "
        "class=\"diagram-image\" />"
        "</div>"
        f"<figcaption>{html.escape(focus.caption)}</figcaption>"
        "</figure>"
    )


def _render_section(section: PlanSection, diagram_data_uri: str) -> str:
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

        section_heading_html = ""
        if section.section_id == "high-level":
            section_heading_html = f"<h2>{html.escape(section.title)}</h2>"

        return (
            f"<section id=\"{html.escape(section.section_id)}\">"
            f"{_render_section_figure(section, diagram_data_uri)}"
            f"{section_heading_html}"
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
    diagram_data_uri: str,
    confluence_url: Optional[str] = None,
) -> str:
    """Render the full HTML report for the integration test plan.

    Args:
        plan: TestPlan data containing strategy and test cases.
        diagram_data_uri: Data URI for the integration diagram image.
        confluence_url: Optional Confluence report URL to reference.

    Returns:
        HTML string containing the rendered report.
    """
    try:
        # Render sections first to surface any template errors early.
        section_html = "".join(
            _render_section(section, diagram_data_uri)
            for section in plan.sections
        )
    except RenderError:
        raise
    except Exception as exc:
        LOGGER.exception("Failed to render HTML report: %s", exc)
        raise RenderError("Failed to render HTML report.") from exc

    confluence_link_html = " | Confluence report not published"
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
      border-left-width: 10px;
      background: #f8fafc;
      box-shadow: 0 10px 18px rgba(0, 0, 0, 0.08);
      border-radius: 10px;
    }}
    section:first-of-type {{
      margin-top: 0;
    }}
    .section-figure {{
      margin: 0 0 24px 0;
      padding: 12px;
      border: 1px solid #d9e2ec;
      background: #ffffff;
    }}
    .section-figure.hand-drawn {{
      background: #fff7e6;
      border-style: dashed;
    }}
    .image-frame {{
      position: relative;
      overflow: hidden;
      height: var(--focus-height, 260px);
      border: 2px solid #cbd2d9;
      border-radius: 6px;
      background: #ffffff;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }}
    .section-figure.hand-drawn .image-frame {{
      border-style: dashed;
      box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12);
    }}
    .section-title-overlay {{
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(13, 27, 42, 0.85);
      color: #ffffff;
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.25);
      z-index: 2;
      max-width: 75%;
    }}
    .section-title-overlay h2 {{
      margin: 0;
      font-size: 1.25rem;
      letter-spacing: 0.3px;
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
      height: 100%;
      object-fit: cover;
      object-position: var(--focus-position, 50% 50%);
      transform: scale(var(--focus-scale, 1));
      filter: grayscale(0.1) contrast(1.2) brightness(1.05) saturate(0.9);
    }}
    .section-figure.hand-drawn .diagram-image {{
      filter: grayscale(0.45) contrast(1.6) brightness(1.15) saturate(0.6)
        sepia(0.12);
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

    diagram_path = PROJECT_ROOT / DIAGRAM_FILENAME
    report_path = PROJECT_ROOT / REPORT_FILENAME

    try:
        diagram_data_uri = load_diagram_data_uri(diagram_path)
        plan = build_test_plan()
        confluence_url = None
        try:
            confluence_url = publish_report_to_confluence(plan)
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
            diagram_data_uri=diagram_data_uri,
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
