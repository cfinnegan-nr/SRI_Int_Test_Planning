"""Generate an HTML integration test plan for SRI AML GA.

This module reads the integration diagram image from the project root and
produces a self-contained HTML report with the test strategy and detailed
test cases for integrated data flows across Sensa Risk Intelligence.
"""

from __future__ import annotations

import base64
import html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DIAGRAM_FILENAME = "SRI_Integration_Diagram.png"
REPORT_FILENAME = "sri_int_test_plan.html"


class PlanGenerationError(RuntimeError):
    """Base exception for plan generation failures."""


class DiagramLoadError(PlanGenerationError):
    """Raised when the integration diagram cannot be loaded."""


class ReportWriteError(PlanGenerationError):
    """Raised when the HTML report cannot be written."""


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
class TestPlan:
    """High-level plan with strategy content and test cases."""

    title: str
    objective: str
    strategy_sections: List[tuple[str, List[str]]]
    integration_flows: List[str]
    out_of_scope: List[str]
    assumptions: List[str]
    test_cases: List[TestCase]


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
    strategy_sections = [
        (
            "Scope and Objectives",
            [
                "Validate end-to-end batch data ingestion, AML inference, "
                "and investigation workflows across Sensa Data, Sensa "
                "Detection, Sensa Investigation, and Sensa Agent.",
                "Confirm integration points between MinIO feeds, OLAP/OLTP "
                "stores, Kafka events, and investigation search.",
                "Demonstrate that the AML GA release operates in batch mode "
                "only and uses MinIO as the sole source feed.",
            ],
        ),
        (
            "Architecture Summary",
            [
                "MinIO feeds are ingested by Sensa Data (Data Connectors) and "
                "validated by Data Quality before landing in OLAP (Iceberg) "
                "and OLTP (PostgreSQL).",
                "Sensa Detection ingests transactions directly from MinIO for "
                "AML Inference (batch), and uses OLAP data for Sensa Training.",
                "Decisioning Service enriches detection events using OLTP "
                "data and publishes detection events to Kafka.",
                "Sensa Investigation consumes Subject Updated Events and "
                "Detection Events from Kafka and queries OLTP for searches.",
                "Sensa Agent capabilities (Operational Assistants and "
                "Investigator Copilot) are accessed from Sensa Investigation.",
            ],
        ),
        (
            "Test Approach",
            [
                "Exercise each integration flow with realistic batch payloads "
                "aligned to the SRI AML GA IFS schema.",
                "Validate both positive flows and critical error handling "
                "(e.g., data quality failures, missing references).",
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

    test_cases = [
        TestCase(
            case_id="TC-01",
            title="Batch ingestion from MinIO into Sensa Data",
            purpose=(
                "Validate that MinIO batch files are ingested, validated, "
                "and stored in OLAP/OLTP with data quality reporting."
            ),
            trace_to_flow="MinIO -> Sensa Data -> Data Quality -> OLAP/OLTP",
            data_notes=[
                "SRI AML GA IFS batch payload including subject, account, "
                "and transaction entities.",
                "Sample batch size with both valid and invalid records to "
                "exercise data quality rules.",
            ],
            steps=[
                TestStep(
                    description=(
                        "Upload a batch file to the MinIO ingestion bucket "
                        "using the agreed folder naming convention."
                    ),
                    expected_result=(
                        "Sensa Data picks up the batch and registers an "
                        "ingestion job with traceable run ID."
                    ),
                    suggested_data_format=(
                        "CSV or Parquet file aligned to SRI AML GA IFS schema "
                        "with required transaction and subject fields."
                    ),
                ),
                TestStep(
                    description=(
                        "Monitor Data Quality checks for the batch and record "
                        "validation outcomes."
                    ),
                    expected_result=(
                        "Validation results include pass/fail metrics, and "
                        "failed records are reported with reasons."
                    ),
                    suggested_data_format=(
                        "Include at least one record with a missing mandatory "
                        "field to verify quality rule reporting."
                    ),
                ),
                TestStep(
                    description=(
                        "Verify that validated data lands in OLAP and OLTP "
                        "stores with correct record counts."
                    ),
                    expected_result=(
                        "OLAP and OLTP contain the expected number of records "
                        "and key fields match the source payload."
                    ),
                    suggested_data_format=(
                        "Use deterministic IDs to reconcile source vs stored "
                        "records."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-02",
            title="Direct MinIO transaction feed to AML Inference",
            purpose=(
                "Confirm Sensa Detection ingests transaction data directly "
                "from MinIO and produces inference results."
            ),
            trace_to_flow="MinIO -> AML Inference (Batch)",
            data_notes=[
                "Transactions must match SRI AML GA IFS schema expected by "
                "Sensa Detection.",
            ],
            steps=[
                TestStep(
                    description=(
                        "Place a transaction batch file in the Sensa Detection "
                        "MinIO feed location."
                    ),
                    expected_result=(
                        "AML Inference job is triggered for the batch and "
                        "records are processed in batch mode."
                    ),
                    suggested_data_format=(
                        "Transactions file with timestamps, amounts, "
                        "counterparties, and product references."
                    ),
                ),
                TestStep(
                    description=(
                        "Collect inference outputs or decisioning inputs for "
                        "the processed batch."
                    ),
                    expected_result=(
                        "Inference outputs are produced for each transaction "
                        "record and are traceable to input IDs."
                    ),
                    suggested_data_format=(
                        "Expected output format: detection candidate records "
                        "with transaction IDs and risk scores."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-03",
            title="Data Quality triggers AML Inference",
            purpose=(
                "Ensure that completion of Data Quality processing triggers "
                "the AML Inference pipeline."
            ),
            trace_to_flow="Data Quality -> AML Inference trigger",
            data_notes=[
                "Use a batch that passes data quality checks to confirm "
                "trigger behavior.",
            ],
            steps=[
                TestStep(
                    description=(
                        "Complete a Data Quality run for a valid batch in "
                        "Sensa Data."
                    ),
                    expected_result=(
                        "A trigger event or job invocation is emitted to start "
                        "AML Inference."
                    ),
                    suggested_data_format=(
                        "Valid batch file with no missing mandatory fields."
                    ),
                ),
                TestStep(
                    description=(
                        "Verify that AML Inference job correlates to the "
                        "Data Quality run ID."
                    ),
                    expected_result=(
                        "Inference job metadata references the Data Quality "
                        "run ID and input batch identifier."
                    ),
                    suggested_data_format=(
                        "Use consistent batch identifiers across systems."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-04",
            title="Model training consumes historic OLAP data",
            purpose=(
                "Validate Sensa Training pulls historic data from OLAP for "
                "model training workflows."
            ),
            trace_to_flow="OLAP (Iceberg) -> Sensa Training",
            data_notes=[
                "OLAP must contain historic data for at least one period."
            ],
            steps=[
                TestStep(
                    description=(
                        "Initiate a Sensa Training run referencing a historic "
                        "time range."
                    ),
                    expected_result=(
                        "Training job reads the specified OLAP partitions and "
                        "records extracted data volume."
                    ),
                    suggested_data_format=(
                        "OLAP partitioned data with date-based partitions and "
                        "transaction summaries."
                    ),
                ),
                TestStep(
                    description=(
                        "Verify training output artifacts or completion status "
                        "for the job."
                    ),
                    expected_result=(
                        "Training job completes successfully and publishes "
                        "status or model artifacts as configured."
                    ),
                    suggested_data_format=(
                        "Model artifact metadata with version and timestamp."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-05",
            title="Decisioning enriches detection events from OLTP",
            purpose=(
                "Confirm Decisioning reads OLTP data to enrich detection "
                "events before publishing to Kafka."
            ),
            trace_to_flow="Decisioning Service -> OLTP -> Kafka Detection Events",
            data_notes=[
                "OLTP must include reference data to enrich detection events."
            ],
            steps=[
                TestStep(
                    description=(
                        "Generate inference outputs that require enrichment "
                        "(e.g., subject or account lookups)."
                    ),
                    expected_result=(
                        "Decisioning retrieves OLTP data and enriches the "
                        "detection event payload."
                    ),
                    suggested_data_format=(
                        "Reference tables with subject and account mappings."
                    ),
                ),
                TestStep(
                    description=(
                        "Validate detection events published to Kafka include "
                        "enriched fields."
                    ),
                    expected_result=(
                        "Kafka messages contain enriched attributes and the "
                        "original inference identifiers."
                    ),
                    suggested_data_format=(
                        "Detection event schema with enrichment fields such as "
                        "subject name, risk category, and account metadata."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-06",
            title="Subject Updated Events to Investigation",
            purpose=(
                "Validate that Sensa Data publishes subject updates to Kafka "
                "and Sensa Investigation consumes them."
            ),
            trace_to_flow="Sensa Data -> Kafka -> Sensa Investigation",
            data_notes=[
                "Subject updates must include identifiers used by "
                "investigation workflows."
            ],
            steps=[
                TestStep(
                    description=(
                        "Update a subject record in a batch ingestion and "
                        "monitor the Kafka Subject Updated Events topic."
                    ),
                    expected_result=(
                        "Kafka receives a Subject Updated event with the "
                        "subject identifier and change summary."
                    ),
                    suggested_data_format=(
                        "Subject update event JSON with subject ID, timestamp, "
                        "and update type."
                    ),
                ),
                TestStep(
                    description=(
                        "Verify Sensa Investigation ingests the subject update "
                        "and reflects it in investigation context."
                    ),
                    expected_result=(
                        "Investigation UI/API shows updated subject context "
                        "without manual refresh of data sources."
                    ),
                    suggested_data_format=(
                        "Investigation cache or index records updated from the "
                        "event payload."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-07",
            title="Detection events to Investigation",
            purpose=(
                "Verify that detection events published to Kafka are consumed "
                "by Sensa Investigation to create or update investigations."
            ),
            trace_to_flow="Decisioning Service -> Kafka -> Sensa Investigation",
            data_notes=[
                "Detection events should map to investigation case creation "
                "rules.",
            ],
            steps=[
                TestStep(
                    description=(
                        "Publish detection events from Decisioning and monitor "
                        "Kafka consumption."
                    ),
                    expected_result=(
                        "Sensa Investigation consumes detection events and "
                        "records event processing status."
                    ),
                    suggested_data_format=(
                        "Detection event JSON including transaction ID, "
                        "risk score, and alert type."
                    ),
                ),
                TestStep(
                    description=(
                        "Validate investigation records created or updated for "
                        "the detection events."
                    ),
                    expected_result=(
                        "Investigation cases are created or updated with "
                        "references to the detection event IDs."
                    ),
                    suggested_data_format=(
                        "Investigation record with event ID and workflow state."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-08",
            title="Investigation transaction search via OLTP",
            purpose=(
                "Ensure investigation transaction search queries OLTP and "
                "returns expected results."
            ),
            trace_to_flow="Sensa Investigation -> OLTP Transaction Search",
            data_notes=[
                "OLTP must contain the transaction records referenced in "
                "search queries.",
            ],
            steps=[
                TestStep(
                    description=(
                        "Perform a transaction search in Sensa Investigation "
                        "using known transaction identifiers."
                    ),
                    expected_result=(
                        "Search results return matching transactions with full "
                        "details."
                    ),
                    suggested_data_format=(
                        "Search criteria including transaction ID, date range, "
                        "and subject ID."
                    ),
                ),
                TestStep(
                    description=(
                        "Confirm search results align with OLTP records for "
                        "the same identifiers."
                    ),
                    expected_result=(
                        "Field-level data matches OLTP stored records and no "
                        "unexpected fields are missing."
                    ),
                    suggested_data_format=(
                        "OLTP query output for the same transaction IDs."
                    ),
                ),
            ],
        ),
        TestCase(
            case_id="TC-09",
            title="Sensa Agent integration for investigation support",
            purpose=(
                "Validate Operational Assistants and Investigator Copilot are "
                "available from investigation workflows."
            ),
            trace_to_flow="Sensa Investigation -> Sensa Agent",
            data_notes=[
                "Investigation cases should be available to drive assistant "
                "responses.",
            ],
            steps=[
                TestStep(
                    description=(
                        "Open an investigation case and invoke Operational "
                        "Assistants for a standard workflow task."
                    ),
                    expected_result=(
                        "Operational Assistants respond with task guidance or "
                        "automations tied to the investigation context."
                    ),
                    suggested_data_format=(
                        "Investigation case with detection event references "
                        "and subject context."
                    ),
                ),
                TestStep(
                    description=(
                        "Invoke Investigator Copilot for analytical guidance "
                        "and record the response."
                    ),
                    expected_result=(
                        "Investigator Copilot responds with contextual insight "
                        "aligned to the case data."
                    ),
                    suggested_data_format=(
                        "Case details including transaction history and risk "
                        "indicators."
                    ),
                ),
            ],
        ),
    ]

    return TestPlan(
        title="Sensa Risk Intelligence AML GA Integration Test Plan",
        objective=(
            "Provide a structured, repeatable integration test strategy and "
            "execution steps for SRI AML GA data flows."
        ),
        strategy_sections=strategy_sections,
        integration_flows=integration_flows,
        out_of_scope=out_of_scope,
        assumptions=assumptions,
        test_cases=test_cases,
    )


def _render_list(items: Iterable[str]) -> str:
    """Render a list of items as HTML list entries."""
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def render_html(plan: TestPlan, diagram_data_uri: str) -> str:
    """Render the full HTML report for the integration test plan.

    Args:
        plan: TestPlan data containing strategy and test cases.
        diagram_data_uri: Data URI for the integration diagram image.

    Returns:
        HTML string containing the rendered report.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    strategy_html_parts = []

    for title, bullets in plan.strategy_sections:
        strategy_html_parts.append(
            f"<h3>{html.escape(title)}</h3>"
            f"<ul>{_render_list(bullets)}</ul>"
        )

    integration_flows_html = (
        "<h3>Integration Flows Under Test</h3>"
        f"<ol>{_render_list(plan.integration_flows)}</ol>"
    )

    out_of_scope_html = (
        "<h3>Out of Scope</h3>"
        f"<ul>{_render_list(plan.out_of_scope)}</ul>"
    )

    assumptions_html = (
        "<h3>Assumptions</h3>"
        f"<ul>{_render_list(plan.assumptions)}</ul>"
    )

    test_cases_html = []
    for test_case in plan.test_cases:
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

        test_cases_html.append(
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
      margin-top: 24px;
    }}
    figure {{
      margin: 16px 0 24px 0;
      padding: 12px;
      border: 1px solid #d9e2ec;
      background: #f8fafc;
    }}
    figure img {{
      max-width: 100%;
      height: auto;
      display: block;
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
  <p class="subtitle">Generated: {html.escape(timestamp)}</p>
  <p><strong>Objective:</strong> {html.escape(plan.objective)}</p>

  <figure>
    <img src="{diagram_data_uri}" alt="SRI Integration Diagram" />
    <figcaption>Figure: SRI AML GA integration data flows.</figcaption>
  </figure>

  <section>
    <h2>Integrated Test Plan Strategy</h2>
    {''.join(strategy_html_parts)}
    {integration_flows_html}
    {out_of_scope_html}
    {assumptions_html}
  </section>

  <section>
    <h2>Test Cases and Execution Steps</h2>
    {''.join(test_cases_html)}
  </section>
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
        html_report = render_html(plan, diagram_data_uri)
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
