# SRI Integration Test Plan Generator

This application generates a comprehensive HTML integration test plan for Sensa Risk Intelligence (SRI) Anti-Money Laundering (AML) General Availability (GA) release. The generator creates a self-contained HTML report that includes test strategy, detailed test cases, and an embedded integration architecture diagram.

## Confluence Page Summary

The architectural diagram referenced in this application (originally documented in Confluence) illustrates the SRI Platform's Anti-Money Laundering (AML) Minimum Viable Product architecture, detailing how customer data flows from MinIO feeds and direct transaction sources through Sensa Data's ingestion and quality processes into OLAP (Iceberg) and OLTP (PostgreSQL) stores. The diagram shows how processed data feeds into Sensa Detection's AML inference capabilities, which then trigger investigation workflows through Kafka event streams, ultimately enabling Sensa Investigation and Sensa Agent tools for operational assistants and investigator copilot functionality.

## How the Application Extracts Required Data

### Data Extraction Process

The application extracts and processes data through the following mechanisms:

1. **Integration Diagram Loading**: The application reads the `SRI_Integration_Diagram.png` file from the project root directory and converts it to a base64-encoded data URI. This allows the diagram to be embedded directly in the generated HTML report without requiring external file dependencies.

2. **Test Plan Data Structure**: The application uses a structured data model (`TestPlan`, `TestCase`, `TestStep`) to organize test strategy information, integration flows, and detailed test execution steps. All test plan content is programmatically defined within the `build_test_plan()` function, which includes:
   - Test strategy sections (Scope, Architecture Summary, Test Approach, Environments, Entry/Exit Criteria)
   - Integration flows under test (8 distinct data flow paths)
   - Out-of-scope items and assumptions
   - Nine detailed test cases (TC-01 through TC-09) with step-by-step execution instructions

3. **HTML Report Generation**: The application renders all extracted data into a self-contained HTML document using the `render_html()` function. The HTML includes:
   - Embedded diagram (via data URI)
   - Formatted test strategy sections
   - Detailed test cases with purpose, traceability, data notes, and execution steps
   - Professional styling for readability

### Key Data Components Extracted

- **Architecture Information**: Integration flows between MinIO, Sensa Data, Data Quality, OLAP/OLTP stores, Kafka, Sensa Detection, Sensa Investigation, and Sensa Agent
- **Test Strategy**: Scope, objectives, approach, environment requirements, and entry/exit criteria
- **Test Cases**: Nine comprehensive test cases covering:
  - Batch ingestion from MinIO
  - Direct MinIO transaction feed to AML Inference
  - Data Quality triggers
  - Model training from OLAP data
  - Decisioning enrichment
  - Subject Updated Events
  - Detection Events
  - Investigation transaction search
  - Sensa Agent integration

### Output

The application generates `sri_int_test_plan.html` in the project root directory, which is a complete, self-contained HTML document that can be opened in any web browser or shared as a standalone file.

## Usage

### Prerequisites

- Python 3.11 or higher
- The `SRI_Integration_Diagram.png` file must be present in the project root directory

### Running the Application

```bash
python sri_int_test_plan_generator.py
```

The application will:
1. Load the integration diagram from `SRI_Integration_Diagram.png`
2. Build the test plan data structure
3. Generate the HTML report
4. Write the output to `sri_int_test_plan.html`

### Output

Upon successful execution, the application creates `sri_int_test_plan.html` containing:
- Complete test strategy documentation
- Embedded integration architecture diagram
- Nine detailed test cases with execution steps
- Timestamp indicating when the report was generated

## Project Structure

```
.
├── README.md                          # This file
├── sri_int_test_plan_generator.py     # Main application script
├── sri_int_test_plan.html            # Generated HTML report (output)
└── SRI_Integration_Diagram.png        # Integration architecture diagram (input)
```

## Error Handling

The application includes comprehensive error handling:
- `DiagramLoadError`: Raised if the integration diagram cannot be found or read
- `ReportWriteError`: Raised if the HTML report cannot be written to disk
- All errors are logged with appropriate context before being raised

## Code Quality

This application follows Python development best practices:
- PEP 8 compliant code style
- Type hints for all function parameters and return values
- Comprehensive docstrings using Google-style format
- Proper exception handling with custom exception classes
- Logging for debugging and monitoring

## License

This project is proprietary and intended for internal use within Sensa Risk Intelligence testing workflows.
