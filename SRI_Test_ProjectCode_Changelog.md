# SRI Test Project Code Changelog

## 2026-02-11
- Updated `sri_int_test_plan_generator.py` to produce four report sections
  (strategy, Day Zero, Day 1, Day 2) with detailed test cases and steps.
- Added per-section diagram focus with a hand-drawn CSS effect for report
  headers and included author/date metadata.
- Skipped AI usage reporting refactor because no reporting code exists in the
  module at this time.
- Removed performance and concurrency-focused test cases from the Day 2
  integration section.
- Enhanced the Day Zero/Day 1/Day 2 section diagrams with a stronger
  hand-drawn visual treatment.
- Added stronger section demarcation and increased diagram focus for
  Day Zero/Day 1/Day 2 sections via CSS and focus parameters.
- Strengthened section borders and overlaid section titles on diagrams.
- Increased hand-drawn styling and focus zoom for the Day Zero/Day 1/Day 2
  section images.
- Restored the High-Level section title by rendering an `<h2>` heading when
  the overlay title is not used.
- Added optional Confluence publishing using environment variables and
  created a Confluence-ready report body with the SRI AML GA architecture
  reference link.
- Linked the published Confluence report from the HTML output when available.
- Documented Confluence publishing configuration and behavior in `README.md`.
- Ensured Confluence publish failures log a warning and do not block HTML report
  generation.
- Added loading of Confluence environment variables from
  `C:\Sensa_NR\2025\QA\GenAI\AINative_Env\.env` when present.
- Fixed Confluence URL fallback to avoid duplicating `/wiki` when `_links.base`
  is missing in the API response.

## 2026-02-12
- Added DOCX parsing to extract I1-I7 phase descriptions using standard library
  ZIP/XML parsing and mapped the phases into Day Zero/Day 1/Day 2 test cases.
- Introduced per-section diagram assets including hand-drawn variants for the
  Day Zero/Day 1/Day 2 sections and stronger sub-application focus via cropping.
- Implemented section images and AI usage reporting in both HTML and Confluence
  outputs, with section titles positioned above images and stronger borders.
- Updated Confluence publishing to attach section images and update the
  existing page `1249018170` instead of creating a new page.
- Added Confluence update support with version lookup and attachment upload,
  and ensured the HTML report links to the Confluence report URL.
- Added a Confluence preflight access check to surface clearer permission
  errors before attempting page updates or attachment uploads.
- Removed the AI Usage Reporting section from HTML/Confluence outputs.
- Enhanced Confluence error messages to include the user email and base URL.
- Updated Confluence publish flow to delete the existing report page and
  recreate it before uploading attachments.
- Added Confluence validation that checks the report page URL matches the
  configured page ID before attempting delete/recreate operations.
- Added strict report URL matching against the expected constant and included
  the report URL in Confluence access check errors.
- Relaxed report URL validation to allow the environment override and only
  enforce page ID consistency between the URL and configured page ID.
