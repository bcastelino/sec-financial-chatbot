from datetime import date

from filing_room.models import CompanyRef, ResearchScope
from filing_room.retrieval import retrieve
from filing_room.sec.parser import FilingParser


def test_parser_keeps_section_and_table_boundaries() -> None:
    html = (
        "<html><head><title>Example 10-K</title></head><body><h2>Item 1A. Risk Factors</h2><p>Cybersecurity incidents could disrupt operations and harm results. "
        + "control systems exposure " * 40
        + "</p><table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2024</td><td>100</td></tr></table></body></html>"
    )
    parsed = FilingParser(target_words=40, overlap_words=5).parse(
        html,
        ticker="TEST",
        accession="0000000001-24-000001",
        form="10-K",
        filing_date=date(2024, 2, 1),
        fiscal_year=2023,
        sec_url="https://www.sec.gov/Archives/example.htm",
    )
    assert any(chunk.section == "Item 1A. Risk Factors" for chunk in parsed.chunks)
    assert any(chunk.is_table for chunk in parsed.chunks)


def test_retrieval_filters_scope_and_issues_server_source_ids() -> None:
    html = (
        "<h2>Item 1A. Risk Factors</h2><p>Cybersecurity incidents could disrupt operations. "
        + "security exposure " * 40
        + "</p>"
    )
    chunks = (
        FilingParser(target_words=40)
        .parse(
            html,
            ticker="TEST",
            accession="0000000001-24-000001",
            form="10-K",
            filing_date=date(2024, 2, 1),
            fiscal_year=2023,
            sec_url="https://www.sec.gov/Archives/example.htm",
        )
        .chunks
    )
    scope = ResearchScope(
        companies=[CompanyRef(ticker="TEST", cik="0000000001", name="Test Inc.")],
        forms=["10-K"],
        fiscalYears=[2023],
    )
    sources = retrieve("cybersecurity risk", scope, chunks)
    assert sources[0].id == "S1"
    assert sources[0].section == "Item 1A. Risk Factors"
