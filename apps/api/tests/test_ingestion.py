from filing_room.ingestion import IngestionService
from filing_room.models import CompanyRef


def test_submission_filter_includes_amendments_and_scope() -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000000001-24-000001",
                    "0000000001-24-000002",
                    "0000000001-21-000001",
                ],
                "form": ["10-K", "10-Q/A", "10-K"],
                "filingDate": ["2024-02-01", "2024-05-01", "2021-02-01"],
                "reportDate": ["2023-12-31", "2024-03-31", "2020-12-31"],
                "primaryDocument": ["annual.htm", "quarter.htm", "old.htm"],
            }
        }
    }
    company = CompanyRef(ticker="TEST", cik="0000000001", name="Test Inc.")
    filings = IngestionService._filings_from_submissions(
        company, payload, {"10-K", "10-Q"}, {2023, 2024}
    )
    assert [filing.form for filing in filings] == ["10-K", "10-Q/A"]
    assert filings[1].isAmendment is True
