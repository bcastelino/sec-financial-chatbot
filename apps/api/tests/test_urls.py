import pytest

from filing_room.sec.urls import company_facts_url, filing_url, normalize_cik


def test_normalizes_cik() -> None:
    assert normalize_cik("320193") == "0000320193"
    assert company_facts_url("320193").endswith("CIK0000320193.json")


def test_builds_known_sec_archive_url() -> None:
    url = filing_url("320193", "0000320193-24-000123", "aapl-20240928.htm")
    assert (
        url == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )


@pytest.mark.parametrize(
    "accession,document", [("bad", "a.htm"), ("0000320193-24-000123", "../secret")]
)
def test_rejects_arbitrary_paths(accession: str, document: str) -> None:
    with pytest.raises(ValueError):
        filing_url("320193", accession, document)
