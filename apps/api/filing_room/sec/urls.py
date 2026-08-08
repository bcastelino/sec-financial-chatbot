import re

ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
DOCUMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalize_cik(cik: str) -> str:
    digits = "".join(character for character in cik if character.isdigit())
    if not digits or len(digits) > 10:
        raise ValueError("Invalid CIK")
    return digits.zfill(10)


def filing_url(cik: str, accession: str, primary_document: str) -> str:
    if not ACCESSION_PATTERN.fullmatch(accession):
        raise ValueError("Invalid accession")
    if not DOCUMENT_PATTERN.fullmatch(primary_document):
        raise ValueError("Invalid primary document")
    cik_number = int(normalize_cik(cik))
    accession_compact = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession_compact}/{primary_document}"


def submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{normalize_cik(cik)}.json"


def company_facts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalize_cik(cik)}.json"
