from filing_room.sec.facts import calculate_growth, select_financial_facts


def fact_unit(
    *, value: float, year: int, accession: str, form: str, filed: str
) -> dict[str, object]:
    return {
        "start": f"{year}-01-01",
        "end": f"{year}-12-31",
        "val": value,
        "accn": accession,
        "fy": year,
        "fp": "FY",
        "form": form,
        "filed": filed,
    }


def payload_with(units: list[dict[str, object]]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": units},
                }
            }
        }
    }


def test_selects_latest_valid_annual_fact() -> None:
    payload = payload_with(
        [
            fact_unit(
                value=100,
                year=2023,
                accession="0000000001-24-000001",
                form="10-K",
                filed="2024-02-01",
            ),
            fact_unit(
                value=110,
                year=2023,
                accession="0000000001-24-000002",
                form="10-K/A",
                filed="2024-03-01",
            ),
        ]
    )
    facts = select_financial_facts(payload, "1")
    assert len(facts) == 1
    assert facts[0].value == 110
    assert facts[0].form == "10-K/A"


def test_growth_is_deterministic() -> None:
    payload = payload_with(
        [
            fact_unit(
                value=80,
                year=2022,
                accession="0000000001-23-000001",
                form="10-K",
                filed="2023-02-01",
            ),
            fact_unit(
                value=100,
                year=2023,
                accession="0000000001-24-000001",
                form="10-K",
                filed="2024-02-01",
            ),
        ]
    )
    facts = select_financial_facts(payload, "1")
    assert calculate_growth(facts[1], facts[0]) == 0.25
