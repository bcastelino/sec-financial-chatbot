from dataclasses import dataclass
from datetime import date
from typing import Any

from filing_room.models import FinancialFact
from filing_room.sec.urls import filing_url


@dataclass(frozen=True)
class ConceptDefinition:
    label: str
    candidates: tuple[str, ...]
    preferred_units: tuple[str, ...]


CONCEPTS: dict[str, ConceptDefinition] = {
    "revenue": ConceptDefinition(
        "Revenue",
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"),
        ("USD",),
    ),
    "net_income": ConceptDefinition("Net income", ("NetIncomeLoss", "ProfitLoss"), ("USD",)),
    "operating_income": ConceptDefinition("Operating income", ("OperatingIncomeLoss",), ("USD",)),
    "assets": ConceptDefinition("Total assets", ("Assets",), ("USD",)),
    "liabilities": ConceptDefinition("Total liabilities", ("Liabilities",), ("USD",)),
    "cash": ConceptDefinition(
        "Cash and equivalents",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        ("USD",),
    ),
    "operating_cash_flow": ConceptDefinition(
        "Operating cash flow", ("NetCashProvidedByUsedInOperatingActivities",), ("USD",)
    ),
    "diluted_eps": ConceptDefinition(
        "Diluted EPS", ("EarningsPerShareDiluted",), ("USD/shares", "USD / shares")
    ),
    "shares_diluted": ConceptDefinition(
        "Diluted shares", ("WeightedAverageNumberOfDilutedSharesOutstanding",), ("shares",)
    ),
}


def _duration_days(unit: dict[str, Any]) -> int | None:
    start, end = unit.get("start"), unit.get("end")
    if not start or not end:
        return None
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _period_matches(unit: dict[str, Any], form: str) -> bool:
    duration = _duration_days(unit)
    if duration is None:
        return True
    return 250 <= duration <= 400 if form.startswith("10-K") else 65 <= duration <= 120


def select_financial_facts(payload: dict[str, Any], cik: str) -> list[FinancialFact]:
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    results: list[FinancialFact] = []
    seen: set[tuple[str, int, str]] = set()
    for registry_key, definition in CONCEPTS.items():
        for candidate in definition.candidates:
            concept = us_gaap.get(candidate)
            if not concept:
                continue
            for unit_name in definition.preferred_units:
                units = concept.get("units", {}).get(unit_name, [])
                for unit in sorted(
                    units,
                    key=lambda item: (item.get("filed", ""), item.get("accn", "")),
                    reverse=True,
                ):
                    form = unit.get("form", "")
                    fiscal_year = unit.get("fy")
                    accession = unit.get("accn")
                    if (
                        form not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}
                        or not fiscal_year
                        or not accession
                    ):
                        continue
                    base_form = form.replace("/A", "")
                    key = (registry_key, int(fiscal_year), unit.get("fp", base_form))
                    if key in seen or not _period_matches(unit, base_form):
                        continue
                    source = (
                        filing_url(cik, accession, unit.get("frame", "filing") + ".htm")
                        if unit.get("frame")
                        else f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
                    )
                    results.append(
                        FinancialFact(
                            concept=registry_key,
                            displayLabel=definition.label,
                            value=float(unit["val"]),
                            unit=unit_name,
                            periodStart=date.fromisoformat(unit["start"])
                            if unit.get("start")
                            else None,
                            periodEnd=date.fromisoformat(unit["end"]),
                            fiscalYear=int(fiscal_year),
                            fiscalQuarter=int(unit["fp"][1:])
                            if str(unit.get("fp", "")).startswith("Q")
                            else None,
                            form=form,
                            filedDate=date.fromisoformat(unit["filed"]),
                            accession=accession,
                            sourceUrl=source,
                        )
                    )
                    seen.add(key)
            break
    return sorted(
        results, key=lambda fact: (fact.fiscalYear, fact.fiscalQuarter or 5, fact.concept)
    )


def calculate_growth(current: FinancialFact, previous: FinancialFact) -> float | None:
    if current.concept != previous.concept or previous.value == 0:
        return None
    return (current.value - previous.value) / abs(previous.value)


def calculate_margin(numerator: FinancialFact, revenue: FinancialFact) -> float | None:
    if (
        revenue.concept != "revenue"
        or revenue.value == 0
        or numerator.periodEnd != revenue.periodEnd
    ):
        return None
    return numerator.value / revenue.value
