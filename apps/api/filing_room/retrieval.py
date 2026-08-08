import math
import re
from collections import Counter

from filing_room.models import Chunk, ResearchScope, SourceRef

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "with",
}


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def retrieve(
    query: str, scope: ResearchScope, corpus: list[Chunk], limit: int = 8
) -> list[SourceRef]:
    selected_tickers = {company.ticker.upper() for company in scope.companies}
    query_counts = Counter(tokens(query))
    scored: list[tuple[float, Chunk]] = []
    for chunk in corpus:
        if (
            chunk.ticker.upper() not in selected_tickers
            or chunk.form.replace("/A", "") not in scope.forms
        ):
            continue
        if scope.fiscalYears and chunk.fiscal_year not in scope.fiscalYears:
            continue
        if scope.accessions and chunk.accession not in scope.accessions:
            continue
        if scope.sections and chunk.section not in scope.sections:
            continue
        document_counts = Counter(tokens(f"{chunk.section} {chunk.text}"))
        lexical = sum(
            (1 + math.log(document_counts[token])) * count
            for token, count in query_counts.items()
            if document_counts[token]
        )
        phrase_bonus = 3.0 if query.lower() in chunk.text.lower() else 0.0
        table_bonus = (
            0.25
            if chunk.is_table
            and any(term in query_counts for term in {"revenue", "income", "cash", "margin"})
            else 0.0
        )
        scored.append((lexical + phrase_bonus + table_bonus, chunk))
    scored.sort(key=lambda item: (item[0], item[1].filing_date), reverse=True)
    supported = [item for item in scored if item[0] > 0][:limit]
    return [
        SourceRef(
            id=f"S{index}",
            ticker=chunk.ticker,
            accession=chunk.accession,
            form=chunk.form,
            filingDate=chunk.filing_date,
            section=chunk.section,
            excerpt=chunk.text[:1_200],
            r2Locator=f"parsed/{chunk.accession}/{chunk.chunk_id}.json.gz",
            secUrl=chunk.sec_url,
        )
        for index, (_, chunk) in enumerate(supported, start=1)
    ]
