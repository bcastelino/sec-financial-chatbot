from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date

from bs4 import BeautifulSoup, Tag

from filing_room.models import Chunk


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    chunks: list[Chunk]


class FilingParser:
    """Produce traceable chunks with a small, deterministic HTML walker.

    EdgarTools 5 owns SEC identity, rate, cache, and XBRL integration concerns. The
    walker is deliberately network-free so golden fixtures remain reproducible.
    """

    def __init__(self, target_words: int = 400, overlap_words: int = 50) -> None:
        self.target_words = target_words
        self.overlap_words = overlap_words

    def parse(
        self,
        html: str,
        *,
        ticker: str,
        accession: str,
        form: str,
        filing_date: date,
        fiscal_year: int,
        sec_url: str,
    ) -> ParsedDocument:
        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else f"{ticker} {form}"
        blocks: list[tuple[str, str, bool]] = []
        current_section = "Filing overview"
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "div", "table"]):
            if (
                not isinstance(element, Tag)
                or element.find_parent("table")
                and element.name != "table"
            ):
                continue
            text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if len(text) < 2:
                continue
            if element.name in {"h1", "h2", "h3", "h4"} or re.match(
                r"(?i)^item\s+\d+[a-z]?\.?\s", text
            ):
                current_section = text[:180]
                continue
            blocks.append((current_section, text, element.name == "table"))
        chunks: list[Chunk] = []
        narrative: dict[str, list[str]] = {}
        for section, text, is_table in blocks:
            if is_table:
                chunks.append(
                    self._chunk(
                        text[:12_000],
                        section,
                        len(chunks),
                        True,
                        ticker,
                        accession,
                        form,
                        filing_date,
                        fiscal_year,
                        sec_url,
                    )
                )
            else:
                narrative.setdefault(section, []).extend(text.split())
        stride = max(1, self.target_words - self.overlap_words)
        for section, words in narrative.items():
            for start in range(0, len(words), stride):
                window = words[start : start + self.target_words]
                if len(window) < 35 and start > 0:
                    break
                chunks.append(
                    self._chunk(
                        " ".join(window),
                        section,
                        len(chunks),
                        False,
                        ticker,
                        accession,
                        form,
                        filing_date,
                        fiscal_year,
                        sec_url,
                        start,
                    )
                )
        return ParsedDocument(title=title, chunks=chunks)

    def _chunk(
        self,
        text: str,
        section: str,
        index: int,
        is_table: bool,
        ticker: str,
        accession: str,
        form: str,
        filing_date: date,
        fiscal_year: int,
        sec_url: str,
        start: int = 0,
    ) -> Chunk:
        digest = hashlib.sha256(f"{accession}:{section}:{index}:{text[:80]}".encode()).hexdigest()[
            :16
        ]
        return Chunk(
            chunk_id=digest,
            ticker=ticker,
            accession=accession,
            form=form,
            filing_date=filing_date,
            fiscal_year=fiscal_year,
            section=section,
            text=text,
            token_start=start,
            token_end=start + len(text.split()),
            sec_url=sec_url,
            is_table=is_table,
        )


def initialize_edgartools(identity: str, requests_per_second: float = 5.0) -> None:
    cache_dir = os.path.join(tempfile.gettempdir(), "filing-room-edgar")
    os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", cache_dir)
    os.environ.setdefault("EDGAR_RATE_LIMIT_PER_SEC", str(max(1, int(requests_per_second))))
    from edgar import set_identity

    set_identity(identity)
