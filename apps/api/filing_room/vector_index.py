import httpx

from filing_room.models import Chunk


class VectorIndexer:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.client = httpx.AsyncClient(timeout=60)

    async def index(self, chunks: list[Chunk]) -> None:
        if self.backend != "cloudflare":
            return
        for start in range(0, len(chunks), 100):
            batch = chunks[start : start + 100]
            payload = {
                "chunks": [
                    {
                        "id": chunk.chunk_id,
                        "text": f"{chunk.section}\n{chunk.text}",
                        "metadata": {
                            "ticker": chunk.ticker,
                            "accession": chunk.accession,
                            "form": chunk.form,
                            "fiscal_year": chunk.fiscal_year,
                            "section": chunk.section,
                            "is_table": chunk.is_table,
                        },
                    }
                    for chunk in batch
                ]
            }
            response = await self.client.post("http://vectors.ai/index", json=payload)
            response.raise_for_status()
