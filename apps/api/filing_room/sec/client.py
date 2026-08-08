import asyncio
import time
from collections.abc import AsyncIterator

import httpx

from filing_room.config import Settings


class SecRateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            delay = self._interval - (time.monotonic() - self._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request = time.monotonic()


class SecClient:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._limiter = SecRateLimiter(settings.sec_requests_per_second)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            transport=transport,
            headers={
                "User-Agent": settings.sec_identity,
                "Accept-Encoding": "gzip, deflate",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_json(self, url: str) -> dict[str, object]:
        response = await self._request(url)
        return response.json()  # type: ignore[no-any-return]

    async def get_text(self, url: str) -> str:
        response = await self._request(url)
        return response.text

    async def _request(self, url: str) -> httpx.Response:
        if not url.startswith(
            ("https://data.sec.gov/", "https://www.sec.gov/Archives/", "https://www.sec.gov/files/")
        ):
            raise ValueError("Only known SEC data and archive URLs are allowed")
        await self._limiter.acquire()
        for attempt in range(4):
            response = await self._client.get(url)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
        response.raise_for_status()
        return response

    async def stream_text(self, url: str) -> AsyncIterator[bytes]:
        if not url.startswith("https://www.sec.gov/Archives/"):
            raise ValueError("Only SEC archive URLs are allowed")
        await self._limiter.acquire()
        async with self._client.stream("GET", url) as response:
            response.raise_for_status()
            async for block in response.aiter_bytes():
                yield block
