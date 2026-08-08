import gzip
import json
from pathlib import Path
from typing import Protocol

import boto3
import httpx

from filing_room.config import Settings


class ObjectStore(Protocol):
    async def put_text(self, key: str, value: str, content_type: str) -> None: ...
    async def put_json_gzip(self, key: str, value: object) -> None: ...


class LocalObjectStore:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        target = (self.root / key).resolve()
        if self.root not in target.parents:
            raise ValueError("Invalid object key")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    async def put_text(self, key: str, value: str, content_type: str) -> None:
        del content_type
        self._path(key).write_text(value, encoding="utf-8")

    async def put_json_gzip(self, key: str, value: object) -> None:
        with gzip.open(self._path(key), "wt", encoding="utf-8") as handle:
            json.dump(value, handle, separators=(",", ":"), default=str)


class R2ObjectStore:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.r2_bucket_name
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
        )

    async def put_text(self, key: str, value: str, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=value.encode(), ContentType=content_type
        )

    async def put_json_gzip(self, key: str, value: object) -> None:
        body = gzip.compress(json.dumps(value, separators=(",", ":"), default=str).encode())
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )


class CloudflareObjectStore:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=30)

    async def put_text(self, key: str, value: str, content_type: str) -> None:
        response = await self.client.put(
            f"http://filings.r2/{key}",
            content=value.encode(),
            headers={"content-type": content_type},
        )
        response.raise_for_status()

    async def put_json_gzip(self, key: str, value: object) -> None:
        body = gzip.compress(json.dumps(value, separators=(",", ":"), default=str).encode())
        response = await self.client.put(
            f"http://filings.r2/{key}",
            content=body,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )
        response.raise_for_status()


def create_object_store(settings: Settings) -> ObjectStore:
    if settings.object_store_backend == "cloudflare":
        return CloudflareObjectStore()
    return (
        R2ObjectStore(settings)
        if settings.object_store_backend == "r2"
        else LocalObjectStore(settings.local_data_dir)
    )
