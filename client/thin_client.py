from __future__ import annotations

import argparse
import base64
import math
import uuid
from pathlib import Path
from typing import Any

import requests

from app.common.json_codec import dumps_bytes, dumps_text, loads


def raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1000]
        raise requests.HTTPError(
            f"{exc}; response body: {body}", response=response
        ) from exc


def print_json(data: Any) -> None:
    print(dumps_text(data, pretty=True))


def response_json(response: requests.Response) -> Any:
    return loads(response.content)


def post_json(url: str, payload: dict[str, Any], timeout: int) -> requests.Response:
    return requests.post(
        url,
        data=dumps_bytes(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def upload(args: argparse.Namespace) -> None:
    path = Path(args.file)
    chunk_size = args.chunk_size_mb * 1024 * 1024
    metadata = load_metadata(args.metadata)
    if not args.no_chunk and path.stat().st_size > chunk_size:
        upload_chunked(args, path, chunk_size, metadata)
        return

    artifact = path.read_bytes()
    payload = {
        "artifact_base64": base64.b64encode(artifact).decode("ascii"),
        "activate": args.activate,
        "metadata": metadata,
    }
    url = f"{args.base_url.rstrip('/')}/v1/models/{args.model}/versions/{args.version}/upload"
    response = post_json(url, payload, timeout=120)
    raise_for_status(response)
    print_json(response_json(response))


def load_metadata(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return loads(Path(path).read_text(encoding="utf-8"))


def upload_chunked(
    args: argparse.Namespace,
    path: Path,
    chunk_size: int,
    metadata: dict[str, Any] | None,
) -> None:
    upload_id = args.upload_id or f"{args.model}-{args.version}-{uuid.uuid4().hex}"
    total_chunks = math.ceil(path.stat().st_size / chunk_size)
    url = f"{args.base_url.rstrip('/')}/v1/models/{args.model}/versions/{args.version}/upload-chunk"
    last_response: dict[str, Any] | None = None

    with path.open("rb") as fh:
        for chunk_index in range(total_chunks):
            chunk = fh.read(chunk_size)
            payload = {
                "upload_id": upload_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "artifact_base64": base64.b64encode(chunk).decode("ascii"),
                "metadata": metadata if chunk_index == total_chunks - 1 else None,
                "activate": args.activate and chunk_index == total_chunks - 1,
            }
            response = post_json(url, payload, timeout=120)
            raise_for_status(response)
            last_response = response_json(response)
            print(
                f"uploaded chunk {chunk_index + 1}/{total_chunks} "
                f"for {args.model}:{args.version} ({last_response.get('status')})"
            )

    if last_response is not None:
        print_json(last_response)


def activate(args: argparse.Namespace) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/models/{args.model}/versions/{args.version}/activate"
    response = requests.post(url, timeout=60)
    raise_for_status(response)
    print_json(response_json(response))


def deactivate(args: argparse.Namespace) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/models/{args.model}/deactivate"
    response = requests.post(url, timeout=60)
    raise_for_status(response)
    print_json(response_json(response))


def rollback(args: argparse.Namespace) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/models/{args.model}/rollback"
    response = requests.post(url, timeout=60)
    raise_for_status(response)
    print_json(response_json(response))


def models(args: argparse.Namespace) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/models"
    if args.model:
        url += f"/{args.model}"
    response = requests.get(url, timeout=60)
    raise_for_status(response)
    print_json(response_json(response))


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin client for local tensor serving")
    parser.add_argument("--base-url", default="http://localhost:8080")
    subparsers = parser.add_subparsers(required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--model", required=True)
    upload_parser.add_argument("--version", required=True)
    upload_parser.add_argument("--file", required=True)
    upload_parser.add_argument("--activate", action="store_true")
    upload_parser.add_argument("--chunk-size-mb", type=int, default=4)
    upload_parser.add_argument("--upload-id", default=None)
    upload_parser.add_argument("--no-chunk", action="store_true")
    upload_parser.add_argument("--metadata", default=None)
    upload_parser.set_defaults(func=upload)

    activate_parser = subparsers.add_parser("activate")
    activate_parser.add_argument("--model", required=True)
    activate_parser.add_argument("--version", required=True)
    activate_parser.set_defaults(func=activate)

    deactivate_parser = subparsers.add_parser("deactivate")
    deactivate_parser.add_argument("--model", required=True)
    deactivate_parser.set_defaults(func=deactivate)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--model", required=True)
    rollback_parser.set_defaults(func=rollback)

    models_parser = subparsers.add_parser("models")
    models_parser.add_argument("--model", default=None)
    models_parser.set_defaults(func=models)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
