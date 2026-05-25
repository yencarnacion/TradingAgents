from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "http://10.17.17.90:8084/mcp"
POLL_SECONDS = 30


def build_message(metadata: dict) -> str:
    ticker = metadata.get("ticker", "?")
    analysis_date = metadata.get("analysis_date", "?")
    status = metadata.get("status", "unknown")
    duration = metadata.get("duration_hms", "?")
    final_url = metadata.get("final_url")
    index_url = metadata.get("index_url")
    best_url = final_url or index_url or ""
    base = f"Trading Agents {ticker} {analysis_date} finished. status={status}. duration={duration}."
    if best_url:
        return f"{base} {best_url}"
    return base


async def send_sms(phone_number: str, message: str) -> None:
    async with streamablehttp_client(MCP_URL) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "send_general_sms",
                {"to": phone_number, "message": message},
            )
            print(result)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: notify_sms_when_done.py <metadata_json> <phone_number>", file=sys.stderr)
        return 2

    metadata_path = Path(sys.argv[1])
    phone_number = sys.argv[2]
    sentinel = metadata_path.with_name(".sms_notified")

    while True:
        if sentinel.exists():
            return 0
        if not metadata_path.exists():
            time.sleep(POLL_SECONDS)
            continue

        metadata = json.loads(metadata_path.read_text())
        status = metadata.get("status")
        if status == "running":
            time.sleep(POLL_SECONDS)
            continue

        if metadata.get("has_final_markdown"):
            metadata["final_url"] = metadata.get("index_url", "").rsplit("/", 1)[0] + "/final.html"
        message = build_message(metadata)
        anyio.run(send_sms, phone_number, message)
        sentinel.write_text(message + "\n", encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
