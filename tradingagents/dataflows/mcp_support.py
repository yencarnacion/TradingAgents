from __future__ import annotations

from typing import Any, Dict, Optional

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import get_config


class MCPToolError(RuntimeError):
    """Raised when an MCP-backed dataflow tool fails."""


def _config_value(key: str, default: Any = None) -> Any:
    return get_config().get(key, default)


def market_data_mcp_url() -> Optional[str]:
    return _config_value("market_data_mcp_url")


def news_mcp_url() -> Optional[str]:
    return _config_value("news_mcp_url")


def fmp_mcp_url() -> Optional[str]:
    return _config_value("fmp_mcp_url")


def mcp_verify_tls() -> bool:
    return bool(_config_value("mcp_verify_tls", True))


async def _call_tool_async(url: str, tool_name: str, arguments: Dict[str, Any], verify: bool) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60, verify=verify) as client:
        async with streamable_http_client(url, http_client=client) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

    if getattr(result, "isError", False):
        raise MCPToolError(f"MCP tool '{tool_name}' returned an error")

    structured = getattr(result, "structuredContent", None) or {}
    if isinstance(structured, dict) and structured:
        return structured

    text_parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text:
            text_parts.append(text)
    if text_parts:
        return {"result": "\n".join(text_parts).strip()}

    return {}


def call_tool(url: str, tool_name: str, arguments: Dict[str, Any], *, verify: Optional[bool] = None) -> Dict[str, Any]:
    verify = mcp_verify_tls() if verify is None else verify
    return anyio.run(_call_tool_async, url, tool_name, arguments, verify)
