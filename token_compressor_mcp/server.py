#!/usr/bin/env python3
"""
token-compressor MCP server

Exposes compress_prompt as an MCP tool for Claude Code and other MCP clients.
Compresses prompts 40-60% using local LLM + embedding validation.

Requirements:
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
"""

import asyncio
import sys
from pathlib import Path

# Allow running from source without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolResult

server = Server("token-compressor")

COMPRESS_TOOL = Tool(
    name="compress_prompt",
    description=(
        "Compress a prompt to its semantic minimum using a local LLM and "
        "embedding validation. Reduces token usage 40-60% while preserving "
        "all conditionals, negations, and critical intent. "
        "Returns the original text unchanged if validation fails. "
        "Requires Ollama running locally with llama3.2:1b and nomic-embed-text."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The prompt text to compress",
            }
        },
        "required": ["text"],
    },
)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [COMPRESS_TOOL]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name != "compress_prompt":
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            is_error=True,
        )

    text = arguments.get("text", "")
    try:
        from compressor import LLMCompressEmbedValidate
        pipeline = LLMCompressEmbedValidate()
        result = pipeline.process(text)
        output = (
            f"{result.output_text}\n\n"
            f"---\n"
            f"mode: {result.mode} | "
            f"coverage: {result.coverage * 100:.1f}% | "
            f"tokens: {result.tokens_in}→{result.tokens_out} "
            f"(-{result.tokens_saved})"
        )
        return CallToolResult(
            content=[TextContent(type="text", text=output)],
            is_error=False,
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"{text}\n\n[token-compressor error: {e} — returned original]",
            )],
            is_error=False,  # graceful fallback, not a hard error
        )


async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
