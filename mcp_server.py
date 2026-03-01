#!/usr/bin/env python3
"""
token-compressor MCP server
Exposes compress_prompt as an MCP tool for Claude Code and other MCP clients.

Usage:
    python3 mcp_server.py

Claude Code config (~/.claude/settings.json):
    {
      "mcpServers": {
        "token-compressor": {
          "command": "python3",
          "args": ["/path/to/token-compressor/mcp_server.py"]
        }
      }
    }

Requires: ollama, numpy
Models:   ollama pull llama3.2:1b && ollama pull nomic-embed-text
"""

import json
import sys
from pathlib import Path

# Add compressor to path
sys.path.insert(0, str(Path(__file__).parent))
from compressor import LLMCompressEmbedValidate

# ---------------------------------------------------------------------------
# MCP protocol (stdio transport)
# ---------------------------------------------------------------------------

def send(obj: dict):
    print(json.dumps(obj), flush=True)


def handle(request: dict) -> dict | None:
    method = request.get("method")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "token-compressor", "version": "1.0.0"},
            }
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": [{
                "name": "compress_prompt",
                "description": (
                    "Compress a prompt to its semantic minimum using local LLM + "
                    "embedding validation. Reduces token usage 40-60% while preserving "
                    "all conditionals and negations. Returns original if validation fails."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The prompt text to compress"}
                    },
                    "required": ["text"],
                }
            }]}
        }

    if method == "tools/call":
        tool = request.get("params", {}).get("name")
        args = request.get("params", {}).get("arguments", {})

        if tool == "compress_prompt":
            text = args.get("text", "")
            try:
                pipeline = LLMCompressEmbedValidate()
                result = pipeline.process(text)
                content = (
                    f"{result.output_text}\n\n"
                    f"---\n"
                    f"mode: {result.mode} | "
                    f"coverage: {result.coverage*100:.1f}% | "
                    f"tokens: {result.tokens_in}→{result.tokens_out} "
                    f"(-{result.tokens_saved})"
                )
            except Exception as e:
                content = f"{text}\n\n[token-compressor error: {e} — returned original]"

            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content}]}
            }

    if method == "notifications/initialized":
        return None  # no response needed

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle(request)
            if response is not None:
                send(response)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            send({"jsonrpc": "2.0", "id": None,
                  "error": {"code": -32603, "message": str(e)}})


if __name__ == "__main__":
    main()
