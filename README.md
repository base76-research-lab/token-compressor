# token-compressor

**Semantic prompt compression for LLM workflows. Reduce token usage by 40–60% without losing meaning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Requires: Ollama](https://img.shields.io/badge/requires-Ollama-111827)](https://ollama.com)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-8b5cf6)](https://modelcontextprotocol.io)

Built by [Base76 Research Lab](https://base76.se) — research into epistemic AI architecture.

---

## What it does

token-compressor is a two-stage pipeline that compresses prompts before they reach an LLM:

1. **LLM compression** — a local model (llama3.2:1b via Ollama) rewrites the prompt to its semantic minimum, preserving all conditionals and negations
2. **Embedding validation** — cosine similarity between original and compressed embeddings must exceed a threshold (default: 0.90) — if not, the original is sent unchanged

The result: shorter prompts, lower costs, same intent.

```
Input prompt (300 tokens)
        ↓
  LLM compresses
        ↓
  Embedding validates (cosine ≥ 0.90?)
        ↓
  Pass → compressed (120 tokens)   Fail → original (300 tokens)
```

**Key design principle:** conditionality is never sacrificed. If your prompt says "only do X if Y", that constraint survives compression.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- Two models pulled:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

- Python dependencies:

```bash
pip install ollama numpy
```

---

## Quick start

```python
from compressor import LLMCompressEmbedValidate

pipeline = LLMCompressEmbedValidate()
result = pipeline.process("Your prompt text here...")

print(result.output_text)   # compressed (or original if validation failed)
print(result.report())      # MODE / COVERAGE / TOKENS saved
```

**Result object:**

| Field | Description |
|-------|-------------|
| `output_text` | Text to send to your LLM |
| `mode` | `compressed` / `raw_fallback` / `skipped` |
| `coverage` | Cosine similarity (0.0–1.0) |
| `tokens_in` | Estimated input tokens |
| `tokens_out` | Estimated output tokens |
| `tokens_saved` | Difference |

---

## CLI usage

```bash
echo "Your long prompt here..." | python3 cli.py
```

Output: compressed text on stdout, stats on stderr.

---

## Claude Code hook (recommended setup)

Add to your `~/.claude/settings.json` under `hooks → UserPromptSubmit`:

```json
{
  "type": "command",
  "command": "echo \"${CLAUDE_USER_PROMPT:-}\" | python3 /path/to/token-compressor/cli.py > /tmp/compressed_prompt.txt 2>/tmp/compress.log || true"
}
```

This runs on every prompt submission and writes the compressed version to a temp file, which can be injected back into context via a second hook or MCP server.

---

## MCP server

The included MCP server exposes compression as a tool callable from any MCP-compatible client (Claude Code, etc.):

```bash
python3 mcp_server.py
```

**Tool:** `compress_prompt`
- Input: `text` (string)
- Output: compressed text + stats

**Claude Code MCP config** (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "token-compressor": {
      "command": "python3",
      "args": ["/path/to/token-compressor/mcp_server.py"]
    }
  }
}
```

---

## Configuration

```python
pipeline = LLMCompressEmbedValidate(
    threshold=0.90,          # cosine similarity floor (lower = more aggressive)
    min_tokens=80,           # skip pipeline below this (not worth compressing)
    compress_model="llama3.2:1b",
    embed_model="nomic-embed-text",
)
```

---

## How it works

**Stage 1 — LLM compression**

The compression prompt instructs the model to:
- Preserve all conditionals (`if`, `only if`, `unless`, `when`, `but only`)
- Preserve all negations
- Remove filler, hedging, redundancy
- Target 40–60% of original length

**Stage 2 — Embedding validation**

Computes cosine similarity between the original and compressed text using `nomic-embed-text`. If similarity falls below threshold, the original is returned unchanged. This prevents silent meaning loss.

---

## Results

Tested across Swedish and English prompts, technical and natural language:

| Input | Tokens in | Tokens out | Saved |
|-------|-----------|------------|-------|
| Research abstract (EN) | 89 | 38 | 57% |
| Session intent (SV) | 32 | 18 | 44% |
| Technical instruction | 47 | 22 | 53% |
| Short command (<80t) | — | — | skipped |

---

## Research background

This tool implements the architecture from:

> Wikström, B. (2026). *When Alignment Reduces Uncertainty: Epistemic Variance
> Collapse and Its Implications for Metacognitive AI.*
> DOI: [10.5281/zenodo.18731535](https://doi.org/10.5281/zenodo.18731535)

Part of the [Base76 Research Lab](https://base76.se) toolchain for epistemic AI infrastructure.

---

## License

MIT — Base76 Research Lab, Sweden
