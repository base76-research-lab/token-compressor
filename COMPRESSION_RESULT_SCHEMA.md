# token-compressor — Schema Reference

> Version 1.0 · Base76 Research Lab · [base76.se](https://base76.se)

---

## CompressionResult object

Returned by `pipeline.process(text)` and included in MCP tool output.

```json
{
  "output_text": "Compress prompt. Preserve conditionals/negations. Remove filler.",
  "mode": "compressed",
  "coverage": 0.94,
  "confident": true,
  "tokens_in": 120,
  "tokens_out": 48,
  "tokens_saved": 72,
  "llm_output": "..."
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `output_text` | `string` | Text to send to your LLM (compressed or original) |
| `mode` | `string` | See modes below |
| `coverage` | `float 0–1` | Cosine similarity between original and compressed |
| `confident` | `bool` | `true` if coverage ≥ threshold (default 0.85) |
| `tokens_in` | `int` | Estimated input tokens (`len(text) // 4`) |
| `tokens_out` | `int` | Estimated output tokens |
| `tokens_saved` | `int` | Difference |
| `llm_output` | `string` | Raw LLM response before validation (useful for debugging fallbacks) |

### Modes

| `mode` | Meaning | When |
|--------|---------|------|
| `compressed` | Compressed text used | Coverage ≥ threshold |
| `raw_fallback` | Original text used | Coverage < threshold — meaning would be lost |
| `skipped` | Pipeline not run | Input < 80 tokens — not worth compressing |

---

## MCP tool schema

### Tool: `compress_prompt`

**Input:**
```json
{
  "text": "Your full prompt text here..."
}
```

**Output:**
```
Compressed prompt text here.

---
mode: compressed | coverage: 94.2% | tokens: 120→48 (-72)
```

The stats line is appended after `---` for easy parsing or stripping.

---

## Claude Code hook integration

Add to `~/.claude/settings.json` under `hooks → UserPromptSubmit`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo \"${CLAUDE_USER_PROMPT:-}\" | python3 /path/to/token-compressor/cli.py > /tmp/compressed_prompt.txt 2>/tmp/compress.log || true"
          }
        ]
      }
    ]
  }
}
```

The compressed text is written to `/tmp/compressed_prompt.txt`.
Stats are written to `/tmp/compress.log`.

**Reading the result in Claude Code:**

```python
# In a second hook or MCP server:
compressed = open("/tmp/compressed_prompt.txt").read().strip()
# Use compressed as the actual prompt
```

---

## MCP server integration

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

Once configured, call the tool directly in Claude Code:

```
compress_prompt("Your long prompt here...")
```

---

## Python SDK example

```python
from compressor import LLMCompressEmbedValidate

pipeline = LLMCompressEmbedValidate(
    threshold=0.85,       # lower = more aggressive compression
    min_tokens=80,        # skip pipeline below this
    compress_model="llama3.2:1b",
    embed_model="nomic-embed-text",
)

result = pipeline.process("Your prompt text...")

# Use result.output_text — always safe, falls back to original if validation fails
send_to_llm(result.output_text)

# Log stats
print(result.report())
# MODE:     COMPRESSED
# COVERAGE: 94.2% ✓
# TOKENS:   120 → 48 (-72, 60%)
```

---

## Design guarantees

1. **Never silently loses meaning** — if cosine similarity drops below threshold, original is returned unchanged
2. **Conditionals always preserved** — `if`, `only if`, `unless`, `when`, `but only` survive compression
3. **Negations always preserved** — `not`, `never`, `without` survive compression
4. **Short prompts skipped** — pipeline has overhead; prompts under 80 tokens are passed through directly
5. **Local only** — compression runs on local Ollama, no data leaves your machine

---

*Schema version 1.0 — Base76 Research Lab — MIT License*
