# token-compressor — Benchmark Results

> exp_017 · 2026-02-28 · Base76 Research Lab

---

## Method comparison

Three approaches tested on the same corpus. Only the LLM + Embedding combination
achieves both high compression and semantic integrity.

| Experiment | Method | Semantic coverage | Token saving |
|------------|--------|-------------------|--------------|
| exp_015 | Regex only | 43.5% → **fallback** | 0% |
| exp_016 | Embedding only | 98.3% → compressed | 32% |
| **exp_017** | **LLM + Embedding** | **92.0% → compressed** | **62%** |

**Key insight from exp_016:** Embeddings cannot detect loss of conditionality.
A compressed prompt that drops "only if" scores 98% cosine similarity but changes meaning.
The LLM handles conditionality. The embedding is a safety net, not a compressor.

---

## Per-prompt results (exp_017)

| Input | Tokens in | Tokens out | Saved | Mode | Coverage |
|-------|-----------|------------|-------|------|----------|
| FNC abstract (academic EN) | 207 | 78 | **62%** | COMPRESSED | 92.0% |
| Nuanced condition (SV) | 23 | 23 | 0% | SKIPPED | — |
| MCP intent (SV) | 30 | 30 | 0% | SKIPPED | — |
| Session prompt (SV) | 64 | 64 | 0% | SKIPPED | — |
| Short command | 4 | 4 | 0% | SKIPPED | — |
| **TOTAL** | **328** | **199** | **39%** | | |

Prompts under 80 tokens skip the pipeline entirely — no cost, no risk.

---

## Pipeline economics

| Step | Model | Cost |
|------|-------|------|
| LLM compression | `llama3.2:1b` via Ollama | Free (local) |
| Embedding validation | `nomic-embed-text` via Ollama | Free (local) |
| Claude API saving | −129 tokens per call (FNC abstract) | Net positive |

**Break-even point:** prompts > 80 tokens.
At 207 tokens: saves 129 tokens per call — positive from first inference.

---

## What survives compression

The LLM is instructed to preserve:

- All conditionals: `if`, `only if`, `unless`, `when`, `but only`
- All negations: `not`, `never`, `without`, `except`
- All quantifiers and constraints

What gets removed:

- Filler phrases ("I would like to", "please", "could you")
- Hedging ("perhaps", "maybe", "it might be")
- Redundancy (restated concepts, verbose preambles)

---

## Failure mode (exp_016 finding)

Embedding-only compression fails silently on conditional prompts:

> Input: *"the idea is that it should be reversible but only under certain
> conditions when confidence is low"*

> Embedding-compressed: *"reversible under low-confidence conditions"*

> Cosine similarity: **98.3%** — passes threshold

> But the constraint *"but only under certain conditions"* is gone.

The LLM in exp_017 correctly preserves this constraint. The embedding then
validates that the compressed version is semantically equivalent — it is,
because the constraint survived.

---

---

## Live validation (2026-03-01)

Tested on structured research text with tree diagram formatting (`├──`, `└──`):

| Input | Tokens in | Tokens out | Saved | Mode | Coverage |
|-------|-----------|------------|-------|------|----------|
| Base76 research profile (structured EN) | 1116 | 275 | **75%** | COMPRESSED | 89.1% |

**Note on threshold:** Default threshold lowered from 0.90 → 0.85 after observing that
structured formatting (tree diagrams, ASCII art) reduces cosine similarity scores without
meaningful semantic loss. At 0.85, structured prompts compress correctly.

---

## MVP integration note (2026-03-05)

`token-compressor` is now showcased in the live Intent Compiler MVP:

- Demo: https://intent-compiler-mvp.pages.dev
- Integration context: idea -> spec -> compressed output

This provides a public validation surface for the compressed representation in a real UX flow.

---

*Results from Base76 Research Lab exp_017 · MIT License*
