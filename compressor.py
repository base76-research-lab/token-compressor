"""
exp_017 — LLM Compression + Embedding Validation
Base76 Research Lab — 2026-02-28

Architecture:
  1. If prompt < MIN_TOKENS: skip pipeline, return raw (not worth compressing)
  2. LLM (local Ollama) compresses, preserving conditionality
  3. Embedding validates: cosine_similarity(original, compressed) >= THRESHOLD
  4. Pass → send compressed. Fail → send raw

Key insight from exp_016:
  Embeddings cannot detect loss of conditionality.
  LLM handles that. Embedding is a cheap safety net, not a compressor.

Requires:
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import ollama

THRESHOLD: float = 0.85
MIN_TOKENS: int = 80          # below this: skip pipeline entirely
COMPRESS_MODEL: str = "llama3.2:1b"
EMBED_MODEL: str = "nomic-embed-text"

COMPRESS_PROMPT = """\
Compress the following text to its semantic minimum.
Rules:
- Preserve all conditionals (if, only if, unless, when, but only)
- Preserve all negations
- Remove filler, hedging, redundancy
- Output ONE compressed version, no explanation
- Target: 40-60% of original length

Text:
{text}

Compressed:"""


@dataclass
class CompressionResult:
    input_text: str
    output_text: str
    coverage: float
    confident: bool
    mode: str          # "compressed" | "raw_fallback" | "skipped"
    tokens_in: int
    tokens_out: int
    tokens_saved: int
    llm_output: str    # raw LLM response before validation

    def report(self) -> str:
        pct = f"{self.tokens_saved/max(self.tokens_in,1)*100:.0f}%"
        return (
            f"MODE:     {self.mode.upper()}\n"
            f"COVERAGE: {self.coverage*100:.1f}% "
            f"{'✓' if self.confident else '⚠ BELOW THRESHOLD → raw'}\n"
            f"TOKENS:   {self.tokens_in} → {self.tokens_out} "
            f"(-{self.tokens_saved}, {pct})"
        )


class LLMCompressEmbedValidate:
    """
    Two-stage pipeline:
      Stage 1 — LLM compresses (preserves conditionality)
      Stage 2 — Embedding validates (cosine similarity floor)
    """

    def __init__(
        self,
        threshold: float = THRESHOLD,
        min_tokens: int = MIN_TOKENS,
        compress_model: str = COMPRESS_MODEL,
        embed_model: str = EMBED_MODEL,
    ):
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.compress_model = compress_model
        self.embed_model = embed_model

    def process(self, text: str) -> CompressionResult:
        tokens_in = len(text) // 4

        # Skip pipeline for short prompts
        if tokens_in < self.min_tokens:
            return CompressionResult(
                input_text=text, output_text=text,
                coverage=1.0, confident=True, mode="skipped",
                tokens_in=tokens_in, tokens_out=tokens_in,
                tokens_saved=0, llm_output="",
            )

        # Stage 1: LLM compression
        compressed = self._llm_compress(text)

        # Stage 2: Embedding validation
        coverage = self._embed_validate(text, compressed)
        confident = coverage >= self.threshold
        output = compressed if confident else text
        tokens_out = len(output) // 4

        return CompressionResult(
            input_text=text,
            output_text=output,
            coverage=coverage,
            confident=confident,
            mode="compressed" if confident else "raw_fallback",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=tokens_in - tokens_out,
            llm_output=compressed,
        )

    def _llm_compress(self, text: str) -> str:
        prompt = COMPRESS_PROMPT.format(text=text)
        response = ollama.generate(
            model=self.compress_model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 256},
        )
        raw = response["response"].strip()
        # Strip any preamble the LLM adds
        for prefix in ["Compressed:", "Here is", "Sure", "Output:"]:
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        return raw

    def _embed_validate(self, original: str, compressed: str) -> float:
        orig_emb = np.array(
            ollama.embeddings(model=self.embed_model, prompt=original)["embedding"]
        )
        comp_emb = np.array(
            ollama.embeddings(model=self.embed_model, prompt=compressed)["embedding"]
        )
        return float(
            np.dot(orig_emb, comp_emb)
            / (np.linalg.norm(orig_emb) * np.linalg.norm(comp_emb) + 1e-9)
        )


# ─── Test suite ───────────────────────────────────────────────────────────────

TESTS = [
    ("FNC abstract",
     "This note examines two converging theoretical frameworks in contemporary "
     "consciousness research: Mohammad Forghani's Consciousness Frequency Model (2024) "
     "and Björn Wikström's Field–Node–Cockpit (FNC) Model (2025). Both models challenge "
     "the classical assumption that consciousness is a local, emergent byproduct of neural "
     "computation. Instead, they describe consciousness as a distributed phenomenon governed "
     "by resonance, coherence, and information exchange across different ontological layers. "
     "The result is a unified interpretation where frequency defines how consciousness "
     "manifests, and field topology defines why it manifests. This synthesis suggests a "
     "research trajectory that bridges analytic idealism, quantum information theory, and "
     "empirical hyperscanning data under one principle: consciousness as structured resonance."),

    ("Nyansrik kondition (exp_016 failure case)",
     "idén är att det bör vara reversibelt men bara under vissa förutsättningar "
     "när confidence är låg"),

    ("Svenska MCP-intent",
     "jag tänker att vi skulle behöva skapa en mcp server med cognos som kan "
     "validera och väga värdet semantiskt för förståelse"),

    ("Original session-prompt",
     "jag tänker att vi skulle behöva skapa en funktion som gör X inom strukturen "
     "för claude code. Idén är att ha en mcp server med cognos som kan vallidera "
     "och väga värdet semantiskt för förståelse. Detta bör då kunna skapa en bättre "
     "och mindre tokens användning?"),

    ("Kort prompt (skip)",
     "fix typo in readme"),
]

if __name__ == "__main__":
    pipeline = LLMCompressEmbedValidate()

    print("exp_017 — LLM Compress + Embed Validate")
    print(f"Compress: {COMPRESS_MODEL} | Validate: {EMBED_MODEL} | Threshold: {THRESHOLD*100:.0f}%")
    print(f"Min tokens for pipeline: {MIN_TOKENS}")
    print("=" * 65)

    total_in = total_out = 0

    for label, text in TESTS:
        print(f"\nTEST: {label}")
        r = pipeline.process(text)
        total_in  += r.tokens_in
        total_out += r.tokens_out

        print(f"  IN:  \"{text[:70]}{'...' if len(text)>70 else ''}\"")
        if r.mode == "compressed":
            print(f"  OUT: \"{r.output_text[:70]}{'...' if len(r.output_text)>70 else ''}\"")
        for line in r.report().split("\n"):
            print(f"  {line}")

        # Visa LLM-output även vid fallback (för analys)
        if r.llm_output and r.mode == "raw_fallback":
            print(f"  LLM tried: \"{r.llm_output[:70]}\"")

    total_saved = total_in - total_out
    print("\n" + "=" * 65)
    print(f"TOTALT: {total_in} → {total_out} tokens")
    print(f"Netto besparing: {total_saved} tokens ({total_saved/max(total_in,1)*100:.0f}%)")
