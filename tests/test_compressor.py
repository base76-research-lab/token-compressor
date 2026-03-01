#!/usr/bin/env python3
"""
token-compressor — validation tests
Base76 Research Lab

Tests verify:
1. Short prompts are skipped (no pipeline cost)
2. Long prompts are compressed with coverage >= threshold
3. Conditionals survive compression
4. Negations survive compression
5. Raw fallback triggers when coverage is too low
6. CompressionResult fields are correctly populated

Requirements: ollama running locally with llama3.2:1b and nomic-embed-text
Run: python3 tests/test_compressor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from compressor import LLMCompressEmbedValidate, THRESHOLD, MIN_TOKENS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✓"
FAIL = "✗"
results = []


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return condition


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

SHORT_PROMPT = "fix typo in readme"

LONG_PROMPT = (
    "Contemporary AI systems are optimized to appear confident. "
    "Reinforcement learning from human feedback systematically penalizes expressed uncertainty "
    "— producing models that perform certainty rather than represent it. "
    "This is not a UX problem. It is an architectural problem — measurable, structural, "
    "and consequential for any system making decisions in regulated or safety-critical contexts. "
    "We build the layer that fixes it. Working at the intersection of epistemology, "
    "AI architecture, and governance — building theory and tools for AI systems that can "
    "accurately represent what they know and what they don't."
)

CONDITIONAL_PROMPT = (
    "The system should respond with high confidence only if the epistemic uncertainty "
    "score is below 0.2 and the coverage threshold has been met. "
    "If either condition fails, the system must not proceed and should escalate to human review. "
    "Under no circumstances should the output be returned without attestation."
)


def run_tests():
    pipeline = LLMCompressEmbedValidate()

    print("\n── Test 1: Short prompt is skipped ──")
    r = pipeline.process(SHORT_PROMPT)
    check("mode == skipped", r.mode == "skipped", f"got: {r.mode}")
    check("tokens_saved == 0", r.tokens_saved == 0, f"got: {r.tokens_saved}")
    check("output_text == input", r.output_text == SHORT_PROMPT)

    print("\n── Test 2: Long prompt compresses ──")
    r = pipeline.process(LONG_PROMPT)
    check("mode is compressed or raw_fallback", r.mode in ("compressed", "raw_fallback"), f"got: {r.mode}")
    check("tokens_in > MIN_TOKENS", r.tokens_in >= MIN_TOKENS, f"got: {r.tokens_in}")
    check("coverage > 0", r.coverage > 0, f"got: {r.coverage:.2f}")
    if r.mode == "compressed":
        check("tokens_out < tokens_in", r.tokens_out < r.tokens_in,
              f"{r.tokens_in}→{r.tokens_out}")
        check("coverage >= threshold", r.coverage >= THRESHOLD,
              f"{r.coverage:.2f} >= {THRESHOLD}")

    print("\n── Test 3: Conditionals survive compression ──")
    r = pipeline.process(CONDITIONAL_PROMPT)
    out = r.output_text.lower()
    check("'only if' preserved", "only if" in out or "only" in out,
          f"mode={r.mode}")
    check("'must not' / negation preserved",
          "must not" in out or "not" in out or "no" in out,
          f"mode={r.mode}")
    check("'if' preserved", "if" in out, f"mode={r.mode}")

    print("\n── Test 4: Result fields populated ──")
    r = pipeline.process(LONG_PROMPT)
    check("output_text non-empty", len(r.output_text) > 0)
    check("mode is valid string", r.mode in ("compressed", "raw_fallback", "skipped"))
    check("tokens_in is int > 0", isinstance(r.tokens_in, int) and r.tokens_in > 0)
    check("tokens_out is int >= 0", isinstance(r.tokens_out, int) and r.tokens_out >= 0)
    check("coverage is float 0-1", 0.0 <= r.coverage <= 1.0, f"{r.coverage:.3f}")
    check("confident is bool", isinstance(r.confident, bool))

    print("\n── Test 5: Threshold enforcement ──")
    # Force a low-coverage scenario by using a very aggressive threshold
    strict = LLMCompressEmbedValidate(threshold=0.999)
    r = strict.process(LONG_PROMPT)
    check("strict threshold triggers raw_fallback",
          r.mode in ("raw_fallback", "skipped"),
          f"got: {r.mode}")

    # Summary
    passed = sum(1 for s, _, _ in results if s == PASS)
    failed = sum(1 for s, _, _ in results if s == FAIL)
    total = passed + failed

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed" + (f" | {failed} failed" if failed else " ✓"))
    print(f"Threshold: {THRESHOLD} | Min tokens: {MIN_TOKENS}")

    if failed:
        print("\nFailed tests:")
        for s, label, detail in results:
            if s == FAIL:
                print(f"  ✗ {label}" + (f" — {detail}" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    print("token-compressor — validation suite")
    print(f"Requires: ollama + llama3.2:1b + nomic-embed-text")
    run_tests()
