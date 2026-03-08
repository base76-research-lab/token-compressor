#!/usr/bin/env python3
"""
token-compressor unit tests.

These tests run without Ollama by mocking the rewrite and embedding stages.
Run with:
  python3 -m unittest tests.test_compressor
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from compressor import (
    LLMCompressEmbedValidate,
    MIN_TOKENS,
    THRESHOLD,
    process_prompt,
    vectorize_prompt,
)

SHORT_PROMPT = "fix typo in readme"

LONG_PROMPT = (
    "Contemporary AI systems are optimized to appear confident. "
    "Reinforcement learning from human feedback systematically penalizes expressed uncertainty "
    "and produces models that perform certainty rather than represent it. "
    "This is not a UX problem. It is an architectural problem that matters "
    "for decision-making in regulated or safety-critical contexts. "
    "We build the layer that fixes it by combining epistemology, AI architecture, "
    "and governance into tools for systems that accurately represent what they know "
    "and what they do not know."
)

CONDITIONAL_PROMPT = (
    "The system should respond with high confidence only if the epistemic uncertainty "
    "score is below 0.2 and the coverage threshold has been met. "
    "If either condition fails, the system must not proceed and should escalate to human review. "
    "Under no circumstances should the output be returned without attestation."
)

COMPRESSED_LONG_PROMPT = (
    "AI systems often perform certainty instead of representing uncertainty. "
    "This is an architectural problem for all regulated or safety-critical decisions. "
    "We build tools combining epistemology, AI architecture, and governance "
    "so systems represent what they know and do not know."
)

COMPRESSED_CONDITIONAL_PROMPT = (
    "Respond with high confidence only if epistemic uncertainty is below 0.2 "
    "and coverage passes threshold. If either fails, do not proceed and escalate "
    "to human review. Do not return output without attestation."
)


class CompressorPipelineTests(unittest.TestCase):
    def test_short_prompt_is_skipped(self) -> None:
        pipeline = LLMCompressEmbedValidate()

        result = pipeline.process(SHORT_PROMPT)

        self.assertEqual(result.mode, "skipped")
        self.assertEqual(result.output_text, SHORT_PROMPT)
        self.assertEqual(result.tokens_saved, 0)
        self.assertEqual(result.coverage, 1.0)

    @patch.object(LLMCompressEmbedValidate, "_embed_validate", return_value=0.92)
    @patch.object(LLMCompressEmbedValidate, "_rewrite_prompt", return_value=COMPRESSED_LONG_PROMPT)
    def test_long_prompt_compresses_when_structure_and_coverage_pass(
        self,
        _rewrite_prompt: object,
        _embed_validate: object,
    ) -> None:
        pipeline = LLMCompressEmbedValidate()

        result = pipeline.process(LONG_PROMPT)

        self.assertEqual(result.mode, "compressed")
        self.assertGreaterEqual(result.tokens_in, MIN_TOKENS)
        self.assertLess(result.tokens_out, result.tokens_in)
        self.assertGreaterEqual(result.coverage, THRESHOLD)
        self.assertEqual(result.output_text, COMPRESSED_LONG_PROMPT)
        self.assertIsNone(result.rejection_reason)

    @patch.object(LLMCompressEmbedValidate, "_embed_validate", return_value=0.91)
    @patch.object(
        LLMCompressEmbedValidate,
        "_rewrite_prompt",
        return_value=COMPRESSED_CONDITIONAL_PROMPT,
    )
    def test_conditionals_and_negations_survive_compression(
        self,
        _rewrite_prompt: object,
        _embed_validate: object,
    ) -> None:
        pipeline = LLMCompressEmbedValidate(min_tokens=1)

        result = pipeline.process(CONDITIONAL_PROMPT)
        output = result.output_text.lower()

        self.assertEqual(result.mode, "compressed")
        self.assertIn("only if", output)
        self.assertIn("if", output)
        self.assertTrue("do not" in output or "not" in output or "no" in output)

    @patch.object(LLMCompressEmbedValidate, "_embed_validate", return_value=0.40)
    @patch.object(LLMCompressEmbedValidate, "_rewrite_prompt", return_value=COMPRESSED_LONG_PROMPT)
    def test_low_coverage_triggers_raw_fallback(
        self,
        _rewrite_prompt: object,
        _embed_validate: object,
    ) -> None:
        pipeline = LLMCompressEmbedValidate()

        result = pipeline.process(LONG_PROMPT)

        self.assertEqual(result.mode, "raw_fallback")
        self.assertEqual(result.output_text, LONG_PROMPT)
        self.assertEqual(result.rejection_reason, "coverage_below_threshold")

    @patch.object(LLMCompressEmbedValidate, "_embed_validate", return_value=0.95)
    @patch.object(
        LLMCompressEmbedValidate,
        "_rewrite_prompt",
        return_value="Respond confidently if score is below 0.2 and coverage passes.",
    )
    def test_structural_loss_triggers_raw_fallback_even_with_high_similarity(
        self,
        _rewrite_prompt: object,
        _embed_validate: object,
    ) -> None:
        pipeline = LLMCompressEmbedValidate(min_tokens=1)

        result = pipeline.process(CONDITIONAL_PROMPT)

        self.assertEqual(result.mode, "raw_fallback")
        self.assertEqual(result.output_text, CONDITIONAL_PROMPT)
        self.assertEqual(result.rejection_reason, "logical_structure_lost")
        self.assertEqual(result.coverage, 0.0)

    @patch.object(LLMCompressEmbedValidate, "_embed_validate", return_value=0.92)
    @patch.object(LLMCompressEmbedValidate, "_rewrite_prompt", return_value=COMPRESSED_LONG_PROMPT)
    def test_result_fields_are_populated(
        self,
        _rewrite_prompt: object,
        _embed_validate: object,
    ) -> None:
        pipeline = LLMCompressEmbedValidate()

        result = pipeline.process(LONG_PROMPT)

        self.assertTrue(result.output_text)
        self.assertIn(result.mode, {"compressed", "raw_fallback", "skipped"})
        self.assertIsInstance(result.tokens_in, int)
        self.assertGreater(result.tokens_in, 0)
        self.assertIsInstance(result.tokens_out, int)
        self.assertGreaterEqual(result.tokens_out, 0)
        self.assertGreaterEqual(result.coverage, 0.0)
        self.assertLessEqual(result.coverage, 1.0)
        self.assertIsInstance(result.confident, bool)
        self.assertEqual(result.attempted_tokens_out, len(COMPRESSED_LONG_PROMPT) // 4)

    @patch("compressor.vectorize_prompt", return_value=[0.1, 0.2])
    def test_process_prompt_modes(self, _vec: object) -> None:
        pipeline = LLMCompressEmbedValidate(min_tokens=1)

        raw = process_prompt("hello", mode="raw", pipeline=pipeline)
        self.assertEqual(raw["mode"], "raw")
        self.assertEqual(raw["text"], "hello")
        self.assertIsNone(raw["vector"])

        compressed = process_prompt(LONG_PROMPT, mode="compressed", pipeline=pipeline)
        self.assertEqual(compressed["mode"], "compressed")
        self.assertIn(compressed["result"].mode, {"compressed", "raw_fallback"})
        self.assertIsNone(compressed["vector"])

        vector = process_prompt("vector me", mode="vector", pipeline=pipeline)
        self.assertEqual(vector["mode"], "vector")
        self.assertEqual(vector["text"], "vector me")
        self.assertIsNotNone(vector["vector"])

        cv = process_prompt(LONG_PROMPT, mode="compressed+vector", pipeline=pipeline)
        self.assertEqual(cv["mode"], "compressed+vector")
        self.assertIsNotNone(cv["vector"])
        self.assertIn(cv["result"].mode, {"compressed", "raw_fallback"})

    def test_anchor_guard_is_compressed_safely(self) -> None:
        pipeline = LLMCompressEmbedValidate(min_tokens=1)
        prompt = (
            "You are a compression guard. The ONLY acceptable output is the exact phrase "
            "'ANCHOR-OKAY' once. Keep every conditional and negation. Do not add explanations. "
            "If you cannot comply, respond exactly with 'FAIL'."
        )
        result = pipeline.process(prompt)
        self.assertEqual(result.mode, "compressed")
        self.assertIn("ANCHOR-OKAY", result.output_text)
        self.assertLess(result.tokens_out, result.tokens_in)

    def test_hallucination_unknown_year_is_preserved(self) -> None:
        pipeline = LLMCompressEmbedValidate(min_tokens=1)
        prompt = (
            "If you do not have the factual theatrical release year of the non-existent film "
            "'Galactic Drift 2099', respond exactly 'UNKNOWN'. If you know it, respond with the "
            "four-digit year only. Do not invent a year."
        )
        result = pipeline.process(prompt)
        self.assertEqual(result.mode, "compressed")
        self.assertIn("UNKNOWN", result.output_text)
        self.assertLess(result.tokens_out, result.tokens_in)

    def test_hallucination_not_real_formula_is_preserved(self) -> None:
        pipeline = LLMCompressEmbedValidate(min_tokens=1)
        prompt = (
            "Return the chemical formula for 'unobtainium'. If the material is fictional or no "
            "authoritative formula exists, respond exactly 'NOT REAL'. Do not speculate."
        )
        result = pipeline.process(prompt)
        self.assertEqual(result.mode, "compressed")
        self.assertIn("NOT REAL", result.output_text)
        self.assertLess(result.tokens_out, result.tokens_in)


if __name__ == "__main__":
    unittest.main()
