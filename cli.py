#!/usr/bin/env python3
"""
token-compressor CLI
Reads prompt from stdin or argument, compresses, writes to stdout.

Usage:
  echo "prompt text" | python3 cli.py
  python3 cli.py "prompt text"

Exit codes:
  0 = compressed or skipped (stdout = output text)
  1 = error (stdout = raw text)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read().strip()

    if not text:
        sys.exit(0)

    tokens = len(text) // 4

    if tokens < 80:
        print(text)
        sys.exit(0)

    try:
        from compressor import LLMCompressEmbedValidate
        pipeline = LLMCompressEmbedValidate()
        result = pipeline.process(text)

        # Skriv komprimerad text till stdout
        print(result.output_text)

        # Skriv statistik till stderr (syns i Claude Code hook-output)
        mode = result.mode.upper()
        saved = result.tokens_saved
        cov = result.coverage * 100
        print(
            f"[b76_compress] {mode} | {result.tokens_in}→{result.tokens_out}t "
            f"(-{saved}) | coverage {cov:.1f}%",
            file=sys.stderr
        )

    except Exception as e:
        # Alltid säker fallback — returnera råtext
        print(text)
        print(f"[b76_compress] ERROR: {e} — raw fallback", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()
