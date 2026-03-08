#!/usr/bin/env python3
"""
token-compressor CLI

Usage:
  echo "prompt text" | python3 cli.py [--mode compressed|raw|vector|compressed+vector]
  python3 cli.py --mode vector "prompt text"

Modes:
  raw                return original text
  compressed         default; return compressed text (or raw fallback)
  vector             return JSON {text, vector}
  compressed+vector  compress then return JSON {text, vector}

Exit codes:
  0 = success (stdout contains text or JSON)
  1 = error (stdout = raw text)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=["raw", "compressed", "vector", "compressed+vector", "compress+vector"],
        default="compressed",
    )
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()

    if args.text:
        text = " ".join(args.text)
    else:
        text = sys.stdin.read().strip()

    if not text:
        sys.exit(0)

    try:
        from compressor import process_prompt

        out = process_prompt(text, mode=args.mode)
        mode = out["mode"]
        vector = out["vector"]
        res = out["result"]
        output_text = out["text"]

        if vector is not None:
            print(json.dumps({"mode": mode, "text": output_text, "vector": vector}))
        else:
            print(output_text)

        if res:
            saved = res.tokens_saved
            cov = res.coverage * 100
            print(
                f"[b76_compress] {mode.upper()} | {res.tokens_in}→{res.tokens_out}t "
                f"(-{saved}) | coverage {cov:.1f}%",
                file=sys.stderr,
            )
        else:
            print(f"[b76_compress] {mode.upper()}", file=sys.stderr)

    except Exception as e:
        print(text)
        print(f"[b76_compress] ERROR: {e} — raw fallback", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
