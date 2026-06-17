"""Tiny CLI to match hospital names with the production cascade.

Examples:
    python run_demo.py "RS ELISABET SEMARNG"          # one name
    python run_demo.py "APOTIK ASEAN" "GYNAE ONCO"    # several names
    python run_demo.py --no-sbert "HOSPITAL PAKAR DAMANSARA"   # lexical only
    python run_demo.py                                # interactive prompt

First run builds + caches the lexical artifact (~90s); later runs load in <1s.
SBERT (when enabled) is loaded lazily and only if a row needs it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline import HospitalMatcher


def show(row):
    flag = "  [NEEDS REVIEW]" if row["needs_review"] else ""
    print(f"\n  input      : {row['input']}")
    print(f"  prediction : {row['prediction']}{flag}")
    if row["needs_review"] and row.get("suggestion"):
        print(f"  suggestion : {row['suggestion']}")
    print(f"  confidence : {row['confidence']:.2f}   (resolved at stage: {row['stage']})")


def main():
    parser = argparse.ArgumentParser(description="Match messy hospital names to canonical names.")
    parser.add_argument("names", nargs="*", help="hospital name(s) to match")
    parser.add_argument("--no-sbert", action="store_true", help="pure-lexical mode (no SBERT weights needed)")
    args = parser.parse_args()

    print("Loading matcher (first run builds the cache, ~90s)...")
    matcher = HospitalMatcher.load_or_build(use_sbert=not args.no_sbert)
    print(f"Ready. Mode: {'lexical-only' if args.no_sbert else 'lexical + SBERT'}")

    if args.names:
        for row in matcher.predict_batch(args.names).to_dict("records"):
            show(row)
        return

    print("\nType a hospital name and press Enter (blank line to quit):")
    while True:
        try:
            name = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not name:
            break
        show(matcher.predict_one(name))


if __name__ == "__main__":
    main()
