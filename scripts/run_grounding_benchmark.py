"""
CLI Runner for the 16-Paper Evaluation Harness (Extraction, Gaps, Contradictions & Ablation).
"""

import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "src"))

from hypothesisforge.grounding_benchmark import run_grounding_benchmark


if __name__ == "__main__":
    print("=" * 94)
    print("HYPOTHESISFORGE - 16-PAPER EVALUATION HARNESS & ABLATION")
    print("=" * 94)
    results = run_grounding_benchmark()
    print(f"{'System':<33} {'Prec%':>7} {'Rec%':>7} {'Unsup%':>8} {'Quote%':>8} {'GapP%':>8} {'ConP%':>8}")
    print("-" * 94)
    for m in results:
        print(
            f"{m.system_name:<33} "
            f"{m.precision:>6.1f}% {m.recall:>6.1f}% "
            f"{m.unsupported_claim_rate:>7.1f}% {m.quote_pass_rate:>7.1f}% "
            f"{m.gap_precision:>7.1f}% {m.contradiction_precision:>7.1f}%"
        )
    print("=" * 94)
