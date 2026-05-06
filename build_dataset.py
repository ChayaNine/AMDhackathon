"""Merge, deduplicate, validate, and split all raw scraped data.

Reads from data/raw/*.jsonl, applies MinHash LSH deduplication,
runs is_valid_pair validation, then splits into train.jsonl (all valid pairs
minus the eval set) and eval.jsonl (50 manually curated pairs).

Usage:
    python build_dataset.py [--eval-seed data/eval_seed.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from dataset import TrainingPair, is_valid_pair, load_api_names

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
EVAL_SIZE = 50
NUM_PERM = 128
DEDUP_THRESHOLD_INSTRUCTION = 0.80
DEDUP_THRESHOLD_OUTPUT = 0.90

# Category targets for the eval set (must be balanced).
EVAL_TARGETS = {
    "api_translation": 15,
    "debugging": 10,
    "optimization": 10,
    "concepts": 10,
    "detection": 5,
}


def build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for word in text.lower().split():
        m.update(word.encode("utf-8"))
    return m


def load_all_raw() -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    for jsonl_file in sorted(RAW_DIR.glob("*.jsonl")):
        count = 0
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                pairs.append(TrainingPair.from_dict(d))
                count += 1
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"  Skip malformed line in {jsonl_file.name}: {e}")
        print(f"  {jsonl_file.name}: {count} pairs loaded")
    return pairs


def deduplicate(pairs: list[TrainingPair]) -> list[TrainingPair]:
    """Remove near-duplicate pairs using MinHash LSH on instruction + output."""
    print(f"\nDeduplicating {len(pairs)} pairs...")

    lsh_instr = MinHashLSH(threshold=DEDUP_THRESHOLD_INSTRUCTION, num_perm=NUM_PERM)
    lsh_output = MinHashLSH(threshold=DEDUP_THRESHOLD_OUTPUT, num_perm=NUM_PERM)

    kept: list[TrainingPair] = []
    for i, pair in enumerate(pairs):
        key = str(i)
        mh_i = build_minhash(pair.instruction)
        mh_o = build_minhash(pair.output)

        dup_i = lsh_instr.query(mh_i)
        dup_o = lsh_output.query(mh_o)

        if dup_i or dup_o:
            continue  # duplicate

        lsh_instr.insert(key, mh_i)
        lsh_output.insert(key, mh_o)
        kept.append(pair)

    removed = len(pairs) - len(kept)
    print(f"  Removed {removed} duplicates ({removed/len(pairs)*100:.1f}%)")
    print(f"  Remaining: {len(kept)}")
    return kept


def validate(pairs: list[TrainingPair], api_names: set[str]) -> list[TrainingPair]:
    """Filter pairs through is_valid_pair."""
    valid = [p for p in pairs if is_valid_pair(p, api_names)]
    removed = len(pairs) - len(valid)
    print(f"\nValidation: removed {removed} invalid pairs, {len(valid)} remaining")

    # Print breakdown by category.
    from collections import Counter
    counts = Counter(p.category for p in valid)
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")

    return valid


def select_eval_set(
    pairs: list[TrainingPair],
    seed_path: Path | None = None,
) -> tuple[list[TrainingPair], list[TrainingPair]]:
    """Split into eval (50 balanced pairs) and train (remainder).

    If a seed file exists, those pairs are used as the eval set directly
    (they should already be manually verified). Otherwise, we auto-select
    representative pairs (for initial use — replace with verified set).
    """
    if seed_path and seed_path.exists():
        print(f"\nLoading eval seed from {seed_path}")
        eval_pairs: list[TrainingPair] = []
        seed_instructions: set[str] = set()
        for line in seed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                p = TrainingPair.from_dict(d)
                p.verified = True
                eval_pairs.append(p)
                seed_instructions.add(p.instruction)
        train_pairs = [p for p in pairs if p.instruction not in seed_instructions]
        print(f"  Eval: {len(eval_pairs)} (from seed)")
        print(f"  Train: {len(train_pairs)}")
        return eval_pairs, train_pairs

    # Auto-select: pick proportionally by category.
    print("\nAuto-selecting eval set (no seed file found)")
    print("NOTE: Replace with manually verified pairs before final submission.")

    by_category: dict[str, list[TrainingPair]] = {}
    for p in pairs:
        by_category.setdefault(p.category, []).append(p)

    eval_pairs = []
    used_indices: set[int] = set()

    for cat, target in EVAL_TARGETS.items():
        candidates = by_category.get(cat, [])
        random.shuffle(candidates)
        selected = candidates[:target]
        eval_pairs.extend(selected)
        for p in selected:
            used_indices.add(id(p))

    # Mark as NOT verified (auto-selected needs human review).
    for p in eval_pairs:
        p.verified = False

    train_pairs = [p for p in pairs if id(p) not in used_indices]

    print(f"  Eval: {len(eval_pairs)} auto-selected (NEEDS MANUAL VERIFICATION)")
    print(f"  Train: {len(train_pairs)}")
    return eval_pairs, train_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RiftTune training dataset")
    parser.add_argument("--eval-seed", default="data/eval_seed.jsonl",
                        help="Path to manually curated eval pairs (optional)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load API names for validation.
    api_names = load_api_names()
    if not api_names:
        print("WARNING: data/hip_api_names.json not found.")
        print("Run: python build_api_allowlist.py")
        print("Continuing without api_translation validation (category check disabled).\n")

    # Load all raw data.
    print("Loading raw data...")
    all_pairs = load_all_raw()
    print(f"Total raw pairs: {len(all_pairs)}")

    if not all_pairs:
        print("ERROR: No raw data found. Run the scrapers first:")
        print("  python scrape_rocm_docs.py")
        print("  python scrape_hipify.py")
        print("  python scrape_gpuopen.py")
        print("  python scrape_github.py")
        return

    # Deduplicate.
    deduped = deduplicate(all_pairs)

    # Validate.
    valid = validate(deduped, api_names)

    if len(valid) < 200:
        print(f"\nWARNING: Only {len(valid)} valid pairs. Target is 800+.")
        print("Consider running scrapers again or adding manual pairs.")

    # Split eval / train.
    seed_path = Path(args.eval_seed)
    eval_pairs, train_pairs = select_eval_set(valid, seed_path if seed_path.exists() else None)

    # Write output files.
    train_out = OUTPUT_DIR / "train.jsonl"
    eval_out = OUTPUT_DIR / "eval.jsonl"

    with train_out.open("w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    with eval_out.open("w", encoding="utf-8") as f:
        for p in eval_pairs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train_pairs)} training pairs → {train_out}")
    print(f"Wrote {len(eval_pairs)} eval pairs → {eval_out}")
    print("\nNext steps:")
    if not seed_path.exists():
        print("  1. Manually verify eval.jsonl pairs against rocm.docs.amd.com")
        print("     Set verified=true on each confirmed pair, save as data/eval_seed.jsonl")
        print("     Re-run: python build_dataset.py --eval-seed data/eval_seed.jsonl")
    print("  2. Run: python train.py --train data/train.jsonl --eval data/eval.jsonl")


if __name__ == "__main__":
    main()
