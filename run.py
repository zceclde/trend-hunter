#!/usr/bin/env python3
"""
Diimsum Trend Hunter Solution: Early-Warning Community Radar
Entrypoint CLI: python run.py --input voices.jsonl --as-of YYYY-MM-DD --out trends.json
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Ensure src can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from src.ingest import load_data
from src.preprocess import TextPreprocessor
from src.clusterer import TrendClusterer
from src.synthesizer import TrendSynthesizer

def validate_date(date_str: str) -> str:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")

def main():
    parser = argparse.ArgumentParser(
        description="Diimsum Trend Hunter: Spot emerging gaming community trends dynamically."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="voices.jsonl",
        help="Path to voices.jsonl (raw player messages)"
    )
    parser.add_argument(
        "--as-of",
        type=validate_date,
        required=True,
        help="Evaluation cutoff date (YYYY-MM-DD). Data after this date will be dropped."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="trends.json",
        help="Path to output trends.json"
    )
    parser.add_argument(
        "--versions",
        type=str,
        default="versions.csv",
        help="Path to versions.csv (studio patch calendar)"
    )

    args = parser.parse_args()

    print(f"==================================================")
    print(f" Diimsum Trend Hunter Radar")
    print(f" Cutoff Date (--as-of): {args.as_of}")
    print(f" Input: {args.input} | Output: {args.out}")
    print(f"==================================================")

    # 1. Ingestion & Temporal Gate
    print(f"[1/4] Ingesting data with strict chronological cutoff ({args.as_of} 23:59:59)...")
    raw_messages, versions = load_data(
        input_path=args.input,
        as_of=args.as_of,
        versions_path=args.versions
    )
    print(f"      Valid historical messages: {len(raw_messages)}")
    print(f"      Known studio patches/events up to cutoff: {len(versions)}")

    if not raw_messages:
        print("[Warning] No messages found on or before the specified as-of date.")
        output_payload = {"as_of": args.as_of, "trends": []}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        return

    # 2. Text Preprocessing & Context Stitching
    print("[2/4] Normalizing scripts (OpenCC Trad->Simp) and stitching reply contexts...")
    preprocessor = TextPreprocessor()
    processed_messages = preprocessor.process_messages(raw_messages)

    # 3. Unsupervised Clustering & Velocity Analysis
    print("[3/4] Running dynamic semantic clustering and temporal velocity analysis...")
    clusterer = TrendClusterer(window_days=14)
    top_clusters = clusterer.cluster_messages(
        messages=processed_messages,
        as_of=args.as_of,
        versions=versions
    )
    print(f"      Identified {len(top_clusters)} significant trend candidates.")

    # 4. LLM / Studio Synthesis & Output
    print("[4/4] Synthesizing trend titles, dynamic categories, and justifications...")
    synthesizer = TrendSynthesizer()
    output_payload = synthesizer.synthesize_trends(
        clusters=top_clusters,
        as_of=args.as_of,
        versions=versions
    )

    # Write output JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated {len(output_payload['trends'])} trends saved to {args.out}:")
    for idx, t in enumerate(output_payload["trends"], 1):
        print(f"  {idx}. [{t['status'].upper()}] {t['title']} ({t['title_en']})")
        print(f"      Category: {t['category']} | First Seen: {t['first_seen']}")
        print(f"      Why: {t['why']}")
        print(f"      Voice IDs: {len(t['voice_ids'])} exemplars\n")

if __name__ == "__main__":
    main()

