"""Scrape answered GitHub issues from ROCm repos into training pairs.

Uses the GitHub REST API (authenticated). Requires GITHUB_TOKEN env var.
Fetches closed issues with accepted answers from ROCm/HIP and ROCm/ROCm.

Target: ~200 pairs, categories: debugging, detection, concepts.

Usage:
    GITHUB_TOKEN=ghp_xxx python scrape_github.py [--output data/raw/github_pairs.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

import requests

from dataset import TrainingPair

DELAY = 2.1  # seconds between API calls (authenticated = 30 req/min → 2s safe)

REPOS = [
    "ROCm/HIP",
    "ROCm/ROCm",
    "ROCm/pytorch",       # common ROCm PyTorch issues
    "RadeonOpenCompute/rocm-cmake",  # build/setup issues
]

# Labels that indicate real developer questions (not bug reports or PRs).
QUESTION_LABELS = ["question", "help wanted", "faq", "documentation"]

# Minimum length for an answer to be useful.
MIN_ANSWER_CHARS = 150


def get_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print(
            "WARNING: GITHUB_TOKEN not set. GitHub API will rate-limit to 10 req/min.\n"
            "Set it with: export GITHUB_TOKEN=ghp_your_token_here"
        )
    return token or None


def make_headers(token: str | None) -> dict:
    h = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RiftTune-scraper/1.0",
    }
    if token:
        h["Authorization"] = f"token {token}"
    return h


def fetch_issues(repo: str, headers: dict, per_page: int = 100, max_pages: int = 3) -> list[dict]:
    issues: list[dict] = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://api.github.com/repos/{repo}/issues"
            f"?state=closed&per_page={per_page}&page={page}"
            f"&sort=updated&direction=desc"
        )
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            print(f"  Repo {repo} not found or not accessible")
            break
        if resp.status_code == 403:
            print(f"  Rate limited on {repo}. Waiting 60s...")
            time.sleep(60)
            continue
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # Filter out pull requests (issues endpoint returns both).
        issues.extend(i for i in batch if "pull_request" not in i)
        print(f"  {repo} page {page}: {len(batch)} issues ({len(issues)} kept)")
        time.sleep(DELAY)
    return issues


def fetch_best_comment(issue: dict, headers: dict) -> dict | None:
    """Fetch the most-upvoted comment, or the first long one."""
    comments_url = issue.get("comments_url", "")
    if not comments_url or issue.get("comments", 0) == 0:
        return None
    resp = requests.get(
        f"{comments_url}?per_page=30",
        headers=headers,
        timeout=15,
    )
    if resp.status_code != 200:
        return None
    comments = resp.json()
    time.sleep(DELAY)

    # Prefer the most-upvoted comment.
    qualified = [
        c for c in comments
        if len(c.get("body", "")) >= MIN_ANSWER_CHARS
    ]
    if not qualified:
        return None
    return max(qualified, key=lambda c: c.get("reactions", {}).get("+1", 0) + c.get("reactions", {}).get("total_count", 0))


def clean_github_markdown(text: str) -> str:
    """Normalize GitHub markdown for training use."""
    # Remove HTML comments.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Normalize excessive blank lines.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def infer_category(title: str, body: str) -> str:
    combined = (title + " " + body).lower()
    if any(w in combined for w in ["not detected", "not found", "install", "driver", "rocm-smi", "device count"]):
        return "detection"
    if any(w in combined for w in ["error", "fail", "crash", "segfault", "exception", "abort", "undefined"]):
        return "debugging"
    if any(w in combined for w in ["port", "migrat", "equivalent", "cuda to hip", "hipify"]):
        return "api_translation"
    if any(w in combined for w in ["optim", "slow", "performance", "bandwidth", "occupancy"]):
        return "optimization"
    return "concepts"


def issue_to_pair(issue: dict, comment: dict, repo: str) -> TrainingPair | None:
    title = issue.get("title", "").strip()
    body = clean_github_markdown(issue.get("body", "") or "")
    answer = clean_github_markdown(comment.get("body", ""))

    if len(title) < 10 or len(answer) < MIN_ANSWER_CHARS:
        return None

    # Build instruction: question title, optionally with body context.
    instruction = title
    if not instruction.endswith("?"):
        instruction = instruction.rstrip(".") + "?"

    # Build input: issue body if it contains code or error output.
    input_text = ""
    if body and (
        "```" in body
        or "error" in body.lower()
        or "traceback" in body.lower()
        or len(body) < 800
    ):
        input_text = body[:1500]  # cap body length

    issue_url = issue.get("html_url", f"https://github.com/{repo}/issues/{issue['number']}")
    output = answer + f"\n\nSource: {issue_url}"

    category = infer_category(title, body + answer)

    return TrainingPair(
        instruction=instruction,
        input=input_text,
        output=output,
        source="github",
        category=category,  # type: ignore[arg-type]
        language="en",
        verified=False,
    )


def is_hip_rocm_relevant(issue: dict) -> bool:
    """Return True if the issue is likely about HIP or ROCm development."""
    text = (
        (issue.get("title") or "")
        + " "
        + (issue.get("body") or "")
    ).lower()
    keywords = {"hip", "rocm", "hipify", "amdgpu", "mi300", "wavefront", "hipmalloc", "hipstream"}
    return any(kw in text for kw in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape GitHub issues into training pairs")
    parser.add_argument("--output", default="data/raw/github_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--repos", nargs="*", default=REPOS)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    token = get_token()
    headers = make_headers(token)

    total = 0
    with output.open("w", encoding="utf-8") as f:
        for repo in args.repos:
            if total >= args.limit:
                break
            print(f"\nFetching issues from {repo}")
            try:
                issues = fetch_issues(repo, headers)
            except Exception as e:
                print(f"  Failed: {e}")
                continue

            relevant = [i for i in issues if is_hip_rocm_relevant(i)]
            print(f"  {len(relevant)}/{len(issues)} relevant issues")

            for issue in relevant:
                if total >= args.limit:
                    break
                comment = fetch_best_comment(issue, headers)
                if not comment:
                    continue
                pair = issue_to_pair(issue, comment, repo)
                if not pair:
                    continue
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                total += 1

            print(f"  Pairs written so far: {total}")

    print(f"\nTotal pairs written: {total}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
