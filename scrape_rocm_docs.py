"""Scrape ROCm documentation pages into instruction-output training pairs.

Targets:
  - HIP Porting Guide (porting concepts, API equivalences)
  - HIP FAQ (real developer questions)
  - HIP Programming Guide sections (optimization, concepts)
  - ROCm installation / GPU detection guides (detection category)

Generates ~400 pairs targeting api_translation, concepts, optimization, detection.

Usage:
    python scrape_rocm_docs.py [--output data/raw/rocm_docs.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup, Tag

from dataset import TrainingPair

HEADERS = {"User-Agent": "RiftTune-scraper/1.0"}
DELAY = 0.4  # seconds between requests


# Pages to scrape, with their category hint.
PAGES: list[tuple[str, str]] = [
    # porting guide — rich source of api_translation + concepts
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html", "api_translation"),
    # programming guide sections
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_model.html", "concepts"),
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html", "optimization"),
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_compilation.html", "concepts"),
    # kernel language reference — maps CUDA keywords to HIP
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html", "api_translation"),
    # math functions
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_functions.html", "api_translation"),
    # error handling
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/reference/error_handling.html", "debugging"),
    # ROCm install / GPU detection
    ("https://rocm.docs.amd.com/en/latest/tutorial/install-overview.html", "detection"),
    ("https://rocm.docs.amd.com/en/latest/how-to/system-optimization/index.html", "optimization"),
    # HIP FAQ — ready-made Q&A format
    ("https://rocm.docs.amd.com/projects/HIP/en/latest/reference/faq.html", "concepts"),
]


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        if resp.status_code == 200:
            return resp.text
        print(f"  HTTP {resp.status_code}: {url}")
    except requests.RequestException as e:
        print(f"  Error fetching {url}: {e}")
    return None


def clean_text(element: Tag) -> str:
    """Extract clean text, preserving code blocks as fenced markdown."""
    parts: list[str] = []
    for child in element.descendants:
        if not hasattr(child, "name"):
            # NavigableString
            text = str(child)
            if text.strip():
                parts.append(text)
        elif child.name in ("code", "tt"):
            parts.append(f"`{child.get_text()}`")
        elif child.name == "pre":
            code = child.get_text()
            lang = "cpp" if any(kw in code for kw in ["hip", "__global__", "kernel"]) else ""
            parts.append(f"\n```{lang}\n{code.strip()}\n```\n")
    text = " ".join(parts)
    # Collapse multiple spaces/newlines but preserve paragraph breaks.
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def section_to_question(heading: str, category: str) -> str:
    """Heuristically convert a section heading to a question."""
    h = heading.strip().rstrip(".")

    # Direct question headings (FAQ-style).
    if h.endswith("?"):
        return h

    # "How to X" → "How do I X?"
    m = re.match(r"(?:How to|How To)\s+(.+)", h, re.IGNORECASE)
    if m:
        return f"How do I {m.group(1).lower()}?"

    # "Using X" → "How do I use X?"
    m = re.match(r"Using\s+(.+)", h, re.IGNORECASE)
    if m:
        return f"How do I use {m.group(1)}?"

    # "Porting X" / "Migrating X" → "How do I port/migrate X?"
    m = re.match(r"(Porting|Migrating)\s+(.+)", h, re.IGNORECASE)
    if m:
        return f"How do I {m.group(1).lower()} {m.group(2).lower()}?"

    # "X vs Y" → "What is the difference between X and Y?"
    m = re.match(r"(.+)\s+vs\.?\s+(.+)", h, re.IGNORECASE)
    if m:
        return f"What is the difference between {m.group(1)} and {m.group(2)}?"

    # Category-specific defaults.
    if category == "api_translation":
        return f"How do I {h.lower()} in HIP?"
    if category == "optimization":
        return f"How do I optimize {h.lower()} on AMD hardware?"
    if category == "detection":
        return f"How do I {h.lower()}?"
    return f"What is {h} in ROCm/HIP?"


def extract_pairs_from_page(html: str, url: str, default_category: str) -> Iterator[TrainingPair]:
    soup = BeautifulSoup(html, "lxml")

    # Remove navigation, header, footer noise.
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    # Find the main content area (Sphinx docs use div.body or article).
    main = (
        soup.find("div", class_="body")
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
        or soup.body
    )
    if not main:
        return

    # Walk sections: h2 and h3 headings introduce sections.
    current_heading: str | None = None
    current_content: list[str] = []
    current_category = default_category

    def flush(heading: str, content_parts: list[str], category: str) -> TrainingPair | None:
        content = "\n\n".join(p for p in content_parts if p.strip())
        if len(content) < 100 or len(heading) < 5:
            return None
        question = section_to_question(heading, category)
        citation = f"\nSource: {url}"
        output = content + citation
        return TrainingPair(
            instruction=question,
            input="",
            output=output,
            source="rocm_docs",
            category=category,  # type: ignore[arg-type]
            language="en",
            verified=False,
        )

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "ul", "ol", "table"]):
        if element.name in ("h1", "h2", "h3", "h4"):
            if current_heading and current_content:
                pair = flush(current_heading, current_content, current_category)
                if pair:
                    yield pair
            current_heading = element.get_text(strip=True)
            current_content = []
            # Refine category based on heading keywords.
            h_lower = current_heading.lower()
            if any(w in h_lower for w in ["port", "migrat", "equivalent", "convert"]):
                current_category = "api_translation"
            elif any(w in h_lower for w in ["debug", "error", "fail", "fix", "issue"]):
                current_category = "debugging"
            elif any(w in h_lower for w in ["optim", "perform", "bandwidth", "throughput", "latency"]):
                current_category = "optimization"
            elif any(w in h_lower for w in ["detect", "install", "setup", "driver", "not found"]):
                current_category = "detection"
            else:
                current_category = default_category
        else:
            text = clean_text(element)
            if text:
                current_content.append(text)

    if current_heading and current_content:
        pair = flush(current_heading, current_content, current_category)
        if pair:
            yield pair


def scrape_faq_pairs(html: str, url: str) -> Iterator[TrainingPair]:
    """FAQ pages have explicit Q&A structure — extract directly."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    # FAQ pattern: dt (question) + dd (answer), or h3 (question) + next sibling p (answer).
    for dt in soup.find_all("dt"):
        question = dt.get_text(strip=True)
        if not question.endswith("?") and len(question) < 200:
            question = question.rstrip(".") + "?"
        dd = dt.find_next_sibling("dd")
        if dd:
            answer = clean_text(dd)
            if len(answer) >= 100:
                yield TrainingPair(
                    instruction=question,
                    input="",
                    output=answer + f"\nSource: {url}",
                    source="rocm_docs",
                    category="concepts",
                    language="en",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ROCm docs into training pairs")
    parser.add_argument("--output", default="data/raw/rocm_docs.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Limit total pairs (0=unlimited)")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output.open("w", encoding="utf-8") as f:
        for url, category in PAGES:
            print(f"Fetching {url}")
            html = fetch(url)
            if not html:
                continue

            is_faq = "faq" in url.lower()
            extractor = scrape_faq_pairs(html, url) if is_faq else extract_pairs_from_page(html, url, category)

            page_count = 0
            for pair in extractor:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                page_count += 1
                total += 1
                if args.limit and total >= args.limit:
                    break

            print(f"  → {page_count} pairs")
            if args.limit and total >= args.limit:
                break
            time.sleep(DELAY)

    print(f"\nTotal pairs written: {total}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
