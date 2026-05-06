"""Scrape GPUOpen technical blog posts into instruction-output training pairs.

Uses rule-based section extraction — no external LLM required.
Each section heading + content block becomes one Q&A pair.

Target: ~300 pairs, categories: optimization, api_translation, concepts.

Usage:
    python scrape_gpuopen.py [--output data/raw/gpuopen_pairs.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from dataset import TrainingPair

HEADERS = {"User-Agent": "RiftTune-scraper/1.0"}
DELAY = 0.6
MAX_POSTS = 60  # cap to avoid hammering the site

GPUOPEN_BASE = "https://gpuopen.com"

# Blog index pages focused on HIP/ROCm content.
INDEX_URLS = [
    "https://gpuopen.com/learn/",
    "https://gpuopen.com/category/hip/",
    "https://gpuopen.com/category/rocm/",
    "https://gpuopen.com/category/performance/",
]

# Known high-quality HIP/ROCm articles (direct URLs as fallback if index scraping misses them).
KNOWN_POSTS = [
    "https://gpuopen.com/learn/porting-cuda-to-hip/",
    "https://gpuopen.com/learn/hip-memory-management/",
    "https://gpuopen.com/learn/hip-streams-asynchronous-execution/",
    "https://gpuopen.com/learn/amd-lab-notes/",
    "https://gpuopen.com/learn/wavefront-size-consideration/",
    "https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-with-large-thread-groups/",
    "https://gpuopen.com/learn/rocm-5-3-hip-improvements/",
]

# Keywords that indicate a post is HIP/ROCm relevant.
RELEVANCE_KEYWORDS = {
    "hip", "rocm", "cuda", "amd gpu", "opencl", "wavefront", "mi300", "vram",
    "hip api", "hipify", "porting", "kernel", "__global__", "hipstream",
}


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        if resp.status_code == 200:
            return resp.text
        print(f"  HTTP {resp.status_code}: {url}")
    except requests.RequestException as e:
        print(f"  Error: {e}")
    return None


def is_relevant(soup: BeautifulSoup) -> bool:
    text = soup.get_text().lower()
    return sum(1 for kw in RELEVANCE_KEYWORDS if kw in text) >= 2


def collect_post_urls(index_urls: list[str]) -> list[str]:
    seen: set[str] = set(KNOWN_POSTS)
    urls: list[str] = list(KNOWN_POSTS)

    for index_url in index_urls:
        print(f"Indexing {index_url}")
        html = fetch(index_url)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(GPUOPEN_BASE, href)
            parsed = urlparse(full)
            # Only follow links within gpuopen.com that look like articles.
            if (
                parsed.netloc == "gpuopen.com"
                and "/learn/" in parsed.path
                and full not in seen
                and len(parsed.path.strip("/").split("/")) >= 2
            ):
                seen.add(full)
                urls.append(full)
                if len(urls) >= MAX_POSTS:
                    break
        time.sleep(DELAY)
        if len(urls) >= MAX_POSTS:
            break

    return urls


def clean_text(element: Tag) -> str:
    parts: list[str] = []
    for child in element.descendants:
        if not hasattr(child, "name"):
            text = str(child)
            if text.strip():
                parts.append(text.strip())
        elif child.name in ("code", "tt"):
            parts.append(f"`{child.get_text(strip=True)}`")
        elif child.name == "pre":
            code = child.get_text()
            lang = "cpp" if any(kw in code for kw in ["hip", "__global__", "kernel", "#include"]) else ""
            parts.append(f"\n```{lang}\n{code.strip()}\n```")
    result = " ".join(parts)
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def infer_category(heading: str, content: str) -> str:
    combined = (heading + " " + content).lower()
    if any(w in combined for w in ["port", "migrat", "equivalent", "cuda to hip", "hipify"]):
        return "api_translation"
    if any(w in combined for w in ["optim", "perform", "bandwidth", "occupancy", "latency", "throughput"]):
        return "optimization"
    if any(w in combined for w in ["debug", "error", "fail", "fix", "crash", "assert"]):
        return "debugging"
    return "concepts"


def heading_to_question(heading: str, category: str) -> str:
    h = heading.strip().rstrip(".")
    if h.endswith("?"):
        return h
    m = re.match(r"(?:How to|How To)\s+(.+)", h, re.IGNORECASE)
    if m:
        return f"How do I {m.group(1).lower()}?"
    m = re.match(r"(Porting|Migrating|Converting)\s+(.+)", h, re.IGNORECASE)
    if m:
        return f"How do I {m.group(1).lower()} {m.group(2).lower()}?"
    if category == "optimization":
        return f"How do I optimize {h.lower()}?"
    if category == "api_translation":
        return f"How does {h} work in HIP?"
    return f"What is {h}?"


def extract_pairs_from_post(html: str, url: str) -> Iterator[TrainingPair]:
    soup = BeautifulSoup(html, "lxml")

    if not is_relevant(soup):
        return

    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Article body — WordPress and similar CMSes use these containers.
    main = (
        soup.find("article")
        or soup.find("div", class_=re.compile(r"entry-content|post-content|article-body"))
        or soup.find("div", class_="content")
        or soup.find("main")
    )
    if not main:
        return

    current_heading: str | None = None
    current_parts: list[str] = []

    def flush() -> TrainingPair | None:
        if not current_heading or not current_parts:
            return None
        content = "\n\n".join(p for p in current_parts if p.strip())
        if len(content) < 100:
            return None
        category = infer_category(current_heading, content)
        question = heading_to_question(current_heading, category)
        return TrainingPair(
            instruction=question,
            input="",
            output=content + f"\n\nSource: {url}",
            source="gpuopen",
            category=category,  # type: ignore[arg-type]
            language="en",
            verified=False,
        )

    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "ul", "ol"]):
        if el.name in ("h1", "h2", "h3", "h4"):
            pair = flush()
            if pair:
                yield pair
            current_heading = el.get_text(strip=True)
            current_parts = []
        else:
            text = clean_text(el)
            if text:
                current_parts.append(text)

    pair = flush()
    if pair:
        yield pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape GPUOpen blog posts into training pairs")
    parser.add_argument("--output", default="data/raw/gpuopen_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    post_urls = collect_post_urls(INDEX_URLS)
    print(f"\nFound {len(post_urls)} post URLs to process")

    total = 0
    with output.open("w", encoding="utf-8") as f:
        for url in post_urls:
            print(f"Scraping {url}")
            html = fetch(url)
            if not html:
                continue

            page_count = 0
            for pair in extract_pairs_from_post(html, url):
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                page_count += 1
                total += 1
                if args.limit and total >= args.limit:
                    break

            if page_count:
                print(f"  → {page_count} pairs")
            if args.limit and total >= args.limit:
                break
            time.sleep(DELAY)

    print(f"\nTotal pairs written: {total}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
