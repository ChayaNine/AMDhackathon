"""Build data/hip_api_names.json from the ROCm Doxygen API reference.

Scrapes the HIP API function list, type list, and macro list from the
Doxygen-generated HTML at rocm.docs.amd.com. Saves a flat JSON list of
valid HIP API name strings used as the hallucination-detection allowlist.

Usage:
    python build_api_allowlist.py [--output data/hip_api_names.json]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html"

# Doxygen global index pages. Each may be paginated by first letter (e.g.
# globals_func_0x61.html for 'a'). We fetch all and deduplicate.
INDEX_PAGES = [
    f"{BASE}/globals_func.html",
    f"{BASE}/globals_type.html",
    f"{BASE}/globals.html",
]

# Alphabet pages that Doxygen generates when the list is long.
LETTER_CODES = [f"0x{ord(c):02x}" for c in "abcdefghijklmnopqrstuvwxyz"]

# Additional names that are HIP kernel built-ins not always in the Doxygen index.
BUILTIN_NAMES = [
    "__syncthreads",
    "__syncwarp",
    "__threadfence",
    "__threadfence_block",
    "__threadfence_system",
    "__activemask",
    "__ballot",
    "__popc",
    "__clz",
    "__ffs",
    "__brev",
]

# Patterns for name extraction from anchor tags in Doxygen HTML.
HIP_NAME_RE = re.compile(
    r"^(hip[A-Z]\w+|hipblas[A-Za-z]\w*|hipfft[A-Za-z]\w*|HIP_[A-Z_]+|hip[a-z]+[A-Z]\w*)"
)


def fetch(url: str, retries: int = 3, delay: float = 1.0) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "RiftTune-scraper/1.0"})
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
        except requests.RequestException as e:
            print(f"  Fetch error {url}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def extract_names_from_html(html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    names: set[str] = set()

    # Doxygen global index: function/type names are in <a> tags within <td class="entry">
    for td in soup.find_all("td", class_="entry"):
        a = td.find("a")
        if a and a.text:
            name = a.text.strip().rstrip("()")
            if HIP_NAME_RE.match(name):
                names.add(name)

    # Also scan all anchor text broadly — handles different Doxygen versions.
    for a in soup.find_all("a"):
        text = a.get_text(strip=True).rstrip("()")
        if HIP_NAME_RE.match(text):
            names.add(text)

    return names


def scrape_allowlist() -> set[str]:
    names: set[str] = set()

    # Try direct index pages first.
    for url in INDEX_PAGES:
        print(f"Fetching {url}")
        html = fetch(url)
        if html:
            found = extract_names_from_html(html)
            print(f"  Found {len(found)} names")
            names.update(found)
        time.sleep(0.3)

    # Try alphabet-paginated sub-pages (Doxygen splits large indexes).
    for suffix in ["globals_func", "globals_type", "globals"]:
        for code in LETTER_CODES:
            url = f"{BASE}/{suffix}_{code}.html"
            html = fetch(url)
            if html:
                found = extract_names_from_html(html)
                if found:
                    print(f"  {url}: +{len(found)}")
                    names.update(found)
            time.sleep(0.15)

    names.update(BUILTIN_NAMES)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HIP API name allowlist")
    parser.add_argument("--output", default="data/hip_api_names.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Scraping ROCm Doxygen API reference...")
    names = scrape_allowlist()

    # Filter: keep names that look like actual HIP identifiers.
    filtered = sorted(n for n in names if len(n) >= 4)
    print(f"\nTotal valid names: {len(filtered)}")

    if len(filtered) < 500:
        print("WARNING: fewer than 500 names found — scrape may have failed.")
        print("Check that rocm.docs.amd.com is reachable and the URL structure is current.")

    output.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
