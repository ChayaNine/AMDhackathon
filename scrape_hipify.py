"""Scrape HIPIFY API translation tables into instruction-output training pairs.

Each row in a HIPIFY table maps a CUDA function to its HIP equivalent.
We generate context-rich pairs (not just name lookups) by including:
  1. The CUDA → HIP mapping
  2. The HIP function signature (if available)
  3. A usage note from the HIPIFY porting guide

Target: ~250 pairs, category=api_translation.

Usage:
    python scrape_hipify.py [--output data/raw/hipify_pairs.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from dataset import TrainingPair

HEADERS = {"User-Agent": "RiftTune-scraper/1.0"}
DELAY = 0.5

# HIPIFY table pages — each covers a different CUDA API surface.
TABLE_PAGES = [
    "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Runtime_API_functions_supported_by_HIP.html",
    "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html",
    "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_cuBLAS_API_functions_supported_by_HIP.html",
    "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Math_API_functions_supported_by_HIP.html",
    "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_cuRAND_API_functions_supported_by_HIP.html",
    "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_cuFFT_API_functions_supported_by_HIP.html",
]

# HIP function signatures for commonly-used functions. Used to enrich pairs
# when the table doesn't include the signature. Extend as needed.
KNOWN_SIGNATURES: dict[str, str] = {
    "hipMalloc": "hipError_t hipMalloc(void** ptr, size_t size)",
    "hipFree": "hipError_t hipFree(void* ptr)",
    "hipMemcpy": "hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)",
    "hipMemcpyAsync": "hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)",
    "hipMemset": "hipError_t hipMemset(void* dst, int value, size_t sizeBytes)",
    "hipMemsetAsync": "hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream)",
    "hipDeviceSynchronize": "hipError_t hipDeviceSynchronize()",
    "hipStreamCreate": "hipError_t hipStreamCreate(hipStream_t* stream)",
    "hipStreamDestroy": "hipError_t hipStreamDestroy(hipStream_t stream)",
    "hipStreamSynchronize": "hipError_t hipStreamSynchronize(hipStream_t stream)",
    "hipEventCreate": "hipError_t hipEventCreate(hipEvent_t* event)",
    "hipEventDestroy": "hipError_t hipEventDestroy(hipEvent_t event)",
    "hipEventRecord": "hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)",
    "hipEventSynchronize": "hipError_t hipEventSynchronize(hipEvent_t event)",
    "hipEventElapsedTime": "hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)",
    "hipGetDeviceCount": "hipError_t hipGetDeviceCount(int* count)",
    "hipSetDevice": "hipError_t hipSetDevice(int deviceId)",
    "hipGetDevice": "hipError_t hipGetDevice(int* deviceId)",
    "hipGetDeviceProperties": "hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId)",
    "hipLaunchKernelGGL": "void hipLaunchKernelGGL(F kernel, dim3 numBlocks, dim3 dimBlocks, uint32_t sharedMemBytes, hipStream_t stream, Args... args)",
}

# Common memory kind mappings (CUDA → HIP).
MEMCPY_KINDS = {
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMemcpyHostToHost": "hipMemcpyHostToHost",
    "cudaMemcpyDefault": "hipMemcpyDefault",
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


def infer_category_from_url(url: str) -> str:
    url_lower = url.lower()
    if "blas" in url_lower or "fft" in url_lower or "rand" in url_lower:
        return "api_translation"
    return "api_translation"


def build_pair(cuda_func: str, hip_func: str, source_url: str, table_name: str) -> TrainingPair | None:
    """Build a context-rich training pair from a CUDA→HIP mapping."""
    if not cuda_func or not hip_func:
        return None
    if hip_func.lower() in ("not supported", "deprecated", "n/a", "", "-"):
        return None
    # Skip if HIP name looks unchanged from CUDA (some entries are identical).
    # We still include these as they confirm the API works in HIP.

    sig = KNOWN_SIGNATURES.get(hip_func, "")
    sig_line = f"\n\nSignature:\n```cpp\n{sig}\n```" if sig else ""

    # Build a usage note for well-known functions.
    usage = _usage_note(cuda_func, hip_func)
    usage_line = f"\n\n{usage}" if usage else ""

    output = (
        f"The HIP equivalent of `{cuda_func}` is `{hip_func}`."
        f"{sig_line}"
        f"{usage_line}"
        f"\n\nSource: {source_url}"
    )

    return TrainingPair(
        instruction=f"What is the HIP equivalent of `{cuda_func}`?",
        input="",
        output=output,
        source="hipify",
        category="api_translation",
        language="en",
        verified=False,
    )


def _usage_note(cuda: str, hip: str) -> str:
    """Return a short usage note for common functions."""
    notes = {
        "hipMalloc": (
            "Use `hipMalloc` the same way as `cudaMalloc` — it allocates device memory. "
            "Remember to check the return value for `hipSuccess`."
        ),
        "hipMemcpy": (
            "The `kind` parameter uses `hipMemcpyKind` values (e.g. `hipMemcpyHostToDevice`). "
            "These map directly from their `cudaMemcpy*` equivalents."
        ),
        "hipLaunchKernelGGL": (
            "`hipLaunchKernelGGL` is the HIP triple-chevron equivalent. "
            "Alternatively, HIP supports the `<<<>>>` syntax directly in `.hip` files. "
            "For portability, prefer `hipLaunchKernelGGL`."
        ),
        "hipDeviceSynchronize": (
            "Blocks the host until all previously queued work on the device completes. "
            "Use `hipStreamSynchronize(stream)` to synchronize a specific stream instead."
        ),
    }
    return notes.get(hip, "")


def parse_table_page(html: str, url: str) -> Iterator[TrainingPair]:
    soup = BeautifulSoup(html, "lxml")

    # Get table name from page title for context.
    title = soup.find("title")
    table_name = title.get_text(strip=True) if title else url

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        # Detect header row to find column positions.
        header = rows[0]
        headers = [th.get_text(strip=True).lower() for th in header.find_all(["th", "td"])]

        cuda_col = next((i for i, h in enumerate(headers) if "cuda" in h), 0)
        hip_col = next((i for i, h in enumerate(headers) if "hip" in h), 1)

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= max(cuda_col, hip_col):
                continue

            cuda_func = cells[cuda_col].get_text(strip=True)
            hip_func = cells[hip_col].get_text(strip=True)

            # Strip footnote markers (e.g. "hipMalloc [1]").
            cuda_func = re.sub(r"\s*\[.*?\]", "", cuda_func).strip()
            hip_func = re.sub(r"\s*\[.*?\]", "", hip_func).strip()

            # Skip rows that are not API names (headers, section labels).
            if not re.match(r"^(cuda|hip)[A-Za-z_]", cuda_func, re.IGNORECASE):
                continue

            pair = build_pair(cuda_func, hip_func, url, table_name)
            if pair:
                yield pair


def generate_memcpy_kind_pairs() -> Iterator[TrainingPair]:
    """Generate pairs for cudaMemcpyKind → hipMemcpyKind mappings."""
    url = "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Runtime_API_functions_supported_by_HIP.html"
    for cuda_kind, hip_kind in MEMCPY_KINDS.items():
        yield TrainingPair(
            instruction=f"What is the HIP equivalent of the `{cuda_kind}` memory kind?",
            input="",
            output=(
                f"The HIP equivalent of `{cuda_kind}` is `{hip_kind}`. "
                f"Memory kind constants map directly — the names change from `cudaMemcpy*` to `hipMemcpy*`.\n\n"
                f"Example:\n```cpp\n"
                f"// CUDA\ncudaMemcpy(dst, src, size, {cuda_kind});\n\n"
                f"// HIP\nhipMemcpy(dst, src, size, {hip_kind});\n```\n\n"
                f"Source: {url}"
            ),
            source="hipify",
            category="api_translation",
            language="en",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape HIPIFY tables into training pairs")
    parser.add_argument("--output", default="data/raw/hipify_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    seen: set[str] = set()  # deduplicate by CUDA function name within this scraper

    with output.open("w", encoding="utf-8") as f:
        # Emit memcpy kind pairs first (always valid, no network needed).
        for pair in generate_memcpy_kind_pairs():
            key = pair.instruction
            if key not in seen:
                seen.add(key)
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                total += 1

        for url in TABLE_PAGES:
            print(f"Fetching {url}")
            html = fetch(url)
            if not html:
                continue

            page_count = 0
            for pair in parse_table_page(html, url):
                key = pair.instruction
                if key in seen:
                    continue
                seen.add(key)
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
