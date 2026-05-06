"""Shared types, validation, and regex for RiftTune dataset pipeline."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Single source of truth for HIP API name extraction.
# Used in is_valid_pair AND evaluate.py — never re-implemented inline.
HIP_API_PATTERN = re.compile(
    r"hip[A-Z]\w+"            # core HIP runtime: hipMalloc, hipError_t, hipStream_t, etc.
    r"|hipblas[A-Za-z]\w*"    # hipBLAS: hipblasCreate, hipblasDestroy, etc.
    r"|hipfft[A-Za-z]\w*"     # hipFFT: hipfftCreate, etc.
    r"|hipcub::\w+"            # hipCUB (namespace-qualified)
    r"|HIP_[A-Z_]+"            # macros: HIP_CHECK, HIP_CALL, etc.
    r"|__syncthreads\b"        # HIP/CUDA built-in (identical in both)
    r"|__syncwarp\b"           # HIP wavefront sync built-in
)

_api_names_cache: set[str] | None = None
API_NAMES_PATH = Path("data/hip_api_names.json")


def load_api_names(path: Path = API_NAMES_PATH) -> set[str]:
    global _api_names_cache
    if _api_names_cache is None:
        if path.exists():
            _api_names_cache = set(json.loads(path.read_text(encoding="utf-8")))
        else:
            _api_names_cache = set()
    return _api_names_cache


def extract_api_names(text: str) -> set[str]:
    return set(HIP_API_PATTERN.findall(text))


@dataclass
class TrainingPair:
    instruction: str
    input: str                  # empty string if no code context
    output: str
    source: Literal["rocm_docs", "hipify", "gpuopen", "github"]
    category: Literal["api_translation", "debugging", "optimization", "concepts", "detection"]
    language: Literal["en", "th"]
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingPair":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def is_valid_pair(pair: TrainingPair, api_names: set[str] | None = None) -> bool:
    """Return True if the pair is suitable for training."""
    if len(pair.instruction.strip()) < 20:
        return False
    if len(pair.output.strip()) < 100:
        return False
    # Reject outputs likely to exceed 2048 tokens when formatted.
    # Code-heavy content tokenizes at ~1.5-2 chars/token; 4500 chars ≈ 2,250-3,000 tokens.
    if len(pair.output) > 4500:
        return False
    # Reject unconverted CUDA answers (output talks about CUDA without mentioning HIP/ROCm).
    output_lower = pair.output.lower()
    if (
        output_lower.count("cuda") > 3
        and "hip" not in output_lower
        and "rocm" not in output_lower
    ):
        return False
    # api_translation pairs must contain at least one valid HIP API name.
    if pair.category == "api_translation":
        if api_names is None:
            api_names = load_api_names()
        mentioned = extract_api_names(pair.output)
        if api_names and not mentioned.intersection(api_names):
            return False
    return True
