"""Enumerations for searchat."""
from enum import Enum


class SearchMode(Enum):
    """Search modes for the 6-mode factorial design.

    Verbatim baseline (full text):
    - verbatim_bm25: BM25 keyword search over raw conversation text
    - verbatim_semantic: Vector similarity search over raw conversation text

    Distilled ablation (compressed, 50× smaller):
    - distill_core: Core distillation only (exchange_core + specific_context)
    - distill_core_files: Core + file paths
    - distill_core_rooms: Core + room metadata
    - distill_all_facets: Core + files + rooms (full palace object)
    """
    VERBATIM_BM25 = "verbatim_bm25"
    VERBATIM_SEMANTIC = "verbatim_semantic"
    DISTILL_CORE = "distill_core"
    DISTILL_CORE_FILES = "distill_core_files"
    DISTILL_CORE_ROOMS = "distill_core_rooms"
    DISTILL_ALL_FACETS = "distill_all_facets"


class AlgorithmType(Enum):
    """Internal search algorithm types (not exposed in public API)."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    CROSS_LAYER = "cross_layer"
    DISTILL = "distill"
