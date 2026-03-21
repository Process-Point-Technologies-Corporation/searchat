"""Hierarchical facet extraction and weighted distinctiveness calculation.

Improvement #2 from Future Improvements: Extract multiple levels of file facets
to avoid false aggregation across projects.

Current: Only basename (storage.py) causing false aggregation.
Better: Full path (palace/storage.py, 3x), directory (palace/*, 2x), basename (storage.py, 1x).
"""
import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from searchat.models.domain import DistilledObject, Room

logger = logging.getLogger(__name__)


def make_hierarchical_facet_id(facet_type: str, facet_text: str, level: str) -> str:
    """Deterministic facet ID from type + text + level.

    Args:
        facet_type: 'file', 'room', or 'project'
        facet_text: The facet text content
        level: 'full', 'directory', or 'basename' (for files)
    """
    key = f"{facet_type}:{level}:{facet_text}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class HierarchicalFacetExtractor:
    """Extracts hierarchical file facets with weighted distinctiveness."""

    # Weights for different facet levels
    FULL_PATH_WEIGHT = 3.0
    DIRECTORY_WEIGHT = 2.0
    BASENAME_WEIGHT = 1.0

    def extract_file_facets(
        self,
        file_path: str,
        project_id: str,
    ) -> List[Dict]:
        """Extract 3 levels of facets from a file path.

        Args:
            file_path: File path like "palace/storage.py" or "src/api/routers/search.py"
            project_id: Project this file belongs to

        Returns:
            List of facet dicts, each with:
                - facet_type: 'file'
                - facet_level: 'full', 'directory', or 'basename'
                - facet_text: The text to embed
                - weight: Relative weight for this level
                - project_ids: Set with single project_id

        Example:
            Input: "palace/storage.py", project_id="searchat"
            Output: [
                {facet_type: 'file', facet_level: 'full', facet_text: 'palace/storage.py',
                 weight: 3.0, project_ids: {'searchat'}},
                {facet_type: 'file', facet_level: 'directory', facet_text: 'palace/*',
                 weight: 2.0, project_ids: {'searchat'}},
                {facet_type: 'file', facet_level: 'basename', facet_text: 'storage.py',
                 weight: 1.0, project_ids: {'searchat'}},
            ]
        """
        p = Path(file_path)
        basename = p.name
        parent = p.parent

        facets = []

        # Level 3: Full path (most specific, highest weight)
        if len(file_path) > 3:
            facets.append({
                "facet_type": "file",
                "facet_level": "full",
                "facet_text": file_path.lower(),
                "weight": self.FULL_PATH_WEIGHT,
                "project_ids": {project_id},
            })

        # Level 2: Directory pattern (medium specificity)
        if str(parent) != "." and len(str(parent)) > 0:
            dir_pattern = f"{parent}/*"
            facets.append({
                "facet_type": "file",
                "facet_level": "directory",
                "facet_text": dir_pattern.lower(),
                "weight": self.DIRECTORY_WEIGHT,
                "project_ids": {project_id},
            })

        # Level 1: Basename only (least specific, lowest weight)
        if len(basename) > 3:
            facets.append({
                "facet_type": "file",
                "facet_level": "basename",
                "facet_text": basename.lower(),
                "weight": self.BASENAME_WEIGHT,
                "project_ids": {project_id},
            })

        return facets

    def extract_from_objects(
        self,
        objects: List[DistilledObject],
        rooms: List[Room],
    ) -> Dict[str, Dict]:
        """Extract all hierarchical facets from a batch of distilled objects.

        Returns:
            Dict mapping facet_key (type:level:text) to facet data with aggregated
            project_ids and weighted counts.
        """
        facet_map: Dict[str, Dict] = {}

        # Extract file facets hierarchically
        for obj in objects:
            for ft in obj.files_touched:
                file_facets = self.extract_file_facets(ft.path, obj.project_id)

                for facet in file_facets:
                    key = f"{facet['facet_type']}:{facet['facet_level']}:{facet['facet_text']}"

                    if key not in facet_map:
                        facet_map[key] = {
                            "facet_type": facet["facet_type"],
                            "facet_level": facet["facet_level"],
                            "facet_text": facet["facet_text"],
                            "weight": facet["weight"],
                            "project_ids": set(),
                            "weighted_count": 0.0,
                        }

                    facet_map[key]["project_ids"].add(obj.project_id)

        # Extract room facets (no hierarchy, single level)
        for room in rooms:
            if room.room_key and len(room.room_key) > 3:
                key = f"room:single:{room.room_key.lower()}"

                if key not in facet_map:
                    facet_map[key] = {
                        "facet_type": "room",
                        "facet_level": "single",
                        "facet_text": room.room_key.lower(),
                        "weight": 1.0,
                        "project_ids": set(),
                        "weighted_count": 0.0,
                    }

                if room.project_id:
                    facet_map[key]["project_ids"].add(room.project_id)

        # Extract project fragments (no hierarchy, single level)
        stop_tokens = {
            "projects", "home", "tmp", "mnt", "users", "data",
            "subtask", "workspaces", "benchmark", "var", "opt",
        }
        seen_pids: Set[str] = set()

        for obj in objects:
            if obj.project_id not in seen_pids:
                seen_pids.add(obj.project_id)
                tokens = re.split(r"[-_]+", obj.project_id)
                for token in tokens:
                    t = token.lower()
                    if len(t) > 2 and t not in stop_tokens:
                        key = f"project:single:{t}"

                        if key not in facet_map:
                            facet_map[key] = {
                                "facet_type": "project",
                                "facet_level": "single",
                                "facet_text": t,
                                "weight": 1.0,
                                "project_ids": set(),
                                "weighted_count": 0.0,
                            }

                        facet_map[key]["project_ids"].add(obj.project_id)

        # Calculate weighted distinctiveness
        for key, facet in facet_map.items():
            project_count = len(facet["project_ids"])
            weight = facet["weight"]

            # Weighted count: fewer projects + higher weight = more distinctive
            # Formula: weight / (1 + project_count)
            # Full path in 1 project: 3.0 / (1+1) = 1.5
            # Basename in 5 projects: 1.0 / (1+5) = 0.167
            facet["weighted_count"] = weight / (1.0 + project_count)

        return facet_map

    def compute_facet_embeddings(
        self,
        facet_map: Dict[str, Dict],
        embedder,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Compute embeddings for all facets.

        Args:
            facet_map: Dict from extract_from_objects()
            embedder: SentenceTransformer instance
            batch_size: Batch size for encoding

        Returns:
            List of facet dicts ready for storage, each with:
                - facet_id
                - facet_type
                - facet_level
                - facet_text
                - weight
                - weighted_count
                - project_ids (sorted list)
                - embedding (np.ndarray)
        """
        if not facet_map:
            return []

        facets = list(facet_map.values())
        texts = [f["facet_text"] for f in facets]

        embeddings = embedder.encode(texts, batch_size=batch_size)
        embeddings = np.array(embeddings, dtype=np.float32)

        result = []
        for i, facet in enumerate(facets):
            facet_id = make_hierarchical_facet_id(
                facet["facet_type"],
                facet["facet_text"],
                facet["facet_level"],
            )

            result.append({
                "facet_id": facet_id,
                "facet_type": facet["facet_type"],
                "facet_level": facet["facet_level"],
                "facet_text": facet["facet_text"],
                "weight": facet["weight"],
                "weighted_count": facet["weighted_count"],
                "project_ids": sorted(facet["project_ids"]),
                "embedding": embeddings[i],
            })

        logger.info(
            "Computed hierarchical facet embeddings: %d total (%d full, %d dir, %d base, %d room, %d project)",
            len(result),
            sum(1 for f in result if f["facet_level"] == "full"),
            sum(1 for f in result if f["facet_level"] == "directory"),
            sum(1 for f in result if f["facet_level"] == "basename"),
            sum(1 for f in result if f["facet_type"] == "room"),
            sum(1 for f in result if f["facet_type"] == "project"),
        )

        return result


class HierarchicalFacetResolver:
    """Resolves queries to projects using hierarchical facets with weighted voting."""

    def __init__(self, storage, embedder):
        """Initialize resolver.

        Args:
            storage: UnifiedStorage instance with facet search capability
            embedder: SentenceTransformer instance
        """
        self.storage = storage
        self.embedder = embedder

    def resolve(
        self,
        query: str,
        top_k: int = 10,
        confidence_threshold: float = 0.6,
        max_project_count: int = 3,
    ) -> Tuple[Optional[List[str]], Dict]:
        """Resolve query to project_ids using hierarchical weighted voting.

        Args:
            query: Search query
            top_k: Number of top facets to consider
            confidence_threshold: Minimum vote share for winner (0.6 = 60%)
            max_project_count: Max projects per facet for distinctiveness filter

        Returns:
            Tuple of (resolved_project_ids, metadata_dict).
            metadata_dict contains:
                - facets: List of matched facets with scores and levels
                - votes: Dict mapping project_id to vote totals
                - winner: Winning project_id (if resolved)
                - confidence: Confidence score (0-1)
                - reason: Human-readable explanation
        """
        if not self.storage._vss_available:
            return None, {"reason": "VSS not available"}

        # Embed query
        query_emb = self.embedder.encode(query)
        query_emb = np.array(query_emb, dtype=np.float32)

        # Search facet embeddings (hierarchical table)
        results = self.storage.search_hierarchical_facets(
            query_emb,
            limit=top_k,
            max_project_count=max_project_count,
        )

        if not results:
            return None, {"reason": "No distinctive facets found"}

        # Accumulate weighted votes per project
        from collections import defaultdict
        votes: Dict[str, float] = defaultdict(float)
        facet_details = []

        for facet in results:
            facet_text = facet["facet_text"]
            facet_type = facet["facet_type"]
            facet_level = facet["facet_level"]
            similarity = facet["score"]
            weight = facet.get("weight", 1.0)
            weighted_count = facet.get("weighted_count", 1.0)
            project_ids = facet["project_ids"]

            # Combined score: similarity * weight * distinctiveness
            combined_score = similarity * weight * weighted_count

            facet_details.append({
                "text": facet_text,
                "type": facet_type,
                "level": facet_level,
                "similarity": similarity,
                "weight": weight,
                "weighted_count": weighted_count,
                "combined_score": combined_score,
                "projects": project_ids,
            })

            # Distribute votes
            for pid in project_ids:
                votes[pid] += combined_score

        if not votes:
            return None, {
                "reason": "No project votes accumulated",
                "facets": facet_details,
            }

        # Find winner
        total_votes = sum(votes.values())
        winner_pid, winner_votes = max(votes.items(), key=lambda x: x[1])
        confidence = winner_votes / total_votes if total_votes > 0 else 0.0

        metadata = {
            "facets": facet_details,
            "votes": dict(votes),
            "total_votes": total_votes,
            "confidence": confidence,
        }

        if confidence >= confidence_threshold:
            metadata["winner"] = winner_pid
            metadata["reason"] = (
                f"Resolved to {winner_pid} with {confidence:.1%} confidence "
                f"({winner_votes:.2f}/{total_votes:.2f} weighted votes)"
            )

            logger.info(
                "Hierarchical facet resolution: '%s' → %s (conf=%.1f%%, votes=%s)",
                query[:100],
                winner_pid,
                confidence * 100,
                ", ".join(f"{pid}={v:.2f}" for pid, v in
                         sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]),
            )

            return [winner_pid], metadata
        else:
            metadata["reason"] = (
                f"Low confidence {confidence:.1%} < {confidence_threshold:.1%} threshold "
                f"(top vote: {winner_pid}={winner_votes:.2f}/{total_votes:.2f})"
            )

            logger.info(
                "Hierarchical facet resolution: '%s' rejected (conf=%.1f%% < %.1f%%)",
                query[:100],
                confidence * 100,
                confidence_threshold * 100,
            )

            return None, metadata
