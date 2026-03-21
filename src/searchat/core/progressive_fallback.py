"""Progressive fallback for empty palace search results.

When facet-scoped palace search returns insufficient results (<3), progressively
relaxes project scoping to avoid returning nothing to the user.

Three-tier fallback strategy:
1. Scoped search (resolved projects only)
2. Related projects expansion (if <3 results)
3. Unscoped search (if still <3 results)

Logs which tier was used for analysis and improvement.
"""
import logging
from typing import List, Optional, Tuple
from enum import Enum

from searchat.models.domain import PalaceSearchResult
from searchat.palace.query import PalaceQuery

logger = logging.getLogger(__name__)


class FallbackTier(Enum):
    """Fallback tier used for search."""
    SCOPED = "scoped"  # Original scoped search
    RELATED = "related"  # Expanded to related projects
    UNSCOPED = "unscoped"  # Full unscoped search


class ProgressiveFallback:
    """Progressive fallback handler for palace searches."""

    def __init__(self, palace_query: PalaceQuery, min_results: int = 3):
        """Initialize progressive fallback.

        Args:
            palace_query: PalaceQuery instance for executing searches
            min_results: Minimum results before triggering fallback (default: 3)
        """
        self.palace_query = palace_query
        self.min_results = min_results
        self._fallback_stats = {
            FallbackTier.SCOPED: 0,
            FallbackTier.RELATED: 0,
            FallbackTier.UNSCOPED: 0,
        }

    def search_with_fallback(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        limit: int = 50,
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> Tuple[List[PalaceSearchResult], FallbackTier]:
        """Execute palace search with progressive fallback.

        Args:
            query: Search query string
            project_ids: Initial scoped project IDs (from facet resolution)
            limit: Maximum results to return
            keyword_weight: Weight for BM25 scores
            semantic_weight: Weight for FAISS scores

        Returns:
            Tuple of (results, tier_used)
        """
        # Tier 1: Scoped search (if project_ids provided)
        if project_ids:
            results = self.palace_query.search_hybrid(
                query,
                limit=limit,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                project_ids=project_ids,
            )

            if len(results) >= self.min_results:
                logger.info(
                    "Scoped search successful: %d results for projects %s",
                    len(results),
                    project_ids,
                )
                self._fallback_stats[FallbackTier.SCOPED] += 1
                return results, FallbackTier.SCOPED

            logger.info(
                "Scoped search insufficient: %d results (min %d) for projects %s",
                len(results),
                self.min_results,
                project_ids,
            )

            # Tier 2: Expand to related projects
            related_project_ids = self._find_related_projects(project_ids)
            if related_project_ids:
                expanded_project_ids = project_ids + related_project_ids
                expanded_results = self.palace_query.search_hybrid(
                    query,
                    limit=limit,
                    keyword_weight=keyword_weight,
                    semantic_weight=semantic_weight,
                    project_ids=expanded_project_ids,
                )

                if len(expanded_results) >= self.min_results:
                    logger.info(
                        "Related projects expansion successful: %d results (added %s)",
                        len(expanded_results),
                        related_project_ids,
                    )
                    self._fallback_stats[FallbackTier.RELATED] += 1
                    return expanded_results, FallbackTier.RELATED

                logger.info(
                    "Related projects expansion insufficient: %d results",
                    len(expanded_results),
                )

        # Tier 3: Unscoped search (last resort)
        unscoped_results = self.palace_query.search_hybrid(
            query,
            limit=limit,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
            project_ids=None,  # No scoping
        )

        if project_ids:
            logger.warning(
                "Fell back to unscoped search: %d results (facet resolution was: %s)",
                len(unscoped_results),
                project_ids,
            )
        else:
            logger.info(
                "Unscoped search (no facet resolution): %d results",
                len(unscoped_results),
            )

        self._fallback_stats[FallbackTier.UNSCOPED] += 1
        return unscoped_results, FallbackTier.UNSCOPED

    def _find_related_projects(self, project_ids: List[str]) -> List[str]:
        """Find related projects based on shared characteristics.

        Current implementation uses project metadata from palace storage:
        - Projects with similar room keys (shared concepts)
        - Projects in same language (heuristic: same file extension patterns)

        Args:
            project_ids: Original scoped project IDs

        Returns:
            List of related project IDs (excluding originals)
        """
        if not project_ids:
            return []

        # Get all rooms for the scoped projects
        try:
            # Query rooms for scoped projects
            placeholders = ", ".join(["?" for _ in project_ids])
            rows = self.palace_query.storage.conn.execute(f"""
                SELECT DISTINCT room_key, room_type
                FROM rooms
                WHERE project_id IN ({placeholders})
            """, project_ids).fetchall()

            if not rows:
                return []

            scoped_room_keys = {r[0] for r in rows}
            scoped_room_types = {r[1] for r in rows}

            # Find projects with overlapping room keys or types
            # (indicating shared concepts or similar domain)
            related_projects_rows = self.palace_query.storage.conn.execute("""
                SELECT DISTINCT project_id
                FROM rooms
                WHERE (room_key IN (SELECT UNNEST(?))
                       OR room_type IN (SELECT UNNEST(?)))
                  AND project_id NOT IN (SELECT UNNEST(?))
                ORDER BY project_id
                LIMIT 5
            """, [
                list(scoped_room_keys),
                list(scoped_room_types),
                project_ids,
            ]).fetchall()

            related = [r[0] for r in related_projects_rows]

            if related:
                logger.debug(
                    "Found %d related projects via shared rooms: %s",
                    len(related),
                    related,
                )

            return related

        except Exception as e:
            logger.error("Failed to find related projects: %s", e)
            return []

    def get_fallback_stats(self) -> dict:
        """Get statistics on fallback tier usage.

        Returns:
            Dict with tier usage counts and percentages
        """
        total = sum(self._fallback_stats.values())
        if total == 0:
            return {
                "total_searches": 0,
                "scoped_count": 0,
                "scoped_pct": 0.0,
                "related_count": 0,
                "related_pct": 0.0,
                "unscoped_count": 0,
                "unscoped_pct": 0.0,
            }

        return {
            "total_searches": total,
            "scoped_count": self._fallback_stats[FallbackTier.SCOPED],
            "scoped_pct": 100.0 * self._fallback_stats[FallbackTier.SCOPED] / total,
            "related_count": self._fallback_stats[FallbackTier.RELATED],
            "related_pct": 100.0 * self._fallback_stats[FallbackTier.RELATED] / total,
            "unscoped_count": self._fallback_stats[FallbackTier.UNSCOPED],
            "unscoped_pct": 100.0 * self._fallback_stats[FallbackTier.UNSCOPED] / total,
        }

    def reset_stats(self) -> None:
        """Reset fallback statistics."""
        for tier in FallbackTier:
            self._fallback_stats[tier] = 0
