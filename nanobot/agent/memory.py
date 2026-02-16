"""Memory system for persistent agent memory.

Combines local markdown files (MEMORY.md + HISTORY.md) with an optional
supermemory cloud layer for semantic search and recall.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from nanobot.utils.helpers import ensure_dir


class MemoryStore:
    """Two-layer memory with optional supermemory augmentation.

    Local layer:
        MEMORY.md  — curated long-term facts (always loaded into context).
        HISTORY.md — append-only grep-searchable event log.

    Cloud layer (supermemory):
        When configured, all writes are synced to supermemory and semantic
        search is available for context enrichment.  If not configured, the
        cloud layer is silently skipped.
    """

    def __init__(
        self,
        workspace: Path,
        supermemory_api_key: str | None = None,
        container_tag: str = "nanobot",
    ) -> None:
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

        self._sm_client = None
        self._container_tag = container_tag
        if supermemory_api_key:
            try:
                from supermemory import Supermemory
                self._sm_client = Supermemory(api_key=supermemory_api_key)
                logger.debug("Supermemory client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize supermemory: {e}")

    # ------------------------------------------------------------------
    # Local layer (unchanged behaviour)
    # ------------------------------------------------------------------

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")
        self._sm_add(content, metadata={"type": "long_term_memory"})

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")
        self._sm_add(entry, metadata={"type": "history"})

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def get_memory_context(self) -> str:
        """Build the memory section for the system prompt."""
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # ------------------------------------------------------------------
    # Supermemory — semantic search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Semantic search across all stored memories.

        Returns a list of content strings.  Returns ``[]`` if supermemory
        is not configured or the request fails.
        """
        if not self._sm_client:
            return []
        try:
            response = self._sm_client.search.documents(
                q=query,
                container_tags=[self._container_tag],
            )
            results = response.results or []
            contents = []
            for r in results[:limit]:
                # Try top-level content first, then extract from chunks
                if hasattr(r, "content") and r.content:
                    contents.append(r.content)
                elif hasattr(r, "chunks") and r.chunks:
                    for chunk in r.chunks:
                        if hasattr(chunk, "content") and chunk.content:
                            contents.append(chunk.content)
                            break
            return contents
        except Exception as e:
            logger.debug(f"Supermemory search failed: {e}")
            return []

    @property
    def has_supermemory(self) -> bool:
        """Whether the supermemory cloud layer is available."""
        return self._sm_client is not None

    # ------------------------------------------------------------------
    # Supermemory — cloud sync
    # ------------------------------------------------------------------

    def sync_to_cloud(self, content: str, memory_type: str = "long_term") -> None:
        """Sync content to supermemory without writing the local file.

        Use this when the local file was already written by another path
        (e.g. the write_file or edit_file tool) and only the cloud layer
        needs updating.
        """
        self._sm_add(content, metadata={"type": f"{memory_type}_memory"})

    # ------------------------------------------------------------------
    # Supermemory — internal helpers
    # ------------------------------------------------------------------

    def _sm_add(self, content: str, metadata: dict | None = None) -> None:
        """Add content to supermemory (fire-and-forget)."""
        if not self._sm_client:
            return
        try:
            self._sm_client.add(
                content=content,
                container_tags=[self._container_tag],
                metadata=metadata or {},
            )
        except Exception as e:
            logger.debug(f"Supermemory add failed: {e}")
