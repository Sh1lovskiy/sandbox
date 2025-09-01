"""Graph traversal helpers."""
from __future__ import annotations

from typing import List, Tuple


class EdgeNavigator:
    """Cycle through graph edges."""

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.edges = edges
        self.idx = 0

    def current(self) -> Tuple[int, int] | None:
        if not self.edges:
            return None
        return self.edges[self.idx]

    def next(self) -> Tuple[int, int] | None:
        if self.edges:
            self.idx = (self.idx + 1) % len(self.edges)
        return self.current()

    def prev(self) -> Tuple[int, int] | None:
        if self.edges:
            self.idx = (self.idx - 1) % len(self.edges)
        return self.current()
