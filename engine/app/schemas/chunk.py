from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RagChunk:
    id: str
    doc_id: str
    source_pdf: str
    content: str
    content_type: str
    page_numbers: list[int]
    section_path: list[str] = field(default_factory=list)
    role: str = "body"
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "doc_id": self.doc_id,
            "source_pdf": self.source_pdf,
            "content": self.content,
            "content_type": self.content_type,
            "page_numbers": self.page_numbers,
            "section_path": self.section_path,
            "section": " > ".join(self.section_path),
            "role": self.role,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding
        return data
