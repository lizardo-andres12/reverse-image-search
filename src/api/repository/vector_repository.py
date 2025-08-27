from typing import Sequence

from connections import ChromaConnectionManager
from models import QueryHit, VectorEntry


class VectorRepository:
    """Encapsulates ChromaDB access functionality"""

    EXPECTED_DIM = 512

    def __init__(self, conn: ChromaConnectionManager):
        self._collection = conn.collection

    def upsert(self, entry: VectorEntry):
        if not entry:
            raise ValueError("Empty entry passed to VectorRepository::upsert")
        self._validate_entries([entry])
        self._collection.upsert(ids=entry.uuid, embeddings=entry.embedding)

    def batch_upsert(self, entries: Sequence[VectorEntry]):
        if not entries:
            raise ValueError("Empty sequence passed to VectorRepository::batch_upsert")
        self._validate_entries(entries)
        ids, embeddings = self._compact_entries(entries)
        self._collection.upsert(ids=ids, embeddings=embeddings)

    def query_similar(
        self, embedding: Sequence[float], limit: int
    ) -> Sequence[QueryHit]:
        pass

    def _compact_entries(
        self, entries: Sequence[VectorEntry]
    ) -> tuple[Sequence[str], Sequence[Sequence[float]]]:
        ids, embeddings = [], []
        for entry in entries:
            ids.append(entry.uuid)
            embeddings.append(entry.embedding)
        return ids, embeddings

    def _validate_entries(self, entries: Sequence[VectorEntry]):
        for entry in entries:
            if not entry:
                raise ValueError("Empty entry")
            if not entry.uuid or not entry.embedding:
                raise ValueError("entry is missing fields")
