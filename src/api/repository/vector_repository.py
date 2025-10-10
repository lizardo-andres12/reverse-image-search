from typing import Sequence

from chromadb import QueryResult
from managers import ChromaConnectionManager
from models import QueryHit, VectorEntry
from pydantic import BaseModel, ValidationError


class VectorRepository:  # TODO: implement caching
    """
    Encapsulates ChromaDB access functionality

    The ChromaDB vector database has the following schema:

    VectorEntry {
        "id": uuid - The (unique) image UUIDv4 string. Used to find related metadata in with ImageRepository,
        "embedding": Sequence[float] - The 512 element normalized embedding produced by CLIP transformer model,
        "metadata": dict {
            "source_domain": str - Enables filtering by domain,
            "indexed_at": int (unix timestamp) - Enables filtering by oldest/newest added
        } - The filtering tags associated with the image
    }

    Retrieval from the database will always be done with `.query` as that returns the `distances` field, showing the
    similarity score of the result. The return object schema will be the following:

    QueryHit {
        "id": uuid,
        "metadata": dict {
            "source_domain": str,
            "indexed_at": int (unix timestamp)
        },
        "similarity": float - The cosine distance similarity
    }
    """

    # Constants
    EXPECTED_DIM = 512
    SOURCE_DOMAIN_KEY = "source_domain"
    INDEXED_AT_KEY = "indexed_at"
    IDS_KEY = "ids"
    METADATAS_KEY = "metadatas"
    DISTANCES_KEY = "distances"
    QUERY_INCLUDE = [METADATAS_KEY, DISTANCES_KEY]

    def __init__(self, conn: ChromaConnectionManager):
        self._collection = conn.collection

    def upsert(self, entry: VectorEntry) -> str:
        """
        Upserts a single VectorEntry to the database. This function MUST be wrapped in try/except block,
        catching only happens here for improved visiblity by adding the function where error was thrown.

        Args:
            entry (VectorEntry): The entry to store.
        Returns:
            str: The id of the add entry or empty string on failure.
        """
        self._collection.upsert(
            ids=entry.id, embeddings=entry.embedding, metadatas=entry.metadata
        )
        fetched_id = self._get_entries([entry.id])[0]
        return fetched_id if fetched_id else ""

    def batch_upsert(self, entries: Sequence[VectorEntry]) -> list[str]:
        """
        Upserts multiple VectorEntry models to the database. This function MUST be wrapped in try/except block,
        catching only happens here for improved visiblity by adding the function where error was thrown.

        Args:
            entry (Sequence[VectorEntry]): The entries to store.
        Returns:
            list[str]: A list of ids that were add and None for ids that failed to be added.
        """
        ids, embeddings, metadatas = self._split_entries(entries)
        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        return self._get_entries(ids)

    def query_similar(self, embedding: Sequence[float], limit: int) -> list[QueryHit]:
        """
        Finds the top limit many similar embeddings in the database and returns a list of the matches as QueryHit pydantic model objects.

        Args:
            embedding (Sequence[float]): The 512 dimenstion embedding vector to match.
            limit (int): The maximum number of results to return.
        Returns:
            list[QueryHit]: The top limit many similar vector's metadata.
        """
        results = self._collection.query(
            query_embeddings=embedding, n_results=limit, include=self.QUERY_INCLUDE
        )

        return self._parse_results_object(results)[0]

    def batch_query_similar(
        self, embeddings: list[Sequence[float]], limit: int
    ) -> list[list[QueryHit]]:
        """
        Batch queries the top limit many similar embeddings in the database for every id and returns a list of QueryHit lists and ValidationError lists,
        one for each id in the order they were input.

        Args:
            embeddings (list[Sequence[float]]): The embeddings to batch query the database with.
            limit (int): The maximum number of QueryHits per query.
        Returns:
            list[list[QueryHit]]: The collection of parsed results.
        """
        results = self._collection.query(
            query_embeddings=embeddings, n_results=limit, include=self.QUERY_INCLUDE
        )

        return self._parse_results_object(results)

    def _get_entries(self, ids: list[str]) -> list[str]:
        """
        Gets all entries with id in ids.

        Args:
            ids (list[str]): The ids to get from database.
        Returns:
            list[str]: An equally sized list of ids or None in the case that an entry with
                id does not exist in the database. The list is returned in the same order it was given.
        """
        fetched_ids = set(self._collection.get(ids=ids)[self.IDS_KEY])

        res = [None] * len(ids)
        for idx, id in enumerate(ids):
            if id in fetched_ids:
                res[idx] = id

        return res

    def _parse_results_object(self, results: QueryResult) -> list[list[QueryHit]]:
        """
        Parses *.query result objects (TypedDict objects) and returns list of results as QueryHit models and errors raised by
        parsing. This function performs result validation with `self._validate_query_results`.

        Args:
            results (QueryResult): The result object returned by *.query calls.
        Returns:
            list[list[QueryHit]]: The collection of parsed results.
        """
        num_queries = len(results[self.IDS_KEY])
        result_pairs: list[list[QueryHit]] = []

        for i in range(num_queries):
            num_results = len(results[self.IDS_KEY][i])
            query_hits: list[QueryHit] = [None] * num_results

            for idx, val in enumerate(
                zip(
                    results[self.IDS_KEY][i],
                    results[self.METADATAS_KEY][i],
                    results[self.DISTANCES_KEY][i],
                )
            ):
                id, metadata, similarity = val
                if not metadata:
                    metadata = {}

                hit = QueryHit(id=id, metadata=dict(metadata), similarity=similarity)
                query_hits[idx] = hit

            result_pairs.append(query_hits)
        return result_pairs

    def _split_entries(
        self, entries: Sequence[VectorEntry]
    ) -> tuple[list[str], list[Sequence[float]], list[dict[str, str]]]:
        """
        Splits all VectorEntry models into separate lists of uuids, embeddings, and metadatas. Since index ordering matters
        to upsert/add ChromaDB functions, this function MUST only be called with valid entries. Empty fields may cause order
        mismatch.

        Args:
            entries (Sequence[VectorEntry]): The entries to split.
        Returns:
            list[str]: The list of uuids.
            list[Sequence[float]]: The list of embeddings.
            list[dict[str, str]]: The list of metadatas.
        """
        ids: list[str] = []
        embeddings: list[Sequence[float]] = []
        metadatas: list[dict[str, str]] = []

        for entry in entries:
            ids.append(entry.id)
            embeddings.append(entry.embedding)
            metadatas.append(entry.metadata)
        return ids, embeddings, metadatas

    # TODO: Move this logic into field validation pydantic function
    def _validate_entry(self, entry: VectorEntry) -> None:
        """
        Validates the entry in provided and raises error on any invalid field. Field checks are done again
        as a safeguard if fields were modified from pydantic loading to now.

        Args:
            entries (VectorEntry): The entry to validate.
        Raises:
            ValueError: Any invalid fields in the entry.
        """
        if not entry:
            raise ValueError("Empty entry")
        if not entry.id:
            raise ValueError(f"Entry is missing id: {entry}")
        if not entry.embedding:
            raise ValueError(f"Entry is missing embedding: {entry}")
        if len(entry.embedding) != self.EXPECTED_DIM:
            raise ValueError(
                f"Entry embedding dimesions incorrect: {len(entry)} != {self.EXPECTED_DIM}"
            )
        if not entry.metadata:
            raise ValueError(f"Entry metadata missing: {entry}")
        if not entry.metadata.get(
            self.SOURCE_DOMAIN_KEY, None
        ) or not entry.metadata.get(self.INDEXED_AT_KEY, None):
            raise ValueError(
                f"Entry metadata missing one or more fields: {entry.metadata}"
            )

    # TODO: Move this logic into field validation pydantic function
    def _get_valid_entries(
        self, entries: Sequence[VectorEntry]
    ) -> tuple[list[VectorEntry], list[ValueError]]:
        """
        Returns a list of all entries that do not raise errors when called with `self._valid_entry`.

        Args:
            entries (Sequence[VectorEntry]): The sequence of entries to process
        Returns:
            tuple[list[VectorEntry], list[ValueError]]: All valid models in entries and all errors to be logged
                in calling function.
        """
        valid_entries: list[VectorEntry] = []
        errors: list[ValueError] = []
        for entry in entries:
            try:
                self._validate_entry(entry)
                valid_entries.append(entry)
            except ValueError as e:
                errors.append(e)
        return valid_entries, errors

    # TODO: Move this logic into field validation pydantic function
    def _validate_query_params(self, embedding: Sequence[float]) -> None:
        """
        Validates all parameters passed to `self.query_similar` and raises an error on invalid param.

        Args:
            embedding (Sequence[float]): The embedding to validate.
        Raises:
            ValueError: Any invalid fields in the entry.
        """
        if not embedding:
            raise ValueError("Empty embedding")
        if len(embedding) != self.EXPECTED_DIM:
            raise ValueError(
                f"Embedding dimesions incorrect: {len(embedding)} != {self.EXPECTED_DIM}"
            )

    def _get_valid_queries(
        self, embeddings: list[Sequence[float]]
    ) -> tuple[list[Sequence[float]], list[ValueError]]:
        """
        Returns a list of all queries that do not raise errors when called with `self._validate_query_params`.

        Args:
            embeddings (list[Sequence[float]]): The sequence of queries to process
        Returns:
            tuple[list[VectorEntry], list[ValueError]]: All valid models in entries and all errors to be logged
                in calling function.
        """
        valid_queries: list[Sequence[float]] = []
        errors: list[ValueError] = []
        for embedding in embeddings:
            try:
                self._validate_query_params(embedding)
            except ValueError as e:
                errors.append(e)
        return valid_queries, errors

    # TODO: Move to controller layer
    def _validate_query_results(self, results: QueryResult, limit: int) -> None:
        """
        Ensures all needed fields exits in results.

        Args:
            results (QueryResult): The results dict to validate.
            limit (int): The maximum length of the results.
        Raises:
            ValueError: Any missing field in results that should have been included.
        """
        for field in self.QUERY_INCLUDE:
            if field not in results:
                raise ValueError(f"Result missing field: {field}")
            if len(results[field]) > limit:
                raise ValueError(
                    f"Too many values returned: {len(results[field])} > {limit} (max)"
                )
