from pydantic import BaseModel


class VectorEntry(BaseModel):
    """Models one entry into the vector database"""

    id: str
    embedding: list[float]
    metadata: dict[str, str]


class QueryHit(BaseModel):
    """Models a query hit"""

    id: str
    metadata: dict[str, str]
    similarity: float
