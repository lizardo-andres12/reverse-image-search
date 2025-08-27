from pydantic import BaseModel


class VectorEntry(BaseModel):
    """Models one entry into the vector database"""

    uuid: str
    embedding: tuple[float]


class QueryHit(BaseModel):
    """Models a query hit"""

    uuid: str
    similarity: float
