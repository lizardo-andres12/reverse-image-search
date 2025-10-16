from pydantic import BaseModel, HttpUrl


class SimilarImage(BaseModel):
    """Response object for image that matches vector embeddings of input image"""

    id: str
    similarity: float
    source_url: HttpUrl
    source_domain: HttpUrl
    filename: str
    file_size: int
    dimensions: str
    tags: list[str]
