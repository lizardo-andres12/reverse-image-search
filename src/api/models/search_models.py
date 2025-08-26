from pydantic import BaseModel, HttpUrl


class SimilarImage(BaseModel):
    """Response object for image that matches vector embeddings of input image"""

    id: str
    similarity: float
    thumbnail_url: HttpUrl | None = None
    source_url: HttpUrl | None = None
    source_domain: str | None = None
    filename: str | None = None
    file_size: int | None = None
    dimensions: str | None = None
    tags: list[str] = list()


class SearchResponse(BaseModel):
    """Response object for reverse-image search"""

    keywords: list[str]
    similar_images: list[SimilarImage]
    total_found: int
