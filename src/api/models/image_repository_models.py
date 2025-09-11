from pydantic import BaseModel, HttpUrl


class ImageTagModel(BaseModel):
    """Represents an image tag"""

    id: int
    image_uuid: str
    tag: str
    confidence: float

    def to_tuple(self) -> tuple[int, str, str, float]:
        return self.id, self.image_uuid, self.tag, self.confidence


class ImageMetadataModel(BaseModel):
    """Represents a full entry of image metadata"""

    id: str
    filename: str
    source_url: HttpUrl
    source_domain: HttpUrl
    file_size: int
    dimensions: str
    tags: list[str]

    def to_tuple(self) -> tuple[str, str, HttpUrl, HttpUrl, int, str]:
        return self.id, self.filename, self.source_url, self.source_domain, self.file_size, self.dimensions
