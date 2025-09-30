from pydantic import BaseModel, HttpUrl


class ImageTagModel(BaseModel):
    """Represents an image tag"""

    id: int
    image_uuid: str
    tag: str
    confidence: float

    def to_tuple(self) -> tuple[str, str, float]:
        return self.image_uuid, self.tag, self.confidence
    
    def __str__(self):
        return self.tag


class ImageMetadataModel(BaseModel):
    """Represents a full entry of image metadata"""

    id: str
    filename: str
    source_url: HttpUrl
    source_domain: HttpUrl
    file_size: int
    dimensions: str
    tags: list[ImageTagModel]

    def to_tuple(self) -> tuple[str, str, str, str, int, str]:
        return self.id, self.filename, str(self.source_url), str(self.source_domain), self.file_size, self.dimensions
