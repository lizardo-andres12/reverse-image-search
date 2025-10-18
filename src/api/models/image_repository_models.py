from pydantic import BaseModel, HttpUrl


class ImageMetadataModel(BaseModel):
    """Represents a full entry of image metadata"""

    id: str
    filename: str
    source_url: HttpUrl
    source_domain: HttpUrl
    file_size: int
    dimensions: str

    def to_tuple(self) -> tuple[str, str, str, str, int, str]:
        return (
            self.id,
            self.filename,
            str(self.source_url),
            str(self.source_domain),
            self.file_size,
            self.dimensions,
        )
