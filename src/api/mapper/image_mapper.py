from asyncpg import Record
from models import ImageMetadataModel
from pydantic import HttpUrl


def image_metadata_db_to_model(metadata: Record) -> ImageMetadataModel:
    """Takes raw database-retrieved objects and converts it into an
    ImageMetadataModel object.

    Args:
        metadata (Record): The column values retrieved from the images and image_tags tables in the database.
    Returns:
        ImageMetadataModel: The data collection object.
    Throws:
        ValueError: If the input is None
        ValidationError: If any field is not found in metadata or if the type is mismatched
    """
    if metadata is None:
        raise ValueError("cannot map None to model")

    model = ImageMetadataModel(
        id=str(metadata.get("uuid", None)),
        filename=metadata.get("filename", None),
        source_url=HttpUrl(metadata.get("source_url", None)),
        source_domain=HttpUrl(metadata.get("source_domain", None)),
        file_size=metadata.get("file_size", None),
        dimensions=metadata.get("dimensions", None),
    )
    return model
