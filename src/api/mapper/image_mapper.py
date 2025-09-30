from asyncpg import Record
from pydantic import HttpUrl
from models import ImageMetadataModel, ImageTagModel


def image_metadata_db_to_model(metadata: Record) -> ImageMetadataModel:
    """Takes raw database-retrieved objects and converts it into an
    ImageMetadataModel object.
    
    Args:
        metadata (Record): The column values retrieved from the images and image_tags tables in the database.
    Returns:
        ImageMetadataModel: The data collection object.
    """
    tags = metadata.get('tags', None) # TODO: retrieve all image_tags rows from db
    tags = tags.split(',') if tags else list()
    tags = [ImageTagModel(id=0, image_uuid='', tag=tag, confidence=0) for tag in tags]

    model = ImageMetadataModel(
        id=str(metadata.get('uuid', '')),
        filename=metadata.get('filename', ''),
        source_url=HttpUrl(metadata.get('source_url', '')),
        source_domain=HttpUrl(metadata.get('source_domain', '')),
        file_size=metadata.get('file_size', 0),
        dimensions=metadata.get('dimensions', ''),
        tags=tags
    )
    return model
