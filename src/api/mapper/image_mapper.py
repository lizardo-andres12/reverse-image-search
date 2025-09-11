from asyncpg import Record
from models import ImageMetadataModel


def image_metdata_db_to_model(metadata: Record) -> ImageMetadataModel:
    """Takes raw database-retrieved objects and converts it into an
    ImageMetadataModel object.
    
    Args:
        metadata (Record): The column values retrieved from the images and image_tags tables in the database.
    Returns:
        ImageMetadataModel: The data collection object.
    """
    tags = metadata.get('tags', None)
    tags = tags.split(',') if tags else None

    model = ImageMetadataModel(
        id=metadata.get('uuid', None),
        filename=metadata.get('filename', None),
        source_url=metadata.get('source_url', None),
        source_domain=metadata.get('source_domain', None),
        file_size=metadata.get('file_size', None),
        dimensions=metadata.get('dimensions', None),
        tags=tags
    )
    return model
