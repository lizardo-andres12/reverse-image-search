from io import BytesIO

from asyncpg import Record
from fastapi import UploadFile
from models import ImageMetadataModel
from PIL import Image
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

async def async_upload_file_to_pil_image(file: UploadFile) -> Image.Image | None:
    """Converts the uploaded file to PIL Image.

    Args:
        file (UploadFile): The file from HTTP Request.
    Returns:
        Image.Image: The object representing the image from the file.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        return image
    finally:
        await file.close()
