import logging

from fastapi import HTTPException, UploadFile
from mapper import async_upload_file_to_pil_image
from ml import CLIPModelService
from models import SimilarImage
from repository import ImageRepository, VectorRepository

logger = logging.getLogger(__name__)


class SearchController:
    """Class encapsulating methods related to searching for similar images in the database"""

    def __init__(
        self,
        clip_service: CLIPModelService,
        vector_repository: VectorRepository,
        image_repository: ImageRepository,
    ):
        """Initializes object of SearchController with specified dependencies."""

        self.clip_service = clip_service
        self.vector_repository = vector_repository
        self.image_repository = image_repository

    async def search(self, file: UploadFile, limit: int) -> list[SimilarImage]:
        """Performs file conversion to a vector and searches for similar images.

        Args:
            file (UploadFile): The file from HTTP request.
            limit (int): The maximum number of similar images to return.
        Returns:
            SearchResponse: The top (limit) many keywords and images.
        """
        image = await async_upload_file_to_pil_image(file)
        embedding = self.clip_service.extract_image_features(image)
        similar_embeddings = self.vector_repository.query_similar(embedding, limit)

        similar_ids = [result.id for result in similar_embeddings]
        similar_metadatas = await self.image_repository.batch_get_image_metadata(
            similar_ids
        )

        if not similar_embeddings or not similar_metadatas:
            raise ValueError("One or more queries returned none")

        results = []
        for image, metadata in zip(similar_embeddings, similar_metadatas):
            results.append(
                SimilarImage(
                    id=image.id,
                    similarity=image.similarity,
                    source_url=metadata.source_url,
                    source_domain=metadata.source_domain,
                    filename=metadata.filename,
                    file_size=metadata.file_size,
                    dimensions=metadata.dimensions,
                )
            )

        return results

    """
    def _validate_upload(self, file: UploadFile):

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if file.size and file.size > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
    """
