import asyncio
import io
import logging
from functools import partial

from fastapi import HTTPException, UploadFile
from ml import CLIPModelService
from models import QueryHit, SearchResponse, SimilarImage
from PIL import Image
from repository import ImageRepository, VectorRepository

logger = logging.getLogger(__name__)
MAX_SIZE = 10 * 1024 * 1024


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

    async def search(self, file: UploadFile, limit: int) -> SearchResponse:
        """
        Performs file conversion to a vector and searches for similar images.

        Args:
            file (UploadFile): The file from HTTP request.
            limit (int): The maximum number of similar images to return.
        Returns:
            SearchResponse: The top (limit) many keywords and images.
        """
        try:
            loop = asyncio.get_event_loop()
            self._validate_upload(file)

            image = await self._process_file_upload(file)
            embedding = await loop.run_in_executor(
                None, partial(self._extract_features, image)
            )

            similar_vectors = self._search_similar_vectors(embedding, limit)
            similar_images = self._fetch_image_metadata(similar_vectors)
            keywords = self._generate_keywords(similar_images)

            return SearchResponse(
                keywords=keywords,
                similar_images=similar_images,
                total_found=len(similar_images),
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def _validate_upload(self, file: UploadFile):
        """
        Ensures the uploaded file is an image

        Args:
            file (UploadFile): The file from HTTP request.
        """

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if file.size and file.size > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

    async def _process_file_upload(self, file: UploadFile) -> Image.Image:
        """
        Converts the uploaded file to PIL Image.

        Args:
            file (UploadFile): The file from HTTP Request.
        Returns:
            Image.Image: The object representing the image from the file.
        """

        try:
            contents = await file.read()
            await file.close()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            return image
        except Exception as e:
            await file.close()
            raise HTTPException(
                status_code=400, detail="Invalid or corrupted image file"
            )

    def _extract_features(self, image: Image.Image) -> list:
        """
        Extract CLIP features from image.

        Args:
            image (Image.Image): The PIL image to process.
        Returns:
            list: The CLIP vector embeddings as python list from numpy array
        """
        try:
            features = self.clip_service.extract_image_features(image)
            if not features:
                raise HTTPException(
                    status_code=500, detail="Failed to extract image features"
                )

            logger.info(f"Extracted features with shape {features.shape}")
            return features.tolist()
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to extract image features"
            )

    def _search_similar_vectors(
        self, embedding: list[float], limit: int
    ) -> list[SimilarImage]:
        """
        Search for similar vectors in ChromaDB. Uses cosine similarity to find matches.

        Args:
            embedding (list): Vector embedding list to perform cosine similarity on.
            limit (int): Upper bound of images to return.
        Returns:
            list[dict]: List of image uuids and corresponding vector embeddings.
        """
        results = self.vector_repository.query_similar(embedding, limit)

    def _fetch_image_metadata(self, similar_vectors: list[dict]) -> list[SimilarImage]:
        """
        Fetch metadata for similar images.

        Args:
            similar_vectors (list[dict]): The similar vector data with uuid and similarity score.
        Returns:
            list[SimilarImage]: List of SimilarImage result objects with uuid, similarity score, and
            retrieval URLs.
        """
        pass

    def _generate_keywords(self, similar_images: list[SimilarImage]) -> list[str]:
        """
        Generate keywords from similar images.

        Args:
            similar_images (list[SimilarImage]): The results of similarity scoring pipeline.
        Returns:
            list[str]: List of the top MAX_TAGS frequent tags retrieved by result image uuids.
        """
        pass
