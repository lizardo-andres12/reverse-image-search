from functools import lru_cache

from controller import SearchController
from fastapi import Depends
from ml import CLIPModelService
from repository import ImageRepository, VectorRepository


@lru_cache
def get_clip_service() -> CLIPModelService:
    # TODO: add config parsing
    return CLIPModelService()


@lru_cache
def get_vector_repo() -> VectorRepository:
    return VectorRepository()


@lru_cache
def get_image_repo() -> ImageRepository:
    return ImageRepository()


@lru_cache
def get_search_controller(
    clip_service: CLIPModelService = Depends(get_clip_service),
    vector_repo: VectorRepository = Depends(get_vector_repo),
    image_repo: ImageRepository = Depends(get_image_repo),
) -> SearchController:
    return SearchController(
        clip_service=clip_service,
        vector_repository=vector_repo,
        image_repository=image_repo,
    )
