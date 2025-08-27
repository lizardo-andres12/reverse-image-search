from functools import lru_cache

from connections import (ChromaConnectionManager, PostgresConnectionManager,
                         RedisConnectionManager)
from controller import SearchController
from fastapi import Depends, Request
from ml import CLIPModelService
from repository import ImageRepository, VectorRepository


@lru_cache
def get_chroma_manager(request: Request):
    return request.app.state.chroma_manager


@lru_cache
def get_redis_manager(request: Request):
    return request.app.state.redis_manager


@lru_cache
def get_postgres_manager(request: Request):
    return request.app.state.pg_manager


@lru_cache
def get_clip_service() -> CLIPModelService:
    # TODO: add config parsing
    return CLIPModelService()


@lru_cache
def get_vector_repo(
    chroma_manager: ChromaConnectionManager = Depends(get_chroma_manager),
    redis_manager: RedisConnectionManager = Depends(get_redis_manager),
) -> VectorRepository:
    return VectorRepository()


@lru_cache
def get_image_repo(
    postgres_manager: PostgresConnectionManager = Depends(get_postgres_manager),
    redis_manager: RedisConnectionManager = Depends(get_redis_manager),
) -> ImageRepository:
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
