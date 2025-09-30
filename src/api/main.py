import asyncio
from contextlib import asynccontextmanager

from models import ImageMetadataModel, ImageTagModel
from config import DatabaseConfig
from connections import (ChromaConnectionManager, PostgresConnectionManager,
                         RedisConnectionManager)
from dependencies import (get_chroma_manager, get_clip_service,
                          get_postgres_manager)
from fastapi import Depends, FastAPI
from handler import search_router
from repository import VectorRepository, ImageRepository
import uuid


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and teardown configuration"""

    # Load .env configs
    database_config = DatabaseConfig()

    # Create db conns
    chromadb_manager = ChromaConnectionManager(database_config)
    redis_manager = RedisConnectionManager(database_config)
    pg_manager = PostgresConnectionManager(database_config)

    # Init db conns
    chromadb_manager.initialize_connection()
    redis_manager.initialize_connection()
    await pg_manager.initialize_connection()

    # Load model on startup
    clip_service = get_clip_service()
    await asyncio.get_event_loop().run_in_executor(None, clip_service.load_model)

    app.state.chromadb_manager = chromadb_manager
    app.state.redis_manager = redis_manager
    app.state.pg_manager = pg_manager

    try:
        yield
    # App teardown
    finally:
        clip_service.unload_model()
        await redis_manager.close_connection()
        await pg_manager.close_connection()


app = FastAPI(lifespan=lifespan)


@app.get("/healthcheck")
async def healthcheck(
    pg: PostgresConnectionManager = Depends(get_postgres_manager),
    chromadb: ChromaConnectionManager = Depends(get_chroma_manager),
):
    pg_health = await pg.healthcheck()
    chroma_db_health = chromadb.healthcheck()

    return pg_health | chroma_db_health


@app.get("/test")
async def test(chromadb: ChromaConnectionManager = Depends(get_chroma_manager)):
    vr = VectorRepository(chromadb)
    results = await vr.get_image_metadata('a8897316-6441-4877-8436-7739f77cbaa9')
    return {'message': 'success', 'results': results}


app.include_router(search_router, prefix="/api")
