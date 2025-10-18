import asyncio
import io
from contextlib import asynccontextmanager
from uuid import uuid4

from config import CLIPConfig, DatabaseConfig
from dependencies import (get_postgres_manager, get_image_repo,
                          get_embedding_model_manager, get_chroma_manager,
                          get_search_controller, get_clip_service,
                          get_vector_repo)
from fastapi import Depends, FastAPI
from handler import search_router
from managers import (ChromaConnectionManager, CLIPManager,
                      PostgresConnectionManager, RedisConnectionManager)
from ml import CLIPModelService
from models import ImageMetadataModel, VectorEntry
from PIL import Image
from repository import ImageRepository, VectorRepository

from fastapi import UploadFile
from controller import SearchController

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and teardown configuration"""

    # Load .env configs
    database_config = DatabaseConfig()
    embedding_model_config = CLIPConfig()

    # Create managers
    embedding_model_manager = CLIPManager(embedding_model_config)
    pg_manager = PostgresConnectionManager(database_config)
    chromadb_manager = ChromaConnectionManager(database_config)
    redis_manager = RedisConnectionManager(database_config)

    # Init db conns
    # TODO: Standardize startup manager
    await asyncio.get_event_loop().run_in_executor(
        None, embedding_model_manager.initialize
    )
    await pg_manager.initialize_connection()
    chromadb_manager.initialize_connection()
    redis_manager.initialize_connection()

    app.state.embedding_model_manager = embedding_model_manager
    app.state.pg_manager = pg_manager
    app.state.chromadb_manager = chromadb_manager
    app.state.redis_manager = redis_manager

    try:
        yield
    finally:
        embedding_model_manager.teardown()
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
async def test(
    search_controller: SearchController = Depends(get_search_controller),
    embedding_service: CLIPModelService = Depends(get_clip_service),
    vector_service: VectorRepository = Depends(get_vector_repo),
    image_service: ImageRepository = Depends(get_image_repo),
):
    with open('imgs/orange.webp', 'rb') as f:
        contents = f.read()
    img = UploadFile(io.BytesIO(contents))
    similarImgs = await search_controller.search(img, 2)
    return {"success": similarImgs}


app.include_router(search_router, prefix="/api")
