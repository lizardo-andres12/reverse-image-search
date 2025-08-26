import asyncio
from contextlib import asynccontextmanager

from dependencies import get_clip_service
from config import DatabaseConfig
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and teardown configuration"""
    database_config = DatabaseConfig()

    # Load model on startup
    clip_service = get_clip_service()
    await asyncio.get_event_loop().run_in_executor(None, clip_service.load_model)

    app.state.database_config = database_config

    try:
        yield
    # App teardown
    finally:
        clip_service.unload_model()


app = FastAPI(lifespan=lifespan)
