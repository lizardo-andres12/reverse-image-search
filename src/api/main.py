import asyncio
from contextlib import asynccontextmanager

from dependencies import get_clip_service
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    clip_service = get_clip_service()
    await asyncio.get_event_loop().run_in_executor(None, clip_service.load_model)
    yield
    clip_service.unload_model()


app = FastAPI(lifespan=lifespan)
