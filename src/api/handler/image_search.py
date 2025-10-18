from controller import SearchController
from dependencies import get_search_controller
from fastapi import APIRouter, Depends, File, Form, UploadFile
from models import SimilarImage

search_router = APIRouter(prefix="/search")


@search_router.post("/image", response_model=list[SimilarImage])
async def search(
    search_controller: SearchController = Depends(get_search_controller),
    file: UploadFile = File(...),
    limit: int = Form(20),
):
    """
    Search vector database for similar images and return response.

    Args:
        file (UploadFile): The image file to consider.
    Returns:
        list[SimilarImage]: The matched images with metadata sorted by confidence.
    """
    similar_images = await search_controller.search(file, limit)
    return similar_images
