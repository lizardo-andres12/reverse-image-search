from controller import SearchController
from dependencies import get_search_controller
from fastapi import APIRouter, Depends, File, UploadFile
from models import SearchResponse

search_router = APIRouter(prefix="/search")


@search_router.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    search_controller: SearchController = Depends(get_search_controller),
    limit: int = 20,
):
    """
    Search vector database for similar images and return response.

    Args:
        file (UploadFile): The image file to consider.
    Returns:
        SearchResponse: The SearchResponse model containing keywords related to image and
        a list of images bounded by limit. If an error occurs, an HTTPException will be raised
    """
    return await search_controller.search(file, limit)
