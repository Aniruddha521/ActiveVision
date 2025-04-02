from pydantic import BaseModel
from typing import Annotated
from .image_state import ImageState
from .reducers import str_reducer, image_reducer

class BatchSubgraphOverallState(BaseModel):
    image: Annotated[ImageState, image_reducer]
    query: Annotated[str, str_reducer]
    max_review: int = 0
    proofreader_approval: bool = False
    proofreader_feedback: dict = {}
    response: Annotated[str, str_reducer]
    output_image: Annotated[ImageState, image_reducer]