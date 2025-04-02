from pydantic import BaseModel
from typing import Annotated
from .image_state import ImageState
from .reducers import image_path_reducer, str_reducer, image_reducer

class ActiveVisionOverallState(BaseModel):
    image: Annotated[ImageState, image_reducer] 
    image_path: Annotated[str, image_path_reducer]
    query: Annotated[str,str_reducer]
    output_image: Annotated[ImageState, image_reducer] = None
    response: Annotated[str, str_reducer] = None
    human_feedback: Annotated[str, str_reducer] = ""