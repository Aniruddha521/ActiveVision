from pydantic import BaseModel
from typing import Annotated
from .image_state import ImageState
from .reducers import str_reducer, image_reducer

class ActiveVisionOutputState(BaseModel):
    output_image: Annotated[ImageState, image_reducer] = None
    response: Annotated[str, str_reducer] = None
