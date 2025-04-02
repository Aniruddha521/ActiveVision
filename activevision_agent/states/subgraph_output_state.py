from pydantic import BaseModel
from typing import Annotated
from .image_state import ImageState
from .reducers import str_reducer, image_reducer

class BatchSubgraphOutputState(BaseModel):
    response: Annotated[str, str_reducer]
    output_image: Annotated[ImageState, image_reducer]