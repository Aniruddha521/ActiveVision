from pydantic import BaseModel
from typing import Annotated
from .image_state import ImageState
from .reducers import str_reducer, image_reducer


class BatchSubgraphEntryState(BaseModel):
    image: Annotated[ImageState, image_reducer]
    query: Annotated[str, str_reducer]