from pydantic import BaseModel
from typing import Annotated
from .reducers import image_path_reducer, str_reducer


class ActiveVisionEntryState(BaseModel):
    image_path: Annotated[str, image_path_reducer]
    query: Annotated[str, str_reducer]