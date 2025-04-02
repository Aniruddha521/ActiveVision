from pydantic import BaseModel

class ImageState(BaseModel):
    image: bytes  = None
    image_name: str = ""