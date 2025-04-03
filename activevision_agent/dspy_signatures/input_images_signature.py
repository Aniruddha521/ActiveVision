import dspy
from pydantic import BaseModel


class InputImage(BaseModel):
    """Contains the input image name and image itself."""

    image_name: str = dspy.InputField(desc="Name of the given image.")
    image: str = dspy.InputField(desc="Given image in bytes")
