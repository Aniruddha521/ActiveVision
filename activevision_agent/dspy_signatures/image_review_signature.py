from pydantic import BaseModel
import dspy


class ImagesReview(BaseModel):
    """Approval status and detailed feedback for each email."""

    image_name: str = dspy.OutputField(desc="Corresponding image file name as given in the input")
    approved: bool = dspy.OutputField(desc="Approval status for each image.")
    feedback: str = dspy.OutputField(
        desc="""
        Detailed feedback for each image if not approved; otherwise, just write "image modified according to user.".
        """
    )
