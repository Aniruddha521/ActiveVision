from typing import List, Tuple
import dspy
from .input_images_signature import InputImage
from .image_review_signature import ImagesReview


class BatchOutputReviewerSignature(dspy.Signature):
    """
    Signature for a batch multimodal review task that processes a user query to modify images 
    and then produces a detailed review of the modified outputs.
    """
    query: str = dspy.InputField(
        desc="Query made by user and so task is to generated the result as per user requirements."
    )
    images: List[InputImage] = dspy.InputField(
        desc="Contains the user given images and their file name."
    )
    bbox: List[List[Tuple]] = dspy.OutputField(
        desc="Bounding boxes accross the items user mention, "
        "for every instance of the mention object bounding box look like [(X_min, Y_min), (X_max, Y_max)]"
        "and if there are n-objects whcih are mention then there will be n-bounding boxes"
        "else return None if user is not asking for any bounding boxes"
        )
    review: List[ImagesReview] = dspy.OutputField(
        desc="Approval status and detailed feedback for each image."
    ) 
