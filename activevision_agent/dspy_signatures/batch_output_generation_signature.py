from typing import Dict, List
import dspy
from .input_images_signature import InputImage
from .output_images_signature import GeneratedOutput


class BatchOutputGenerationSignature(dspy.Signature):
    """
    Signature for batch processing of image modifications based on a user query. 
    The system processes a set of input images, applies modifications according 
    to the query, and outputs the transformed images along with relevant bounding 
    boxes. Additionally, the system incorporates previous reviewer feedback 
    to refine the modifications.
    """

    query: str = dspy.InputField(
        desc="User query describing the modifications to be applied to the images."
    )
   
    feedback: Dict[str, str] = dspy.InputField(
        desc="Previous feedback from a reviewer on modified images, mapping filenames to comments.",
        default_factory=dict,
    )
    output: GeneratedOutput = dspy.OutputField(
        desc="Processed image and bounding boxes reflecting the applied modifications."
    )