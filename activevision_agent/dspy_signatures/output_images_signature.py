from pydantic import BaseModel
import dspy
from typing import List, Tuple

class GeneratedOutput(BaseModel):
    """Generated email mapped to the lead_id."""
    image_name: str = dspy.OutputField(desc="Name of the images file as given in the input")
    image: str = dspy.OutputField(desc="Generated or modified image as per user request")
    # bbox: List[List[List]] = dspy.OutputField(
    #     desc="Bounding boxes accross the items user mention, "
    #     "for every instance of the mention object bounding box look like [(X_min, Y_min), (X_max, Y_max)]"
    #     "and if there are n-objects whcih are mention then there will be n-bounding boxes"
    #     "else return None if user is not asking for any bounding boxes"
    #     )
