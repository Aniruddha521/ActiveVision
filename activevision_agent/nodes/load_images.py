import os
import io
from PIL import Image
from activevision_agent.states import (
    ActiveVisionEntryState, 
    ActiveVisionOverallState,
    ImageState
)

def load_images(state: ActiveVisionEntryState) -> ActiveVisionOverallState:

    if not os.path.exists(state.image_path):
        raise FileNotFoundError(f"Image not found: {state.image_path}")
    
    with open(state.image_path, "rb") as img_file:
        image_bytes = img_file.read()
    
    image_state = ImageState(
        image=image_bytes,
        image_name=os.path.basename(state.image_path)
    )
    
    new_state = ActiveVisionOverallState(
        image=image_state,
        image_path=state.image_path,
        query=state.query,
        output_image=ImageState(),
        response=""
    )
    
    return new_state
