from activevision_agent.states import (
    ActiveVisionEntryState, 
    ActiveVisionOverallState,
    ImageState
)
import os

def load_images(state: ActiveVisionEntryState) -> ActiveVisionOverallState:
    """
    Loads an image from the given file path, converts it to bytes,
    and stores it in the ImageState.
    """
    if not os.path.exists(state.image_path):
        raise FileNotFoundError(f"Image not found: {state.image_path}")

    with open(state.image_path, "rb") as img_file:
        img_bytes = img_file.read()

    image = ImageState(
        image=img_bytes,
        image_name=os.path.basename(state.image_path)
    )

    new_state = ActiveVisionOverallState(
        image=image,
        image_path=state.image_path,
        query=state.query,
        output_image=ImageState(),
        response=""
    )

    return new_state
