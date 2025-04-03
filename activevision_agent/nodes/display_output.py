from activevision_agent.states import ActiveVisionOverallState, ImageState
import os
import io
from PIL import Image
from IPython.display import display

def display_output(state: ActiveVisionOverallState) -> ActiveVisionOverallState:
    image_bytes = state.output_image.image
    
    pil_img = Image.open(io.BytesIO(image_bytes))
    
    os.makedirs("output_images", exist_ok=True)
    
    output_path = f"output_images/{state.output_image.image_name}"
    pil_img.save(output_path)
    print(f"Image saved to: {output_path}")
    display(pil_img)  
    
    return state
