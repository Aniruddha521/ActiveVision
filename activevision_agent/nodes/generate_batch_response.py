import os
import io
import base64
import numpy as np
import cv2 as cv
from PIL import Image
from dotenv import load_dotenv
import dspy
from activevision_agent.dspy_signatures import BatchOutputGenerationSignature
from activevision_agent.states import BatchSubgraphOverallState, ImageState

import torch
from diffusers import AutoPipelineForInpainting

# Initialize the diffusers inpainting pipeline globally.
# Note: This requires a CUDA-enabled device.
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cpu")

generate_batch_output = dspy.ChainOfThought(BatchOutputGenerationSignature)

def modify_image_diffusers_inpainting(image_bytes: bytes, prompt: str) -> bytes:
    """
    Modifies the given image according to the prompt using the diffusers inpainting model 
    (stable-diffusion-xl-1.0-inpainting-0.1).
    
    The image is loaded from bytes using PIL and resized to (1024, 1024). A mask image is created 
    (currently a white mask, meaning the whole image will be inpainted). The pipeline then produces 
    a modified image based on the prompt.
    
    Args:
        image_bytes (bytes): Input image data (e.g., from a JPEG file) in bytes.
        prompt (str): The textual prompt describing the modification.
    
    Returns:
        bytes: The modified image in PNG format as bytes.
    
    Raises:
        Exception: If the inpainting pipeline fails.
    """
    # Load the image from bytes using PIL.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((1024, 1024))
    
    # Create a mask image of the same size.
    # Here we create a grayscale ("L") mask that is fully white (255) so the entire image is modified.
    mask_image = Image.new("L", image.size, 255)
    
    # Set a generator seed for reproducibility.
    generator = torch.Generator(device="cuda").manual_seed(0)
    
    # Run the inpainting pipeline.
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well
        strength=0.99,  # ensure strength is below 1.0
        generator=generator,
    )
    
    # Extract the first image from the pipeline output.
    modified_image = result.images[0]
    
    # Convert the modified image to bytes (PNG format).
    buffer = io.BytesIO()
    modified_image.save(buffer, format="PNG")
    return buffer.getvalue()

def generate_output(state: BatchSubgraphOverallState) -> BatchSubgraphOverallState:
    """
    Generates a batch output by creating a text response via dspy and modifying the input image 
    using the diffusers inpainting model (stable-diffusion-xl-1.0-inpainting-0.1).
    
    It uses a language model (meta-llama/Llama-3.2-3B-Instruct from Hugging Face via dspy) to generate text 
    instructions (which are used as the inpainting prompt) and then modifies the input image accordingly.
    
    Updates:
        - state.response: the generated textual instructions.
        - state.output_image: an ImageState containing the modified image.
    
    Returns:
        BatchSubgraphOverallState: The updated state.
    """
    load_dotenv()
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    lm = dspy.LM(model="meta-llama/Llama-3.2-3B-Instruct", api_key=hf_api_key, custom_llm_provider="huggingface")
    with dspy.context(lm=lm):
        result = generate_batch_output(
            query=state.query,
            feedback=state.proofreader_feedback
        )
    
    instructions = result.get("response", "")
    state.response = instructions
    
    try:
        modified_img_bytes = modify_image_diffusers_inpainting(state.image.image, prompt=instructions)
        state.output_image = ImageState(image=modified_img_bytes, image_name=f"modified_{state.image.image_name}")
    except Exception as e:
        state.response += f"\nFailed to modify image: {e}"
        state.output_image = ImageState()
    
    return state
