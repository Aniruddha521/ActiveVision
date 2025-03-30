import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import dspy
from activevision_agent.dspy_signatures import BatchOutputGenerationSignature
from activevision_agent.states import BatchSubgraphOverallState, ImageState

generate_batch_output = dspy.ChainOfThought(BatchOutputGenerationSignature)

def modify_image_hf_api(image_bytes: bytes, prompt: str, model_id: str = "stabilityai/stable-diffusion-inpainting") -> bytes:
    
    load_dotenv()
    api_token = os.getenv("HUGGINGFACE_API_KEY")
    if not api_token:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.new("L", image.size, 255)
    
    def image_to_base64(img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    img_b64 = image_to_base64(image)
    mask_b64 = image_to_base64(mask)
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": img_b64,
            "mask_image": mask_b64,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
        }
    }
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Hugging Face API call failed: {response.status_code} {response.text}")
    
    try:
        result = response.json()
        if "generated_image" in result:
            output_b64 = result["generated_image"]
        elif isinstance(result, list) and "generated_image" in result[0]:
            output_b64 = result[0]["generated_image"]
        else:
            return response.content
    except Exception:
        return response.content

    return base64.b64decode(output_b64)

def generate_output(state: BatchSubgraphOverallState) -> BatchSubgraphOverallState:

    load_dotenv()
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")
    
    with dspy.context(lm=dspy.LM(model="mosaicml/mpt-7b-chat", api_key=hf_api_key)):
        result = generate_batch_output(
            query=state.query,
            images=[state.image],
            feedback=state.proofreader_feedback
        )
    
    instructions = result.get("response", "")
    state.response = instructions
    
    try:
        modified_img_bytes = modify_image_hf_api(state.image.image, prompt=instructions)
        state.output_image = ImageState(image=modified_img_bytes, image_name=f"modified_{state.image.image_name}")
    except Exception as e:
        state.response += f"\nFailed to modify image: {e}"
        state.output_image = ImageState()
    
    return state
