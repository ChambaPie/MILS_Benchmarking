# image_captioning_experiment.py

from PIL import Image
import requests
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from openai import OpenAI
import os

# Load BLIP-2 (Salesforce)
def load_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

# Generate caption using BLIP-2
def generate_blip2_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, return_tensors="pt").to(model.device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

# Generate caption using GPT-4V (if API key is set)
def generate_gpt4v_caption(image_path):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_bytes.encode("base64")}}
            ]}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content']

# Run on COCO-style dataset
def run_captioning_experiment(image_dir):
    processor, model = load_blip2_model()
    results = []

    for image_name in os.listdir(image_dir):
        if not image_name.endswith((".jpg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_name)

        blip_caption = generate_blip2_caption(image_path, processor, model)
        print(f"BLIP-2 caption: {blip_caption}")

        # Optional: uncomment if using GPT-4V
        # gpt4v_caption = generate_gpt4v_caption(image_path)
        # print(f"GPT-4V caption: {gpt4v_caption}")

        results.append({
            "image": image_name,
            "blip2": blip_caption,
            # "gpt4v": gpt4v_caption
        })

    return results

# Example usage
if __name__ == "__main__":
    image_folder = "sample_images"  # directory with test images
    captions = run_captioning_experiment(image_folder)
    print(captions)
