#!/usr/bin/env python
# GPT-4V Image Captioning Benchmark

import os
import json
import argparse
import base64
import requests
import time
from tqdm import tqdm
import glob
from PIL import Image
from paths import IMAGEC_COCO_ANNOTATIONS, IMAGEC_COCO_IMAGES, IMAGEC_COCO_SPLITS, OUTPUT_DIR

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt4v_caption(api_key, image_path, prompt="Describe this image concisely."):
    """Get a caption from GPT-4V for an image"""
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    try:
        print(f"Sending request to OpenAI API...")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error getting caption for {image_path}: {e}")
        if 'response' in locals():
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
        return f"Error: {str(e)}"

def extract_image_id(filename):
    """Extract the numeric image ID from COCO filename format: COCO_val2014_000000000042.jpg -> 42"""
    try:
        # Extract the numeric part from the filename
        numeric_part = filename.split('_')[-1].split('.')[0]
        # Remove leading zeros and convert to int
        return int(numeric_part)
    except (IndexError, ValueError):
        # Fallback for filenames not in the expected format
        print(f"Warning: Could not extract numeric ID from {filename}")
        return None

def process_images(args):
    """Process images using GPT-4V and save captions in the format needed for eval"""
    
    # Check for API key
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load split to get test images
    with open(args.split_path) as f:
        split_data = json.load(f)['images']
    
    # Get test images and their IDs
    test_images = []
    for img in split_data:
        if img['split'] == 'test':
            # CRITICAL: Use cocoid (NOT imgid) for the directory name
            # This is what the evaluation script is looking for
            test_images.append({
                'filename': img['filename'],
                'image_id': img['cocoid'] if 'cocoid' in img else extract_image_id(img['filename']),
                'real_image_id': img['imgid'] if 'imgid' in img else None  # Store for reference
            })
    
    print(f"Found {len(test_images)} test images in split")
    
    # Limit to specified number of images if requested
    if args.max_images and args.max_images < len(test_images):
        test_images = test_images[:args.max_images]
    
    print(f"Processing {len(test_images)} images")
    
    # Process each image
    captions = []
    for img_data in tqdm(test_images):
        image_filename = img_data['filename']
        image_id = img_data['image_id']
        real_id = img_data['real_image_id']
        
        print(f"Processing image {image_filename} with COCO ID {image_id} (imgid: {real_id})")
        
        image_path = os.path.join(args.images_path, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found, skipping")
            continue
        
        # Create output directory for this image using COCO ID
        image_output_dir = os.path.join(args.output_dir, str(image_id))
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Get caption from GPT-4V
        caption = get_gpt4v_caption(api_key, image_path, args.prompt)
        
        # Save caption to log.txt in the format expected by the eval script
        with open(os.path.join(image_output_dir, "log.txt"), "w") as f:
            f.write(f"{caption}\n")
        
        # Also save in a format that's easier to review
        captions.append({
            "image_id": image_id, 
            "real_id": real_id,
            "filename": image_filename,
            "caption": caption
        })
        
        # Optional rate limiting to avoid OpenAI API rate limits
        if args.rate_limit > 0:
            time.sleep(args.rate_limit)
    
    # Save all captions to a JSON file for easy review
    with open(os.path.join(args.output_dir, "gpt4v_captions.json"), "w") as f:
        json.dump(captions, f, indent=2)
    
    print(f"\nCaptions saved to {args.output_dir}")
    print(f"To evaluate, run: python MILS/eval_utils/image_captioning.py {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate image captions using GPT-4V")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--images_path", type=str, default=IMAGEC_COCO_IMAGES, help="Path to images")
    parser.add_argument("--split_path", type=str, default=IMAGEC_COCO_SPLITS, help="Path to Karpathy split JSON")
    parser.add_argument("--output_dir", type=str, default=os.path.join(OUTPUT_DIR, "gpt4v_captions"), 
                        help="Directory to save output")
    parser.add_argument("--max_images", type=int, default=50, help="Maximum number of images to process")
    parser.add_argument("--prompt", type=str, default="Describe this image concisely in one sentence.", 
                        help="Prompt to send to GPT-4V")
    parser.add_argument("--rate_limit", type=float, default=2.0, 
                        help="Seconds to wait between API calls (to avoid rate limits)")
    parser.add_argument("--run_eval", action="store_true", 
                        help="Run evaluation script automatically after generating captions")
    
    args = parser.parse_args()
    process_images(args)

if __name__ == "__main__":
    main() 