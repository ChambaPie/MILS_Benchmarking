#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from PIL import Image
import numpy as np
import transformers
import argparse
import json
from open_clip import create_model_and_transforms, get_tokenizer
from optimization_utils import (
    Scorer as S,
    Generator as G,
    get_text_features,
    get_image_features as get_image_features_batch,
)

from paths import OUTPUT_DIR

def get_image_features(clip_model, image_path, preprocess, device):
    """Process a single image path and return its features."""
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            im = Image.open(image_path)
            im = im.convert("RGB")
            im = preprocess(im).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(im)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

def optimize_for_image(
    args, text_pipeline, image_path, text_prompt, clip_model, tokenizer, preprocess
):
    save_location = os.path.join(args.output_dir, os.path.basename(image_path))
    os.makedirs(save_location, exist_ok=True)
    logger = open(os.path.join(save_location, "log.txt"), "w")

    generator = G(
        text_pipeline,
        args.text_model,
        requested_number=args.requested_number,
        keep_previous=args.keep_previous,
        prompt=text_prompt,
        key=lambda x: -x[0],
        batch_size=args.batch_size,
        device=args.device,
        exploration=args.exploration,
    )
    
    target_features = (
        get_image_features(
            clip_model,
            image_path,
            preprocess,
            args.device
        )
        .detach()
        .cpu()
        .numpy()
    )

    def clip_scorer(sentences, target_feature):
        text_features = get_text_features(
            clip_model,
            tokenizer,
            sentences,
            args.device,
            args.batch_size,
            amp=True,
            use_format=False,
        )
        return text_features.detach().cpu().numpy() @ target_feature

    scorers = {
        "image": {
            "func": clip_scorer,
            "target_feature": target_features[0],
        }
    }

    scorer = S(
        scorers, args.batch_size, key=lambda x: -x, keep_previous=args.keep_previous
    )
    
    # Initialize the pool
    with open(args.init_descriptions, "r") as w:
        init_sentences = [i.strip() for i in w.readlines()]
    if args.init_descriptions_set_size != "all":
        init_sentences = init_sentences[:int(args.init_descriptions_set_size)]
    
    lines_with_scores = {}
    initial_scores = {}
    
    init_scores = scorer.score("image", init_sentences)
    lines_with_scores["image"] = [
        (s, l) for (s, l) in zip(init_scores, init_sentences)
    ]
    best_score = sorted(lines_with_scores["image"], key=lambda x: -x[0])[0]
    initial_scores["image"] = best_score
    mean_score = np.mean(init_scores)
    bs = best_score[1].strip()
    logger.write(f"{best_score[0]}\t{mean_score}\t{bs}\n")
    
    # Do the optimization
    for it in range(args.iterations):
        torch.cuda.empty_cache()
        new_lines = generator(lines_with_scores)
        lines_with_scores = scorer(new_lines)
        best_value = scorer.get_best_value()
        best = scorer.get_best()
        average_value = scorer.get_average_value()
        for key in average_value:
            logger.write(f"{best_value[key][0]}\t{average_value[key]}\t{best[key]}\n")
    
    logger.close()
    
    # Return the best caption
    return best["image"]


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device for Apple Silicon
    if torch.backends.mps.is_available():
        args.device = "mps"
    elif torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    print(f"Using device: {args.device}")
    
    with open(args.prompt, "r") as w:
        text_prompt = w.read()

    clip_model, _, preprocess = create_model_and_transforms(
        args.clip_model, pretrained=args.pretrained
    )
    tokenizer = get_tokenizer(args.clip_model)
    clip_model.to(args.device)
    clip_model.eval()
    
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.float32},
        device_map="cpu"  # Use CPU for the large language model
    )
    if 'Ministral' in args.text_model:
        text_pipeline.tokenizer.pad_token_id = text_pipeline.model.config.eos_token_id
    
    best_caption = optimize_for_image(
        args,
        text_pipeline,
        args.image_path,
        text_prompt,
        clip_model,
        tokenizer,
        preprocess,
    )
    
    print(f"\nBest caption for {args.image_path}:")
    print(best_caption)


def get_args_parser():
    parser = argparse.ArgumentParser("Image Captioning for a Single Image", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--image_path", required=True, help="Path to the image to caption")
    
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--keep_previous", default=50, type=int, help="Keep previous")
    parser.add_argument(
        "--requested_number", default=50, type=int, help="How many to request"
    )
    parser.add_argument(
        "--iterations", default=10, type=int, help="Optimization iterations"
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-SO400M-14-SigLIP",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="webli", type=str)

    parser.add_argument(
        "--text_model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help="Text model",
    )
    parser.add_argument(
        "--init_descriptions",
        default="init_descriptions/image_descriptions_per_class.txt",
        type=str,
        help="Initial descriptions",
    )
    parser.add_argument("--init_descriptions_set_size", default="all", type=str)
    parser.add_argument(
        "--prompt",
        default="prompts/image_captioning_shorter_all.txt",
        type=str,
        help="Finetuning Prompt for the LLM",
    )
    parser.add_argument("--exploration", default=0.0, type=float, help="Exploration")
    
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image Captioning for a Single Image", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 