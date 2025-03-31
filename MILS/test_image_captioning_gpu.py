# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import transformers
import torchvision.transforms as transforms
import argparse
import tqdm
import random
import json
from open_clip import create_model_and_transforms, get_tokenizer
from optimization_utils import (
    Scorer as S,
    Generator as G,
    get_text_features,
    get_image_features,
)

from paths import IMAGEC_COCO_ANNOTATIONS, IMAGEC_COCO_IMAGES, IMAGEC_COCO_SPLITS, OUTPUT_DIR


def optimize_for_images(
    args, text_pipeline, image_filenames, text_prompt, clip_model, tokenizer, preprocess
):
    print(f"Processing batch of {len(image_filenames)} images...")
    
    loggers = {}
    save_locations = {}
    for image_filename in image_filenames:
        save_locations[image_filename] = os.path.join(args.output_dir, image_filename.replace('.jpg', ''))
        os.makedirs(save_locations[image_filename], exist_ok=True)
        loggers[image_filename] = open(
            os.path.join(save_locations[image_filename], "log.txt"), "w"
        )

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
    
    print("Preparing image paths...")
    # Format image paths correctly for GPU version
    image_paths = [
        os.path.join(args.images_path, image_filename)
        for image_filename in image_filenames
    ]
    
    target_features = (
        get_image_features(
            clip_model,
            preprocess,
            image_paths,
            args.device,
            args.batch_size
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

    scorers = {}
    for i, image_filename in enumerate(image_filenames):
        scorers[image_filename] = {
            "func": clip_scorer,
            "target_feature": target_features[i],
        }

    scorer = S(
        scorers, args.batch_size, key=lambda x: -x, keep_previous=args.keep_previous
    )
    ###
    # Initialize the pool
    ###
    with open(args.init_descriptions, "r") as w:
        init_sentences = [i.strip() for i in w.readlines()]
    if args.init_descriptions_set_size != "all":
        random.seed(0) # Choose a different seed than args as it is already used (should not matter though)
        init_sentences = random.sample(init_sentences, int(args.init_descriptions_set_size)) # Must be all or an int
    lines_with_scores = {}
    initial_scores = {}
    for i, image_filename in enumerate(image_filenames):
        init_scores = scorer.score(image_filename, init_sentences)
        lines_with_scores[image_filename] = [
            (s, l) for (s, l) in zip(init_scores, init_sentences)
        ]
        best_score = sorted(lines_with_scores[image_filename], key=lambda x: -x[0])[0]
        initial_scores[image_filename] = best_score
        mean_score = np.mean(init_scores)
        bs = best_score[1].strip()
        loggers[image_filename].write(f"{best_score[0]}\t{mean_score}\t{bs}\n")
    
    # No need for optimization iterations in testing mode
    # Just use the best initial description
    for k, logger in loggers.items():
        logger.close()


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
    with open(args.annotations_path, "r") as w:
        annotations = json.load(w)["annotations"]

    clip_model, _, preprocess = create_model_and_transforms(
        args.clip_model, pretrained=args.pretrained
    )
    tokenizer = get_tokenizer(args.clip_model)
    clip_model.to(args.device)
    clip_model.eval()
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
    )
    if 'Ministral' in args.text_model:
        text_pipeline.tokenizer.pad_token_id = text_pipeline.model.config.eos_token_id
    
    # Load images from split.json
    with open(IMAGEC_COCO_SPLITS) as f:
        split_data = json.load(f)['images']
    
    # Get image filenames from split file
    image_filenames = [x['filename'] for x in split_data]
    print(f"Found {len(image_filenames)} images in split file")
    
    # Limit to specified number of images
    if args.max_images is not None:
        print(f"Limiting to {args.max_images} images")
        image_filenames = image_filenames[:args.max_images]
    
    # Only filter out existing directories if force_process is not set
    if not args.force_process:
        image_filenames = [x for x in image_filenames if not os.path.exists(os.path.join(args.output_dir, x.replace('.jpg', '')))]
    print(f"Length of the data is {len(image_filenames)}")

    # Process images in batches
    image_filenames = image_filenames[args.process :: args.num_processes]
    while len(image_filenames):
        current_batch = []
        while len(current_batch) < args.llm_batch_size and image_filenames:
            image_filename = image_filenames[0]
            image_id = image_filename.replace('.jpg', '')
            if (
                (not os.path.exists(os.path.join(args.output_dir, image_id)) or args.force_process)
                and image_filename not in current_batch
            ):
                current_batch.append(image_filename)
            image_filenames = image_filenames[1:]
        if current_batch:
            optimize_for_images(
                args,
                text_pipeline,
                current_batch,
                text_prompt,
                clip_model,
                tokenizer,
                preprocess,
            )


def get_args_parser():
    parser = argparse.ArgumentParser("Image Captioning with COCO", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument(
        "--annotations_path",
        default=IMAGEC_COCO_ANNOTATIONS,
    )
    parser.add_argument(
        "--images_path", default=IMAGEC_COCO_IMAGES
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--process", default=0, type=int)
    parser.add_argument("--max_images", default=None, type=int, help="Maximum number of images to process")
    parser.add_argument("--force_process", action="store_true", help="Process images even if output directory exists")
    

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--llm_batch_size", default=16, type=int, help="Batch size for llms"
    )
    parser.add_argument("--keep_previous", default=50, type=int, help="Keep previous")
    parser.add_argument(
        "--requested_number", default=50, type=int, help="How many to request"
    )
    parser.add_argument(
        "--iterations", default=10, type=int, help="Optimization iterations"
    )
    parser.add_argument(
        "--clip_model",
        default="RN50",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="openai", type=str)

    parser.add_argument(
        "--text_model",
        default="facebook/opt-125m",
        type=str,
        help="Text model",
    )

    parser.add_argument(
        "--init_descriptions",
        default="init_descriptions/image_descriptions_per_class.txt",
        type=str,
        help="init descriptions pool",
    )
    parser.add_argument(
        "--init_descriptions_set_size",
        default="all",
        type=str,
        help="How many descriptions to choose, should be either int or an int",
    )
    parser.add_argument(
        "--prompt", default="prompts/image_captioning_shorter.txt", type=str, help="Prompt"
    )
    parser.add_argument("--exploration", default=0.0, type=float, help="exploration")
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    prompt = args.prompt.split("/")[-1].split(".")[0]
    name = "imagec_test_gpu"  # Changed to indicate this is for testing with GPU
    args.output_dir = os.path.join(
        args.output_dir,
        f"{name}_{text_model}_{args.iterations}_{args.exploration}_{args.keep_previous}_{args.requested_number}_{args.clip_model}_{args.pretrained}_{prompt}_{args.init_descriptions_set_size}",
    )
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args) 