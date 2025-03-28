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

# Force CPU usage
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

def optimize_for_images(
    args, text_pipeline, image_ids, text_prompt, clip_model, tokenizer, preprocess
):
    loggers = {}
    save_locations = {}
    for image_id in image_ids:
        save_locations[f"{image_id}"] = os.path.join(args.output_dir, f"{image_id}")
        os.makedirs(save_locations[f"{image_id}"], exist_ok=True)
        loggers[f"{image_id}"] = open(
            os.path.join(save_locations[f"{image_id}"], "log.txt"), "w"
        )

    generator = G(
        text_pipeline,
        args.text_model,
        requested_number=args.requested_number,
        keep_previous=args.keep_previous,
        prompt=text_prompt,
        key=lambda x: -x[0],
    )

    def clip_scorer(sentences, target_feature):
        return S.clip_scorer(
            sentences,
            target_feature,
            clip_model,
            tokenizer,
            preprocess,
            args.device,
        )

    scorers = {}
    for i, image_id in enumerate(image_ids):
        scorers[f"{image_id}"] = {
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
    for i, image_id in enumerate(image_ids):
        init_scores = scorer.score(f"{image_id}", init_sentences)
        lines_with_scores[f"{image_id}"] = [
            (s, l) for (s, l) in zip(init_scores, init_sentences)
        ]
        best_score = sorted(lines_with_scores[f"{image_id}"], key=lambda x: -x[0])[0]
        initial_scores[f"{image_id}"] = best_score
        mean_score = np.mean(init_scores)
        bs = best_score[1].strip()
        loggers[f"{image_id}"].write(f"{best_score[0]}\t{mean_score}\t{bs}\n")
    ###
    # Do the optimization:
    ###
    for it in range(args.iterations):
        torch.cuda.empty_cache()
        new_lines = generator(lines_with_scores)
        # new_lines is similar to lines in structure
        lines_with_scores = scorer(
            new_lines
        )  # This suppose to return dict of description -> (score, text)
        best_value = scorer.get_best_value()  # Text to score
        best = scorer.get_best()  # Text to (text, image)
        average_value = scorer.get_average_value()  # Text to score
        for key in average_value:
            # assert initial_scores[key] <= best_value[key][0], (initial_scores[key], best_value[key][0])
            loggers[key].write(
                f"{best_value[key][0]}\t{average_value[key]}\t{best[key]}\n"
            )
    for k, logger in loggers.items():
        logger.close()


def main(args):
    # Load annotations
    with open(args.annotations_path, "r") as f:
        annotations = json.load(f)
    
    # Get image IDs and limit to max_images
    image_ids = [ann["image_id"] for ann in annotations["annotations"]]
    image_ids = list(set(image_ids))[:args.max_images]  # Limit to max_images unique images
    
    # Load CLIP model
    clip_model, _, preprocess = create_model_and_transforms(
        args.clip_model, pretrained=args.pretrained
    )
    clip_model.to(device)
    
    # Load tokenizer
    tokenizer = get_tokenizer(args.clip_model)
    
    # Load text pipeline
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        device=-1,  # Force CPU
        model_kwargs={"torch_dtype": torch.float32},
    )
    
    # Load prompt
    with open(args.prompt, "r") as f:
        text_prompt = f.read()
    
    # Run optimization
    optimize_for_images(
        args, text_pipeline, image_ids, text_prompt, clip_model, tokenizer, preprocess
    )


def get_args_parser():
    parser = argparse.ArgumentParser("MILS Image Captioning", add_help=False)
    parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument("--images_path", default=IMAGEC_COCO_IMAGES, type=str)
    parser.add_argument("--annotations_path", default=IMAGEC_COCO_ANNOTATIONS, type=str)
    parser.add_argument("--splits_path", default=IMAGEC_COCO_SPLITS, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--process", default=0, type=int)
    parser.add_argument("--max_images", default=5, type=int, help="Maximum number of images to process")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
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
    # Dataset parameters
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no_ablation", action="store_false", dest="ablation")
    parser.set_defaults(ablation=False)
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    prompt = args.prompt.split("/")[-1].split(".")[0]
    name = "imagec_g" if not args.ablation else "imagec_a"
    args.output_dir = os.path.join(
        args.output_dir,
        f"{name}_{text_model}_{args.iterations}_{args.exploration}_{args.keep_previous}_{args.requested_number}_{args.clip_model}_{args.pretrained}_{prompt}_{args.init_descriptions_set_size}",
    )
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
