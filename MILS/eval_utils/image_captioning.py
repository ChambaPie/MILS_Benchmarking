# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from collections import defaultdict
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import tempfile
import sys
import subprocess
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import IMAGEC_COCO_ANNOTATIONS, IMAGEC_COCO_IMAGES, IMAGEC_COCO_SPLITS, OUTPUT_DIR


def extract_captions(folder_path, index_to_choose=-1):
    captions = []
    for root, _, files in os.walk(folder_path):
        if 'log.txt' in files:
            with open(os.path.join(root, 'log.txt'), 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Warning: Empty log.txt file in {root}")
                    continue
                    
                # Make sure we don't go out of bounds
                if abs(index_to_choose) > len(lines):
                    print(f"Warning: Index {index_to_choose} out of bounds for file with {len(lines)} lines. Using last line.")
                    index_to_choose = -1
                
                line = lines[index_to_choose].strip()
                # Check if line has tab separators (old format with scores)
                if '\t' in line:
                    caption = line.split('\t')[-1]
                else:
                    # New format without scores, just the caption
                    caption = line
                    
                # Get the image ID from the directory name
                sample_id = os.path.basename(root)
                try:
                    image_id = int(sample_id)
                    captions.append({"image_id": image_id, "caption": caption})
                except ValueError:
                    print(f"Warning: Could not convert {sample_id} to integer. Skipping this caption.")
    return captions


def main():
    # Make output more verbose
    print(f"Starting image captioning evaluation...")
    
    # Process command line arguments
    annotation_file = IMAGEC_COCO_ANNOTATIONS
    
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = OUTPUT_DIR
        print(f"No directory specified, using default: {result_dir}")
    
    try:
        if len(sys.argv) > 2:
            index_to_choose = int(sys.argv[2])
        else:
            index_to_choose = -1  # last line by default
    except:
        index_to_choose = -1  # last line by default
    
    print(f"Using caption index: {index_to_choose} (negative means from end, -1 = last line)")
    print(f"Reading captions from: {result_dir}")
    
    # Extract captions from the result directory
    result_data = extract_captions(result_dir, index_to_choose)
    
    if not result_data:
        print("Error: No captions found in the specified directory!")
        sys.exit(1)
    
    print(f"Found {len(result_data)} captions to evaluate")
    
    # Print the first few captions for reference
    print("\nFirst few captions:")
    for i, cap in enumerate(result_data[:3]):
        print(f"Image {cap['image_id']}: {cap['caption']}")
    
    # Create COCO objects for reference annotations
    print(f"\nLoading reference annotations from {annotation_file}")
    coco = COCO(annotation_file)
    
    # Check for image ID matches
    our_image_ids = [x['image_id'] for x in result_data]
    coco_image_ids = coco.getImgIds()
    matches = [id for id in our_image_ids if id in coco_image_ids]
    
    print(f"\nImage ID matching:")
    print(f"Our image IDs: {our_image_ids}")
    print(f"Matching IDs in COCO: {matches}")
    
    if not matches:
        print("\nWARNING: None of our image IDs match the COCO dataset IDs!")
        print("This will result in low or zero scores.")
        sys.exit(1)
    
    # Create temporary file for result data (needed for COCO format)
    print("\nPreparing results for evaluation...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(result_data, temp_file, indent=2)
        temp_file.flush()
        result_file_path = temp_file.name
    
    # Initialize scores dictionary
    scores = {}
    
    # Load model results into COCO format
    coco_result = coco.loadRes(result_file_path)
    eval_img_ids = coco_result.getImgIds()
    
    # ===== DIRECT METRIC CALCULATION =====
    # This bypasses the problematic Java dependency entirely
    print("\nCalculating evaluation metrics directly...")
    
    # Format data for metric calculations
    gts = {}
    res = {}
    
    # Get all ground truth captions
    for img_id in eval_img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            print(f"Warning: No reference captions found for image {img_id}")
            continue
        
        anns = coco.loadAnns(ann_ids)
        # Store just the caption strings, not dictionaries
        gts[img_id] = [ann['caption'] for ann in anns]
    
    # Format our captions the same way - just strings, not dictionaries
    for item in result_data:
        res[item['image_id']] = [item['caption']]
    
    # Calculate BLEU score
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        print("Computing BLEU scores...")
        bleu_scorer = Bleu(n=4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        for i, score in enumerate(bleu_scores):
            scores[f'Bleu_{i+1}'] = score
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
    
    # Calculate METEOR score
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        print("Computing METEOR score...")
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
        scores['METEOR'] = meteor_score
    except Exception as e:
        print(f"Error calculating METEOR: {e}")
    
    # Calculate ROUGE-L score
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        print("Computing ROUGE-L score...")
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        scores['ROUGE_L'] = rouge_score
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}")
    
    # Calculate CIDEr score
    try:
        from pycocoevalcap.cider.cider import Cider
        print("Computing CIDEr score...")
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        scores['CIDEr'] = cider_score
    except Exception as e:
        print(f"Error calculating CIDEr: {e}")
    
    # Print final scores and save to file
    print("\nEvaluation Results:")
    results = {}
    for metric, score in sorted(scores.items()):
        results[metric] = float(score)
        print(f"{metric}: {score:.6f}")
    
    # Save results to a file
    output_dir = os.path.dirname(result_dir) if os.path.isdir(result_dir) else result_dir
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {results_file}")
    
    # Clean up
    try:
        os.remove(result_file_path)
    except:
        pass
    
    print("\nEvaluation complete")


if __name__ == "__main__":
    main() 