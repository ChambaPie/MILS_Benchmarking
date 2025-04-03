#!/usr/bin/env python3
import os
import json
import shutil
import argparse
import glob
from pathlib import Path

def extract_subset_images(split_json_dir=None, split_index=None):
    """
    Extract images from val2014 based on split JSON files.
    
    Args:
        split_json_dir: Directory containing the split JSON files
        split_index: Optional index to process only one split file
    """
    # Paths
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    val2014_dir = root_dir / "val2014"
    val2014_split_dir = root_dir / "val2014_split"
    
    # If no directory is specified, use the default location
    if split_json_dir is None:
        split_json_dir = root_dir / "MILS" / "data" / "coco" / "test" / "split_val_partitions"
    else:
        split_json_dir = Path(split_json_dir)
    
    # Create the main output directory
    val2014_split_dir.mkdir(exist_ok=True)
    print(f"Created main directory: {val2014_split_dir}")
    
    # Check if val2014 exists
    if not val2014_dir.exists():
        print(f"Error: Source directory {val2014_dir} does not exist")
        return
    
    # Find all split JSON files
    if split_index is not None:
        # Process only one specific split file
        split_files = [split_json_dir / f"split_val_{split_index}.json"]
        if not split_files[0].exists():
            print(f"Error: Split file {split_files[0]} does not exist")
            return
    else:
        # Process all split files
        split_files = sorted(split_json_dir.glob("split_val_*.json"))
    
    if not split_files:
        print(f"Error: No split files found in {split_json_dir}")
        return
    
    print(f"Found {len(split_files)} split files to process")
    
    # Process each split file
    total_copied = 0
    total_images = 0
    
    for split_file in split_files:
        # Extract the split index from the filename
        split_name = split_file.stem  # Gets filename without extension
        split_index = split_name.split('_')[-1]  # Extract the number part
        
        # Create subdirectory for this split
        output_dir = val2014_split_dir / f"val2014_{split_index}"
        output_dir.mkdir(exist_ok=True)
        print(f"\nProcessing split {split_index} -> {output_dir}")
        
        # Load the split JSON file
        print(f"Loading data from {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Extract the filenames from the split data
        filenames = []
        for img in split_data.get('images', []):
            if 'filename' in img:
                filenames.append(img['filename'])
        
        total_images += len(filenames)
        print(f"Found {len(filenames)} images in split {split_index}")
        
        # Copy the images
        copied_count = 0
        for filename in filenames:
            src_path = val2014_dir / filename
            dst_path = output_dir / filename
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                # Print progress for every 10th image
                if copied_count % 10 == 0:
                    print(f"Progress: {copied_count}/{len(filenames)} images copied")
            else:
                print(f"Warning: Source file not found: {src_path}")
        
        total_copied += copied_count
        print(f"Split {split_index} complete: {copied_count} of {len(filenames)} images copied to {output_dir}")
    
    # Final summary
    print(f"\n===== EXTRACTION SUMMARY =====")
    print(f"Total splits processed: {len(split_files)}")
    print(f"Total images extracted: {total_copied} of {total_images}")
    
    # Verify the extraction
    if total_copied == total_images:
        print("✅ All images successfully extracted")
    else:
        print(f"⚠️ Only {total_copied} of {total_images} images were found and copied")
        
    return val2014_split_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images based on split JSON files")
    parser.add_argument("--split-dir", type=str, default=None,
                       help="Directory containing split JSON files")
    parser.add_argument("--split-index", type=int, default=None,
                       help="Process only a specific split index (e.g., 1 for split_val_1.json)")
    
    args = parser.parse_args()
    
    output_dir = extract_subset_images(args.split_dir, args.split_index)
    print(f"Main output directory: {output_dir}") 