import os
import sys
import modal
import subprocess

from MILS.modal.paths_modal import OUTPUT_DIR, IMAGEC_COCO_ANNOTATIONS, IMAGEC_COCO_IMAGES, IMAGEC_COCO_SPLITS

sub = "MILS_7" # Edit this manually for each run (MILS_1, MILS_2, etc.)

# Set up Modal environment
app = modal.App("mils-image-captioning-wrapper")

# Create an image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0", 
        "torchvision==0.16.0", 
        "transformers>=4.38.0",
        "accelerate>=0.26.0",
        "matplotlib", 
        "pillow", 
        "tqdm", 
        "numpy<2.0",
        "open_clip_torch",
    )
    .apt_install("git")
    .pip_install("git+https://github.com/openai/CLIP.git")
)

# Add the entire MILS directory
image = image.add_local_dir("./MILS", "/root/MILS")
# Add the validation images
image = image.add_local_dir("./MILS/data/coco/val2014", "/root/data/coco/val2014")

# Create a volume to persist output data
output_volume = modal.Volume.from_name("image-captioning-output", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-80GB", 
    timeout=86400,
    volumes={"/root/output_data": output_volume}
)
def run_captioning():
    """Run the image captioning on Modal with A100 GPU by executing the main script"""
    import os
    import subprocess
    import shutil
    
    # Use the same subdirectory name as defined at the top of the file
    volume_subdir = sub
    
    # Create the paths_modal.py file to paths.py so it will be used by the script
    from shutil import copyfile
    copyfile("/root/MILS/modal/paths_modal.py", "/root/MILS/paths.py")
    print("Using modal-specific paths from paths_modal.py")
    
    # Create hardcoded subdirectory for this run
    run_output_dir = os.path.join(OUTPUT_DIR, volume_subdir)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Created output directory: {run_output_dir}")
    
    # Set HuggingFace token
    token = os.environ.get("HUGGINGFACE_TOKEN", "Your Token Here")
    os.environ["HF_TOKEN"] = token
    os.environ["TRANSFORMERS_TOKEN"] = token
    
    # Create the necessary directories
    os.makedirs("/root/data/coco/annotations", exist_ok=True)
    
    # Build command to execute the script directly with the new output directory
    cmd = ["python", "-u", "/root/MILS/main_image_captioning_original_version.py", 
           "--prompt", "/root/MILS/prompts/image_captioning_shorter.txt",
           "--init_descriptions", "/root/MILS/init_descriptions/image_descriptions_per_class.txt",
           "--output_dir", run_output_dir]
    
    # Execute the script
    print("\n=== STARTING SCRIPT EXECUTION ===")
    result = subprocess.run(cmd, capture_output=False, text=True)
    print("=== SCRIPT EXECUTION COMPLETE ===\n")
    
    # Copy results to the volume
    if os.path.exists(run_output_dir) and os.path.isdir(run_output_dir):
        # Use a hardcoded name in the volume that can be manually edited between runs
        output_data_path = os.path.join("/root/output_data/coco_captions", volume_subdir)
        
        # Remove existing directory if it exists
        if os.path.exists(output_data_path):
            shutil.rmtree(output_data_path)
        
        os.makedirs(output_data_path, exist_ok=True)
        
        print(f"Copying results from {run_output_dir} to volume at {output_data_path}...")
        
        # Copy all contents from run directory to the volume subdirectory
        for item in os.listdir(run_output_dir):
            src_path = os.path.join(run_output_dir, item)
            dst_path = os.path.join(output_data_path, item)
            
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            elif os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
        
        print(f"✅ Successfully copied results to volume subdirectory: {volume_subdir}")
        
        # Commit the volume to ensure data is persisted
        output_volume.commit()
        print("✅ Committed changes to volume")
    
    return {
        "output_dir": run_output_dir,
        "success": result.returncode == 0
    }

@app.function(
    volumes={"/root/output_data": output_volume}
)
def download_results():
    """Download the files from the output volume"""
    import os
    import base64
    
    # Use the same subdirectory name as defined at the top of the file
    volume_subdir = sub
    output_dir_path = os.path.join("/root/output_data/coco_captions", volume_subdir)
    
    print(f"Downloading results from {output_dir_path}...")
    
    if not os.path.exists(output_dir_path):
        print(f"Warning: Path {output_dir_path} does not exist in the volume")
        return {}
    
    file_contents = {}
    
    for root, dirs, files in os.walk(output_dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, output_dir_path)
            
            try:
                with open(file_path, 'rb') as f:
                    file_contents[rel_path] = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    print(f"Read {len(file_contents)} files from volume")
    return file_contents

@app.local_entrypoint()
def main():
    # Create local directory to save downloaded files
    local_output_dir = f"./MILS/output/coco_captions/{sub}"
    os.makedirs(local_output_dir, exist_ok=True)
    
    print("Starting image captioning on Modal with A100 GPU")
    print(f"Results will be saved in volume subdirectory: {sub}")
    print("(Edit sub variable at the top of the file to change this for each run)")
    
    # Run the captioning job
    result = run_captioning.remote()
    
    print(f"Captioning completed with success: {result['success']}")
    print(f"Results saved to: {result['output_dir']}")
    
    # Download the results
    download_info = download_results.remote()
    print(f"Found {len(download_info)} result files in the volume subdirectory {sub}")
    
    # Save the downloaded files to the local directory
    if len(download_info) > 0:
        import base64
        count = 0
        
        print(f"\nDownloading files to your local machine at: {local_output_dir}")
        
        for rel_path, content_b64 in download_info.items():
            local_file_path = os.path.join(local_output_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            try:
                with open(local_file_path, 'wb') as f:
                    content = base64.b64decode(content_b64)
                    f.write(content)
                count += 1
            except Exception as e:
                print(f"Error saving file {rel_path}: {e}")
        
        print(f"✅ Successfully downloaded {count} files to {local_output_dir}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 