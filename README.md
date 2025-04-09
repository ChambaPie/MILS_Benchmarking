# MILS Benchmarking: Image Captioning Comparison

This repository contains code for benchmarking MILS against GPT-4V and BLIP-2 for image captioning tasks.

For information on how to set up the virtual environment to run locally, refer to the README.md file in the MILS directory

## Experiment: Image Captioning

### Setup
1. **Dataset**: 1K images from COCO Val2014
2. **Prompt**: "Describe this image in detail."
3. **Evaluation Metrics**: BLEU, METEOR, CIDEr

### Tools Used
- MILS (access from Meta repo or API)
- BLIP-2: Salesforce GitHub
- GPT-4V: OpenAI's gpt-4-vision-preview model
- Hugging Face for dataset loading
- Evaluation libraries: nlg-eval, pycocoevalcap

## Modal Setup and Usage

### Prerequisites
1. Install Modal:
```bash
pip install modal
```

2. Set up Modal CLI:
```bash
modal token new
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

### Running Image Captioning on Modal 

#### Using the Original Script Directly
This approach runs the original MILS image captioning script directly on Modal without modifications.

1. Update the paths for the split file and image directory location in **modal/paths_modal.py**

```python
IMAGEC_COCO_IMAGES = f'/root/MILS/data/coco/{image_directory}'
IMAGEC_COCO_SPLITS = f'/root/MILS/data/coco/test/{split_file}.json'
```

These values are hardcoded, so please enter your specific directory name and corresponding split file

2. Update the subdirectory in the **main_image_captioning_original_version.py** script

```python
sub = "MILS_1" # Edit this manually for each run (MILS_1, MILS_2, etc.)
```

3. Run the execution script:
```bash
modal run wrapper_executor.py
```

4. The script will:
   - Create a Modal container with all required dependencies
   - Execute the original MILS/main_image_captioning_original_version.py script
   - Save results to the Modal volume
   - Download results to your local machine


### Evaluation
After generating captions, evaluate them using:
```bash
python MILS/eval_utils/image_captioning.py MILS/output/coco_captions/full_output_val_1
```

### Notes
- The script uses Modal's A100 GPU 80GB for processing
- Rate limiting is implemented to avoid API limits
- Results are persisted using Modal volumes
