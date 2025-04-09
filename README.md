# MILS Benchmarking: Image Captioning Comparison

This repository contains code for benchmarking MILS against GPT-4V and BLIP-2 for image captioning tasks.

## Experiment: Image Captioning

### Setup
1. **Dataset**: 1K images from COCO Val2014
2. **Prompt**: "Describe this image in detail."
3. **Evaluation Metrics**: BLEU, METEOR, CIDEr

### Tools Used
- 🧠 MILS (access from Meta repo or API)
- 🤖 BLIP-2: Salesforce GitHub
- 📷 GPT-4V: OpenAI's gpt-4-vision-preview model
- 📦 Hugging Face for dataset loading
- 📊 Evaluation libraries: nlg-eval, pycocoevalcap

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

### Running Image Captioning on Modal (Two Options)

#### Option 1: Using the Original Script Directly
This approach runs the original MILS image captioning script directly on Modal without modifications.

1. Run the execution script:
```bash
python image_captioning_modal_execute.py
```

2. The script will:
   - Create a Modal container with all required dependencies
   - Execute the original MILS/main_image_captioning_original_version.py script
   - Save results to the Modal volume
   - Download results to your local machine

#### Option 2: Using the Optimized Version
This approach uses a rewritten, optimized version of the image captioning script.

1. Run the optimized script:
```bash
python image_captioning_modal_optimized.py
```

2. The script will:
   - Create a Modal container with all required dependencies
   - Run the optimized image captioning code
   - Save results to the Modal volume
   - Download results to your local machine

### Output
- Captions are saved in the `MILS/output/coco_captions/full_output_val_1` directory
- Each image gets its own directory with a `log.txt` file containing the caption
- The log.txt file contains scores and the caption for each iteration
- The final caption is found on the last line of the log.txt file

### Evaluation
After generating captions, evaluate them using:
```bash
python MILS/eval_utils/image_captioning.py MILS/output/coco_captions/full_output_val_1
```

### Notes
- The script uses Modal's A100 GPU for processing
- Rate limiting is implemented to avoid API limits
- Results are persisted using Modal volumes
