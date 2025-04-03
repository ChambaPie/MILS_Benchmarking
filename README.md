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

### Running the Modal Script

1. The main script for running image captioning on Modal is `image_captioning_modal.py`

2. To run the script:
```bash
python image_captioning_modal.py
```

3. The script will:
   - Process images from the COCO dataset
   - Generate captions using the specified model
   - Save results to the output directory
   - Create evaluation metrics

### Output
- Captions are saved in the `output/coco_captions` directory
- Each image gets its own directory with a `log.txt` file containing the caption
- Evaluation results are saved in `evaluation_results.json`

### Notes
- The script uses Modal's A100 GPU for processing
- Rate limiting is implemented to avoid API limits
- Results are persisted using Modal volumes
