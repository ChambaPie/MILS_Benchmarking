# Benchmarking MILS
 ### The point of this excercise is to compare MILS to other benchmark applications (GPT-4V and BLIP-2)

#### We will conduct 3 Experiments:

1. **Image Captioning** \
**Prompt:** "Describe this image in detail." \
**Evaluation Metrics:** BLEU, METEOR, CIDEr, SPICE \
**Human Evaluation:** Fluency + relevance (1–5 scale)

    **Dataset:** 1K images from COCO Val2014

2. **Visual Question Answering (VQA)** \
**Prompt:** Natural language questions (e.g., “What is the boy doing?”) \
**Evaluation Metrics:** Exact match with ground truth answers \
**Additional:** Optional visual reasoning questions

    **Dataset:** VQAv2

3. **Referring Expressions** \
**Prompt:** “Point to the red car” or “Highlight the person wearing a hat.” \
**Output:** Either a bounding box or textual description of the region. \
**Evaluation Metrics:** Intersection over Union (IoU), referring accuracy

    **Dataset:** RefCOCO

##### Tools:

🧠 MILS (access from Meta repo or API) \
🤖 BLIP-2: Salesforce GitHub \
📷 GPT-4V: Use OpenAI’s gpt-4-vision-preview model (if available) \
📦 Hugging Face for dataset loading: datasets, torchvision \
📊 Evaluation libraries: nlg-eval, pycocoevalcap, scikit-image
