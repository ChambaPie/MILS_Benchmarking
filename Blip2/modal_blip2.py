import modal

# --- Modal Setup ---
app = modal.App("blip2-evaluator")
volume = modal.Volume.from_name("coco-blip2-data", create_if_missing=False)

image = (
    modal.Image.debian_slim()
    .apt_install("git", "default-jre")
    .pip_install("torch", "transformers", "nltk", "pillow", "tqdm", "pycocotools")
    .run_commands(
        "git clone https://github.com/salaniz/pycocoevalcap.git /pycocoevalcap",
        "cd /pycocoevalcap && pip install ."
    )
    .run_commands("python -m nltk.downloader wordnet omw-1.4")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 360,
    volumes={"/data": volume},
)
def run_blip2():
    import os
    import json
    import torch
    from PIL import Image
    from tqdm import tqdm
    import nltk
    import tempfile
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    from pycocotools.coco import COCO
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    nltk.download('wordnet')
    nltk.download('omw-1.4')

    Meteor.JAR_PATH = "/usr/local/lib/python3.9/site-packages/pycocoevalcap/meteor/meteor-1.5.jar"

    def safe_compute_score(self, gts, res):
        import subprocess
        import tempfile

        assert gts.keys() == res.keys()
        scores = []

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as hyp_file, \
             tempfile.NamedTemporaryFile(mode="w", delete=False) as ref_file:

            for img_id in gts:
                hypothesis = res[img_id][0].strip().replace('\n', ' ')
                references = [r.strip().replace('\n', ' ') for r in gts[img_id]]
                hyp_file.write(f"{hypothesis}\n")
                ref_file.write("|||".join(references) + "\n")

            hyp_file.flush()
            ref_file.flush()

            cmd = [
                "java", "-jar", self.JAR_PATH,
                hyp_file.name, ref_file.name,
                "-l", "en", "-norm"
            ]

            print("[METEOR] Running METEOR Java subprocess...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            score = 0.0
            for line in process.stdout:
                decoded = line.decode("utf-8", errors="replace").strip()
                print("[METEOR Raw Output]:", decoded)
                if decoded.lower().startswith("final score:"):
                    try:
                        score = float(decoded.split(":")[1].strip())
                        scores = [score] * len(gts)
                        break
                    except Exception as e:
                        print("⚠️ Failed to parse METEOR score:", e)
                        score = 0.0

            process.stdout.close()
            process.wait()
            os.unlink(hyp_file.name)
            os.unlink(ref_file.name)

            return score, scores

    Meteor.compute_score = safe_compute_score

    class COCOEvalCap:
        def __init__(self, coco, cocoRes):
            self.evalImgs = []
            self.eval = {}
            self.imgToEval = {}
            self.coco = coco
            self.cocoRes = cocoRes
            self.evalType = 'caption'
            self.include_metrics = {
                "Bleu": Bleu(4),
                "METEOR": Meteor(),
                "ROUGE_L": Rouge(),
                "CIDEr": Cider()
            }

        def evaluate(self):
            imgIds = self.cocoRes.getImgIds()
            gts = {imgId: [ann["caption"] for ann in self.coco.imgToAnns[imgId]] for imgId in imgIds}
            res = {imgId: [ann["caption"] for ann in self.cocoRes.imgToAnns[imgId]] for imgId in imgIds}

            for method, scorer in self.include_metrics.items():
                print(f"Evaluating {method}...")
                score, scores = scorer.compute_score(gts, res)
                if isinstance(score, list):
                    for i, s in enumerate(score):
                        self.eval[f"{method}_{i + 1}"] = s
                    for imgId, sc in zip(imgIds, zip(*scores)):
                        self.imgToEval.setdefault(imgId, {})
                        for i, s in enumerate(sc):
                            self.imgToEval[imgId][f"{method}_{i + 1}"] = s
                else:
                    self.eval[method] = score
                    if isinstance(scores, list) and len(scores) == len(imgIds):
                        for i, imgId in enumerate(imgIds):
                            self.imgToEval.setdefault(imgId, {})
                            self.imgToEval[imgId][method] = scores[i]
                    else:
                        for imgId in imgIds:
                            self.imgToEval.setdefault(imgId, {})
                            self.imgToEval[imgId][method] = score

    COCO_IMAGES_DIR = "/data/val2014"
    COCO_CAPTIONS_FILE = "/data/captions_val2014.json"
    COCO_SPLIT_FILE = "/data/split_5000.json"

    with open(COCO_CAPTIONS_FILE, 'r') as f:
        coco_data = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
    image_refs = {}
    for ann in coco_data["annotations"]:
        image_refs.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    with open(COCO_SPLIT_FILE, 'r') as f:
        split_data = json.load(f)

    selected_image_ids = [img["cocoid"] for img in split_data["images"]]
    sample_image_ids = [img_id for img_id in selected_image_ids if img_id in image_refs]

    print(f"Evaluating {len(sample_image_ids)} images...")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")

    predictions, annotations, images_info = [], [], []

    for idx, image_id in enumerate(tqdm(sample_image_ids, desc="Captioning"), 1):
        file_name = image_id_to_file.get(image_id)
        image_path = os.path.join(COCO_IMAGES_DIR, file_name)
        print(f"[{idx}/{len(sample_image_ids)}] {file_name}")

        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        refs = image_refs.get(image_id, [])
        if not refs:
            continue

        try:
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        except Exception:
            caption = "error"

        predictions.append({
            "image_id": image_id,
            "file_name": file_name,
            "caption": caption,
            "references": refs
        })

        for i, r in enumerate(refs):
            annotations.append({
                "image_id": image_id,
                "id": image_id * 10 + i,
                "caption": r
            })

        images_info.append({"id": image_id})

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as pred_file, \
         tempfile.NamedTemporaryFile(mode="w+", delete=False) as ref_file:

        json.dump([{ "image_id": p["image_id"], "caption": p["caption"] } for p in predictions], pred_file)
        pred_file.flush()

        json.dump({
            "annotations": annotations,
            "images": images_info,
            "type": "captions",
            "info": {},
            "licenses": []
        }, ref_file)
        ref_file.flush()

        coco = COCO(ref_file.name)
        coco_res = coco.loadRes(pred_file.name)

        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.evaluate()

    with open("/data/blip2_results_5000.json", "w") as f:
        json.dump({
            "metrics": coco_eval.eval,
            "captions": predictions
        }, f, indent=2)

    print("\n✅ Final COCO-style Metrics:")
    for k, v in coco_eval.eval.items():
        print(f"{k}: {v:.4f}")

    print("\n📁 Results saved to /data/blip2_results_5000.json")