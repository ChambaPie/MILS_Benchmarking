import json
import tempfile
from tqdm import tqdm
import evaluate
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Load results
with open("blip2_results_5000.json", "r") as f:
    data = json.load(f)

captions = data["captions"]

annotations, results, images = [], [], []
preds_for_meteor, refs_for_meteor = [], []

for entry in captions:
    image_id = entry["image_id"]
    pred = entry["caption"].strip()
    refs = [r.strip() for r in entry["references"] if r.strip()]

    if pred.lower() == "error" or not pred or not refs:
        continue

    for i, r in enumerate(refs):
        annotations.append({
            "image_id": image_id,
            "id": image_id * 10 + i,
            "caption": r
        })

    results.append({
        "image_id": image_id,
        "caption": pred
    })
    images.append({"id": image_id})

    # For HuggingFace METEOR
    preds_for_meteor.append(pred)
    refs_for_meteor.append(refs)

print(f"\n📦 Cleaned {len(results)} valid predictions with references...")

# Save temp files for COCO-style eval
with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as ref_file, \
     tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as res_file:

    json.dump({
        "annotations": annotations,
        "images": images,
        "type": "captions",
        "licenses": [],
        "info": {}
    }, ref_file)
    ref_file.flush()

    json.dump(results, res_file)
    res_file.flush()

    coco = COCO(ref_file.name)
    coco_res = coco.loadRes(res_file.name)

    gts = {img["id"]: [ann["caption"] for ann in coco.imgToAnns[img["id"]]] for img in images}
    res = {img["id"]: [ann["caption"]] for img in images for ann in coco_res.imgToAnns[img["id"]]}

    scorers = [
        ("BLEU", Bleu(4)),
        ("ROUGE_L", Rouge()),
        ("CIDEr", Cider())
    ]

    print("\n📊 Final Metrics:")

    for name, scorer in scorers:
        print(f"\n▶ {name}")
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(score, list):
                for i, s in enumerate(score):
                    print(f"{name}_{i+1}: {s:.4f}")
            else:
                print(f"{name}: {score:.4f}")
        except Exception as e:
            print(f"⚠️ Failed to compute {name}: {e}")

    # Hugging Face METEOR
    print("\n▶ METEOR (HuggingFace)")
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=preds_for_meteor, references=refs_for_meteor)
    print(f"METEOR: {meteor_result['meteor']:.4f}")