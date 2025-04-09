# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Output dir - will be created if it doesn't exist
OUTPUT_DIR = '/root/MILS/output/coco_captions/'

# Image captioning paths for Modal execution
IMAGEC_COCO_ANNOTATIONS = '/root/MILS/data/coco/annotations/captions_val2014.json'
IMAGEC_COCO_IMAGES = '/root/MILS/data/coco/val2014'
IMAGEC_COCO_SPLITS = '/root/MILS/data/coco/test/split.json'

# Video captioning
VIDEOC_MSRVTT_ANNOTATIONS = '/path/to/msrvtt/test_videodatainfo.json'
VIDEOC_MSRVTT_VIDEOS = '/path/to/msrvtt/TestVideo/'

# Audio captioning
AUDIOC_CLOTHO_ANNOTATIONS = '/path/to/Clotho2/clotho_captions_evaluation.csv'
AUDIOC_CLOTHO_FILES = '/path/to/Clotho2/wav/'

# Image generation

# Style transfer 