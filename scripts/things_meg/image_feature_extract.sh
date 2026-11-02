#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

python scripts/data/extract_feature.py --image_set_dir "./data/things_meg/image_set" --output_dir "./data/things_meg/image_feature/RN50" --aug_type "None" --num_images_per_object 12 --device "cuda:0"
