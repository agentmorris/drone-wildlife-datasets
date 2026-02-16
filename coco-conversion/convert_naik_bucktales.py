"""
Convert naik-bucktales dataset to COCO format.

Source: COCO JSON files (train.json, val.json, test.json) in Detection_Dataset/coco_format_v1.
Images in train_images/, val_images/, test_images/.
Original categories: drone->other, bird->bird, unknown->other, shadow->other,
                     bbfemale->mammal, bbmale->mammal
"""

import json
import os
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'naik-bucktales'
COCO_BASE = os.path.join(DATA_ROOT, DATASET, 'Detection_Dataset', 'coco_format_v1')

SPLITS = {
    'train': ('train.json', 'train_images'),
    'val': ('val.json', 'val_images'),
    'test': ('test.json', 'test_images'),
}

ORIGINAL_CAT_TO_CATEGORY = {
    'drone': 'other',
    'bird': 'bird',
    'unknown': 'other',
    'shadow': 'other',
    'bbfemale': 'mammal',
    'bbmale': 'mammal',
}


def convert():
    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    total_disk_images = 0

    for split, (json_file, image_folder) in SPLITS.items():
        json_path = os.path.join(COCO_BASE, json_file)
        image_dir = os.path.join(COCO_BASE, image_folder)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Build category map from original data
        orig_cat_map = {c['id']: c['name'] for c in data['categories']}

        # Build image_id -> annotations mapping
        img_id_to_anns = {}
        for ann in data['annotations']:
            img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

        disk_files = set(os.listdir(image_dir)) if os.path.isdir(image_dir) else set()
        total_disk_images += len([f for f in disk_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        for im in tqdm(data['images'], desc=f'Processing {split}'):
            fn = im['file_name']
            full_path = os.path.join(image_dir, fn)
            if not os.path.isfile(full_path):
                continue

            image_id += 1
            # Build relative path from data root
            rel_from_dataset = os.path.relpath(full_path, os.path.join(DATA_ROOT, DATASET)).replace('\\', '/')
            rel_path = f'{DATASET}/{rel_from_dataset}'

            img_entry = {
                'id': image_id,
                'file_name': rel_path,
                'width': im['width'],
                'height': im['height'],
                'original_split': split,
            }
            images.append(img_entry)

            anns_for_image = img_id_to_anns.get(im['id'], [])
            for ann in anns_for_image:
                orig_cat_name = orig_cat_map[ann['category_id']]
                category_name = ORIGINAL_CAT_TO_CATEGORY[orig_cat_name]
                cat_id = get_category_id(category_name)

                bbox = [float(x) for x in ann['bbox']]

                ann_id += 1
                ann_entry = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'original_category': orig_cat_name,
                }
                annotations.append(ann_entry)

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES,
    }

    output_path = os.path.join(OUTPUT_DIR, f'{DATASET}.json')
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=1)

    print(f'Wrote {len(images)} images, {len(annotations)} annotations to {output_path}')
    print(f'Total images on disk: {total_disk_images}')

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
