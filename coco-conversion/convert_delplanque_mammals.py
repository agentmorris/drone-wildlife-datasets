"""
Convert delplanque-mammals dataset to COCO format.

Source: COCO JSON files for train/val/test splits.
Images in train/, val/, test/ subfolders.
Categories: Alcelaphinae, Buffalo, Kob, Warthog, Waterbuck, Elephant (all -> mammal)
"""

import json
import os
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'delplanque-mammals'
dataset_dir = os.path.join(DATA_ROOT, DATASET)

ANNOTATION_FILES = {
    'train': 'groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json',
    'val': 'groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json',
    'test': 'groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json',
}

# All categories are mammals
ORIGINAL_CAT_TO_CATEGORY = {
    'alcelaphinae': 'mammal',
    'buffalo': 'mammal',
    'kob': 'mammal',
    'warthog': 'mammal',
    'waterbuck': 'mammal',
    'elephant': 'mammal',
}


def convert():
    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    for split, ann_rel_path in ANNOTATION_FILES.items():
        ann_file = os.path.join(dataset_dir, ann_rel_path)
        image_folder = os.path.join(dataset_dir, split)

        with open(ann_file, 'r') as f:
            data = json.load(f)

        orig_cat_map = {c['id']: c['name'] for c in data['categories']}

        img_id_to_anns = defaultdict(list)
        for ann in data['annotations']:
            img_id_to_anns[ann['image_id']].append(ann)

        for im in tqdm(data['images'], desc=f'Processing {split}'):
            fn = im['file_name']
            full_path = os.path.join(image_folder, fn)
            if not os.path.isfile(full_path):
                continue

            image_id += 1
            rel_path = f'{DATASET}/{split}/{fn}'

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
                category_name = ORIGINAL_CAT_TO_CATEGORY.get(orig_cat_name.lower(), 'mammal')
                cat_id = get_category_id(category_name)
                bbox = [float(x) for x in ann['bbox']]

                ann_id += 1
                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'original_category': orig_cat_name.lower(),
                })

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES,
    }

    output_path = os.path.join(OUTPUT_DIR, f'{DATASET}.json')
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=1)

    print(f'Wrote {len(images)} images, {len(annotations)} annotations to {output_path}')

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
