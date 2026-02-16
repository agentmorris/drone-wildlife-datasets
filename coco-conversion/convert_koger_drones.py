"""
Convert koger-drones dataset to COCO format.

Source: Two COCO JSON files:
  - kenyan-ungulates/.../annotations-clean-name-pruned.json (zebra, gazelle, wbuck, buffalo, other)
  - geladas/gelada-annotations/train_males.json (adult_male, gelada, human)
  Also gelada val/test splits available.
Categories: zebra->mammal, gazelle->mammal, wbuck(waterbuck)->mammal, buffalo->mammal,
            other->other, adult_male->mammal, gelada->mammal, human->other
"""

import json
import os
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'koger-drones'
dataset_dir = os.path.join(DATA_ROOT, DATASET)

ORIGINAL_CAT_TO_CATEGORY = {
    'zebra': 'mammal',
    'gazelle': 'mammal',
    'wbuck': 'mammal',
    'buffalo': 'mammal',
    'other': 'other',
    'adult_male': 'mammal',  # gelada adult male
    'gelada': 'mammal',
    'human': 'other',
}

# Annotation files and their corresponding image roots
ANNOTATION_SETS = [
    {
        'ann_file': os.path.join(dataset_dir, 'kenyan-ungulates', 'ungulate-annotations',
                                 'annotations-clean-name-pruned', 'annotations-clean-name-pruned.json'),
        'image_root': os.path.join(dataset_dir, 'kenyan-ungulates', 'ungulate-annotations'),
        'split': None,  # no split info for this file
    },
    {
        'ann_file': os.path.join(dataset_dir, 'geladas', 'gelada-annotations', 'train_males.json'),
        'image_root': os.path.join(dataset_dir, 'geladas', 'gelada-annotations', 'annotated_images'),
        'split': 'train',
    },
    {
        'ann_file': os.path.join(dataset_dir, 'geladas', 'gelada-annotations',
                                 'coco_males_export-2022-01-05T15_54_11.401Z-val.json'),
        'image_root': os.path.join(dataset_dir, 'geladas', 'gelada-annotations', 'annotated_images'),
        'split': 'val',
    },
    {
        'ann_file': os.path.join(dataset_dir, 'geladas', 'gelada-annotations',
                                 'coco_males_export-2022-01-05T15_54_50.050Z-test.json'),
        'image_root': os.path.join(dataset_dir, 'geladas', 'gelada-annotations', 'annotated_images'),
        'split': 'test',
    },
]


def convert():
    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    # Track which abs paths we've already processed (gelada images may overlap between splits)
    processed_abs_paths = {}

    for ann_set in ANNOTATION_SETS:
        ann_file = ann_set['ann_file']
        image_root = ann_set['image_root']
        split = ann_set['split']

        if not os.path.isfile(ann_file):
            print(f'Skipping missing annotation file: {ann_file}')
            continue

        with open(ann_file, 'r') as f:
            data = json.load(f)

        orig_cat_map = {c['id']: c['name'] for c in data['categories']}

        img_id_to_anns = defaultdict(list)
        for ann in data['annotations']:
            img_id_to_anns[ann['image_id']].append(ann)

        for im in tqdm(data['images'], desc=f'Processing {os.path.basename(ann_file)}'):
            fn = im['file_name']
            full_path = os.path.join(image_root, fn)

            if not os.path.isfile(full_path):
                continue

            abs_path = os.path.abspath(full_path)

            # If we already processed this image (e.g. gelada image in multiple splits),
            # just add annotations to the existing image
            if abs_path in processed_abs_paths:
                existing_image_id = processed_abs_paths[abs_path]
                anns_for_image = img_id_to_anns.get(im['id'], [])
                for ann in anns_for_image:
                    orig_cat_name = orig_cat_map[ann['category_id']]
                    category_name = ORIGINAL_CAT_TO_CATEGORY.get(orig_cat_name, 'other')
                    cat_id = get_category_id(category_name)
                    bbox = [float(x) for x in ann['bbox']]
                    ann_id += 1
                    annotations.append({
                        'id': ann_id,
                        'image_id': existing_image_id,
                        'category_id': cat_id,
                        'bbox': bbox,
                        'original_category': orig_cat_name,
                    })
                continue

            image_id += 1
            rel_from_dataset = os.path.relpath(full_path, dataset_dir).replace('\\', '/')
            rel_path = f'{DATASET}/{rel_from_dataset}'

            img_entry = {
                'id': image_id,
                'file_name': rel_path,
                'width': im['width'],
                'height': im['height'],
            }
            if split:
                img_entry['original_split'] = split
            images.append(img_entry)

            processed_abs_paths[abs_path] = image_id

            anns_for_image = img_id_to_anns.get(im['id'], [])
            for ann in anns_for_image:
                orig_cat_name = orig_cat_map[ann['category_id']]
                category_name = ORIGINAL_CAT_TO_CATEGORY.get(orig_cat_name, 'other')
                cat_id = get_category_id(category_name)
                bbox = [float(x) for x in ann['bbox']]
                ann_id += 1
                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'original_category': orig_cat_name,
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
