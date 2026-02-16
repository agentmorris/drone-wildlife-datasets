"""
Convert reinhard-savmap dataset to COCO format.

Source: Already converted to COCO format at savmap-huggingface/converted-to-coco/annotations.json.
Images at savmap-huggingface/converted-to-coco/images/.
Category: animal -> mammal (Namibian savanna wildlife)
3545 images are explicitly negative (no annotations) -> empty.
"""

import json
import os
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'reinhard-savmap'
dataset_dir = os.path.join(DATA_ROOT, DATASET)
coco_dir = os.path.join(dataset_dir, 'savmap-huggingface', 'converted-to-coco')
ann_file = os.path.join(coco_dir, 'annotations.json')


def convert():
    with open(ann_file, 'r') as f:
        data = json.load(f)

    print(f'Source: {len(data["images"])} images, {len(data["annotations"])} annotations')

    # Build image_id -> annotations mapping
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    n_empty = 0

    for im in tqdm(data['images'], desc='Processing images'):
        image_id += 1

        # file_name is like "images/image_000001.jpg"
        orig_file_name = im['file_name']
        full_path = os.path.join(coco_dir, orig_file_name)
        if not os.path.isfile(full_path):
            continue

        # Build path relative to data root
        rel_from_dataset = os.path.relpath(full_path, dataset_dir).replace('\\', '/')
        rel_path = f'{DATASET}/{rel_from_dataset}'

        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': im['width'],
            'height': im['height'],
        }
        images.append(img_entry)

        anns_for_image = img_id_to_anns.get(im['id'], [])

        if len(anns_for_image) == 0:
            # Explicitly empty image (negative sample)
            n_empty += 1
            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': get_category_id('empty'),
                'original_category': 'empty',
            })
        else:
            for ann in anns_for_image:
                cat_id = get_category_id('mammal')
                bbox = [float(x) for x in ann['bbox']]
                ann_id += 1
                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'original_category': 'animal',
                })

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES,
    }

    output_path = os.path.join(OUTPUT_DIR, f'{DATASET}.json')
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=1)

    print(f'Wrote {len(images)} images ({n_empty} empty), {len(annotations)} annotations to {output_path}')

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
