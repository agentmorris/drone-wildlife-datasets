"""
Convert aerial-elephants dataset to COCO format.

Source: CSV files (no header) with columns: image_id, x, y
image_id is the filename stem (no extension). Images in training_images/ and test_images/.
Species: elephant (-> mammal)
Annotation type: points
"""

import json
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'aerial-elephants'
dataset_dir = os.path.join(DATA_ROOT, DATASET)

ANNOTATION_FILES = {
    'training_elephants.csv': 'train',
    'test_elephants.csv': 'test',
}

IMAGE_FOLDERS = {
    'train': 'training_images',
    'test': 'test_images',
}


def convert():
    # Map image stems to (split, relative path)
    image_stem_to_info = {}
    total_disk_images = 0
    for split, folder in IMAGE_FOLDERS.items():
        folder_path = os.path.join(dataset_dir, folder)
        for fn in os.listdir(folder_path):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = os.path.splitext(fn)[0]
                image_stem_to_info[stem] = {
                    'split': split,
                    'rel_path': f'{DATASET}/{folder}/{fn}',
                    'full_path': os.path.join(folder_path, fn),
                }
                total_disk_images += 1

    print(f'Found {total_disk_images} images on disk')

    # Read annotation files
    all_annotations = defaultdict(list)
    image_stem_to_split = {}
    total_anns = 0

    for ann_file, split in ANNOTATION_FILES.items():
        df = pd.read_csv(os.path.join(dataset_dir, ann_file), header=None,
                         names=['image_id', 'x', 'y'])
        for _, row in df.iterrows():
            stem = row['image_id']
            all_annotations[stem].append({'x': row['x'], 'y': row['y']})
            image_stem_to_split[stem] = split
            total_anns += 1

    print(f'Read {total_anns} annotations for {len(all_annotations)} images')

    # Check coverage
    annotated_stems = set(all_annotations.keys())
    disk_stems = set(image_stem_to_info.keys())
    missing_on_disk = annotated_stems - disk_stems
    missing_in_annotations = disk_stems - annotated_stems

    if missing_on_disk:
        print(f'WARNING: {len(missing_on_disk)} annotated images not found on disk')
    if missing_in_annotations:
        print(f'NOTE: {len(missing_in_annotations)} images on disk have no annotations')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    for stem in tqdm(sorted(all_annotations.keys()), desc='Processing images'):
        if stem not in image_stem_to_info:
            continue

        info = image_stem_to_info[stem]
        full_path = info['full_path']

        im = Image.open(full_path)
        w, h = im.size
        im.close()

        image_id += 1
        img_entry = {
            'id': image_id,
            'file_name': info['rel_path'],
            'width': w,
            'height': h,
            'original_split': info['split'],
        }
        images.append(img_entry)

        for ann_data in all_annotations[stem]:
            cat_id = get_category_id('mammal')
            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'point': [float(ann_data['x']), float(ann_data['y'])],
                'original_category': 'elephant',
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

    # Validation
    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
