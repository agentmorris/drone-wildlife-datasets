"""
Convert eikelboom-savanna dataset to COCO format.

Source: CSV with columns FILE, x1, y1, x2, y2, SPECIES
Images in train/val/test subfolders.
Species: Zebra, Elephant, Giraffe (all -> mammal)
"""

import json
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'eikelboom-savanna'
SPLITS = ['train', 'val', 'test']

dataset_dir = os.path.join(DATA_ROOT, DATASET)
annotation_file = os.path.join(dataset_dir, 'annotations_images.csv')

SPECIES_TO_CATEGORY = {
    'zebra': 'mammal',
    'elephant': 'mammal',
    'giraffe': 'mammal',
}


def convert():
    df = pd.read_csv(annotation_file)
    print(f'Read {len(df)} annotation rows')

    # Build image name -> split mapping
    image_name_to_split = {}
    for split in SPLITS:
        split_dir = os.path.join(dataset_dir, split)
        for fn in os.listdir(split_dir):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_name_to_split[fn] = split

    print(f'Found {len(image_name_to_split)} images on disk')

    # Get unique image names from annotations
    annotated_images = set(df['FILE'].unique())
    disk_images = set(image_name_to_split.keys())

    missing_on_disk = annotated_images - disk_images
    missing_in_annotations = disk_images - annotated_images

    if missing_on_disk:
        print(f'WARNING: {len(missing_on_disk)} annotated images not found on disk')
    if missing_in_annotations:
        print(f'NOTE: {len(missing_in_annotations)} images on disk have no annotations')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    # Group annotations by filename
    grouped = df.groupby('FILE')

    for image_name, group in tqdm(grouped, desc='Processing images'):
        if image_name not in image_name_to_split:
            continue

        split = image_name_to_split[image_name]
        rel_path = f'{DATASET}/{split}/{image_name}'
        full_path = os.path.join(dataset_dir, split, image_name)

        im = Image.open(full_path)
        w, h = im.size
        im.close()

        image_id += 1
        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
            'original_split': split,
        }
        images.append(img_entry)

        for _, row in group.iterrows():
            species = row['SPECIES'].lower()
            category_name = SPECIES_TO_CATEGORY[species]
            cat_id = get_category_id(category_name)

            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': bbox,
                'original_category': species,
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
