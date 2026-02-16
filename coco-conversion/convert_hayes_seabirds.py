"""
Convert hayes-seabirds dataset to COCO format.

Source: CSV files (no header) with columns: filename, x1, y1, x2, y2, label
Two sub-datasets: Albatross and Penguin, each with train/val/test splits.
Images in Albatross_LabeledTiles/ and Penguin_LabeledTiles/.
Categories: albatross -> bird, penguin -> bird
"""

import json
import os
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'hayes-seabirds'
dataset_dir = os.path.join(DATA_ROOT, DATASET)

TILES_BASE = os.path.join(dataset_dir, 'Training, Validation, and Testing Labels and Tiles')

DATASET_NAME_TO_IMAGE_FOLDER = {
    'Albatross': os.path.join(TILES_BASE, 'Albatross_LabeledTiles'),
    'Penguin': os.path.join(TILES_BASE, 'Penguin_LabeledTiles'),
}

SPECIES_TO_CATEGORY = {
    'albatross': 'bird',
    'penguin': 'bird',
}


def convert():
    csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'annotations' in f.lower()]
    print(f'Found {len(csv_files)} annotation CSV files')

    # (dataset_name, image_name) -> list of annotations
    image_annotations = defaultdict(list)
    image_split_map = {}

    for csv_file in csv_files:
        # Determine dataset name
        dataset_name = None
        for name in DATASET_NAME_TO_IMAGE_FOLDER:
            if name.lower() in csv_file.lower():
                dataset_name = name
                break
        assert dataset_name is not None, f'Cannot determine dataset for {csv_file}'

        # Determine split from filename
        csv_basename = os.path.basename(csv_file).lower()
        if 'train' in csv_basename:
            split = 'train'
        elif 'val' in csv_basename:
            split = 'val'
        elif 'test' in csv_basename:
            split = 'test'
        else:
            split = None

        df = pd.read_csv(csv_file, header=None, names=['filename', 'x1', 'y1', 'x2', 'y2', 'label'])

        for _, row in df.iterrows():
            key = (dataset_name, row['filename'])
            image_annotations[key].append({
                'label': row['label'].lower(),
                'x1': row['x1'],
                'y1': row['y1'],
                'x2': row['x2'],
                'y2': row['y2'],
            })
            if split:
                image_split_map[key] = split

    total_anns = sum(len(v) for v in image_annotations.values())
    print(f'Read {total_anns} annotations for {len(image_annotations)} images')

    # Count images on disk
    disk_count = 0
    for folder in DATASET_NAME_TO_IMAGE_FOLDER.values():
        if os.path.isdir(folder):
            disk_count += len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    print(f'Found {disk_count} images on disk')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    for (dataset_name, image_name) in tqdm(sorted(image_annotations.keys()), desc='Processing images'):
        image_folder = DATASET_NAME_TO_IMAGE_FOLDER[dataset_name]
        full_path = os.path.join(image_folder, image_name)
        if not os.path.isfile(full_path):
            continue

        # Build relative path from data root
        rel_from_dataset = os.path.relpath(full_path, dataset_dir).replace('\\', '/')
        rel_path = f'{DATASET}/{rel_from_dataset}'

        im = Image.open(full_path)
        w, h = im.size
        im.close()

        image_id += 1
        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
        }
        key = (dataset_name, image_name)
        if key in image_split_map:
            img_entry['original_split'] = image_split_map[key]
        images.append(img_entry)

        for ann_data in image_annotations[key]:
            label = ann_data['label']
            category_name = SPECIES_TO_CATEGORY.get(label, 'bird')
            cat_id = get_category_id(category_name)

            x1, y1, x2, y2 = ann_data['x1'], ann_data['y1'], ann_data['x2'], ann_data['y2']
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': bbox,
                'original_category': label,
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

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
