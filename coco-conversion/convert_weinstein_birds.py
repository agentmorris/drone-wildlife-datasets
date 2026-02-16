"""
Convert weinstein-birds dataset to COCO format.

Source: Multiple CSV files in subdirectories, each with columns:
  image_path, xmin, ymin, xmax, ymax, label
Images are in subdirectories named by sub-dataset (everglades, hayes, etc.).
Each subdirectory has train/test CSV files.
Species: bird (-> bird)
"""

import json
import os
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'weinstein-birds'
dataset_dir = os.path.join(DATA_ROOT, DATASET)


def convert():
    csv_files = glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True)
    print(f'Found {len(csv_files)} CSV files')

    # Read all annotations, tracking split info from filename
    image_annotations = defaultdict(list)
    image_split = {}

    for csv_file in csv_files:
        sub_dataset = os.path.basename(os.path.dirname(csv_file))
        csv_name = os.path.basename(csv_file).lower()

        # Determine split from filename
        if 'train' in csv_name:
            split = 'train'
        elif 'test' in csv_name:
            split = 'test'
        elif 'val' in csv_name:
            split = 'val'
        else:
            split = None

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            rel_path = f'{sub_dataset}/{row["image_path"]}'
            label = row['label'].lower()

            image_annotations[rel_path].append({
                'label': label,
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
            })
            if split:
                image_split[rel_path] = split

    total_anns = sum(len(v) for v in image_annotations.values())
    print(f'Read {total_anns} annotations for {len(image_annotations)} images')

    # Count images on disk
    disk_images = set()
    for root, dirs, files in os.walk(dataset_dir):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                rel = os.path.relpath(os.path.join(root, fn), dataset_dir).replace('\\', '/')
                disk_images.add(rel)

    print(f'Found {len(disk_images)} images on disk')

    annotated_set = set(image_annotations.keys())
    missing_on_disk = annotated_set - disk_images
    if missing_on_disk:
        print(f'WARNING: {len(missing_on_disk)} annotated images not found on disk')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    for rel_path in tqdm(sorted(image_annotations.keys()), desc='Processing images'):
        full_path = os.path.join(dataset_dir, rel_path)
        if not os.path.isfile(full_path):
            continue

        im = Image.open(full_path)
        w, h = im.size
        im.close()

        image_id += 1
        img_entry = {
            'id': image_id,
            'file_name': f'{DATASET}/{rel_path}',
            'width': w,
            'height': h,
        }
        if rel_path in image_split:
            img_entry['original_split'] = image_split[rel_path]
        images.append(img_entry)

        for ann_data in image_annotations[rel_path]:
            label = ann_data['label']
            cat_id = get_category_id('bird')

            x1, y1, x2, y2 = ann_data['xmin'], ann_data['ymin'], ann_data['xmax'], ann_data['ymax']
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
