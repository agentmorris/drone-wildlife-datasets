"""
Convert kabra-birds dataset to COCO format.

Source: Per-image CSV files with columns: class_id, desc, x, y, width, height.
Each CSV has a matching .jpg in the same folder.
All categories are birds.
Boxes are in absolute pixel coordinates (x, y, w, h).
"""

import json
import os
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'kabra-birds'
dataset_dir = os.path.join(DATA_ROOT, DATASET)
annotations_dir = os.path.join(dataset_dir, 'Good annotations')


def convert():
    csv_files = sorted(glob.glob(os.path.join(annotations_dir, '*.csv')))
    print(f'Found {len(csv_files)} CSV files')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    # Count images on disk
    image_files = glob.glob(os.path.join(annotations_dir, '*.jpg'))
    print(f'Found {len(image_files)} images on disk')

    for csv_file in tqdm(csv_files, desc='Processing images'):
        image_file = csv_file.replace('.csv', '.jpg')
        if not os.path.isfile(image_file):
            continue

        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue

        im = Image.open(image_file)
        w, h = im.size
        im.close()

        image_id += 1
        image_basename = os.path.basename(image_file)
        rel_path = f'{DATASET}/Good annotations/{image_basename}'

        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
        }
        images.append(img_entry)

        for _, row in df.iterrows():
            cat_id = get_category_id('bird')
            bbox = [float(row['x']), float(row['y']),
                    float(row['width']), float(row['height'])]

            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': bbox,
                'original_category': row['desc'].lower(),
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
