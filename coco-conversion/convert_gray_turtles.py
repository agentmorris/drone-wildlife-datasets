"""
Convert gray-turtles dataset to COCO format.

Source: CSV with columns file_location, filename, top, left, label, ImageHeight, ImageWidth.
Only rows with label == "Certain Turtle" are used (following the preview code).
Points given as top/left (no width/height).
Species: olive ridley turtle (-> reptile)
"""

import json
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'gray-turtles'
dataset_dir = os.path.join(DATA_ROOT, DATASET)
annotation_file = os.path.join(dataset_dir, 'turtle_image_metadata.csv')


def convert():
    df = pd.read_csv(annotation_file, low_memory=False)
    print(f'Read {len(df)} total rows')

    # Filter to "Certain Turtle" only, per preview code
    df_turtles = df[df['label'] == 'Certain Turtle'].copy()
    print(f'Filtered to {len(df_turtles)} "Certain Turtle" annotations')

    # Build relative path for each row
    df_turtles['rel_image'] = df_turtles.apply(
        lambda r: os.path.join(r['file_location'], r['filename']), axis=1
    )

    # Find all images on disk
    disk_images = set()
    for root, dirs, files in os.walk(dataset_dir):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel = os.path.relpath(os.path.join(root, fn), dataset_dir).replace('\\', '/')
                disk_images.add(rel)

    print(f'Found {len(disk_images)} images on disk')

    # Get unique annotated images
    annotated_images = set(df_turtles['rel_image'].str.replace('\\', '/', regex=False))

    missing_on_disk = annotated_images - disk_images
    if missing_on_disk:
        print(f'WARNING: {len(missing_on_disk)} annotated images not found on disk')
        # Try normalizing paths
        disk_images_lower = {p.lower(): p for p in disk_images}
        found = 0
        for m in list(missing_on_disk):
            if m.lower() in disk_images_lower:
                found += 1
        if found:
            print(f'  ({found} can be found with case-insensitive matching)')

    # Group annotations by image
    grouped = df_turtles.groupby('rel_image')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    for rel_image, group in tqdm(grouped, desc='Processing images'):
        rel_image_normalized = rel_image.replace('\\', '/')
        full_path = os.path.join(dataset_dir, rel_image_normalized)

        if not os.path.isfile(full_path):
            continue

        # Use ImageHeight/ImageWidth from CSV (faster than opening each image)
        row0 = group.iloc[0]
        w = int(row0['ImageWidth'])
        h = int(row0['ImageHeight'])

        image_id += 1
        rel_path = f'{DATASET}/{rel_image_normalized}'
        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
        }
        images.append(img_entry)

        for _, row in group.iterrows():
            cat_id = get_category_id('reptile')
            # top/left are point coordinates
            point = [float(row['left']), float(row['top'])]

            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'point': point,
                'original_category': 'olive ridley turtle',
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
