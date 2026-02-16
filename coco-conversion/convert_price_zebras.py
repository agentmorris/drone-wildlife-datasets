"""
Convert price-zebras dataset to COCO format.

Source: Labelme JSON files in annotated_videos/round1 and round2 subfolders.
Each JSON has imagePath (relative to JSON location, starting with ../), shapes with rectangles.
Labels are like "zebra_0", "person_1", "vehicle_2".
Category prefix determines mapping: zebra->mammal, person->other, vehicle->other.
"""

import json
import os
import glob
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'price-zebras'
dataset_dir = os.path.join(DATA_ROOT, DATASET)
video_folder = os.path.join(dataset_dir, 'annotated_videos')

CATEGORY_PREFIX_TO_CATEGORY = {
    'zebra': 'mammal',
    'person': 'other',
    'vehicle': 'other',
}


def convert():
    json_files = glob.glob(os.path.join(video_folder, '**', '*.json'), recursive=True)
    print(f'Found {len(json_files)} JSON annotation files')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    n_bad_points = 0
    n_missing_image = 0
    processed_abs_paths = set()

    for json_file in tqdm(json_files, desc='Processing price-zebras'):
        with open(json_file, 'r') as f:
            d = json.load(f)

        # Resolve image path relative to the JSON file location
        image_path = d['imagePath']
        json_dir = os.path.dirname(json_file)
        image_abs = os.path.normpath(os.path.join(json_dir, image_path))

        if not os.path.isfile(image_abs):
            n_missing_image += 1
            continue

        # Skip if we've already processed this image (from a different JSON)
        abs_key = os.path.abspath(image_abs).lower()
        if abs_key in processed_abs_paths:
            continue
        processed_abs_paths.add(abs_key)

        shapes = d.get('shapes', [])
        if len(shapes) == 0:
            continue

        # Get dimensions from JSON metadata (faster than opening image)
        w = d.get('imageWidth')
        h = d.get('imageHeight')
        if w is None or h is None:
            im = Image.open(image_abs)
            w, h = im.size
            im.close()

        image_id += 1
        rel_from_dataset = os.path.relpath(image_abs, dataset_dir).replace('\\', '/')
        rel_path = f'{DATASET}/{rel_from_dataset}'

        # Determine split from path
        rel_lower = rel_from_dataset.lower()
        if 'round1' in rel_lower:
            split = 'round1'
        elif 'round2' in rel_lower:
            split = 'round2'
        else:
            split = None

        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
        }
        if split:
            img_entry['original_split'] = split
        images.append(img_entry)

        for shape in shapes:
            if shape['shape_type'] != 'rectangle':
                continue
            if len(shape['points']) != 2:
                n_bad_points += 1
                continue

            label = shape['label']
            category_prefix = label.split('_')[0].lower()
            category_name = CATEGORY_PREFIX_TO_CATEGORY.get(category_prefix, 'other')
            cat_id = get_category_id(category_name)

            x0 = shape['points'][0][0]
            y0 = shape['points'][0][1]
            x1 = shape['points'][1][0]
            y1 = shape['points'][1][1]

            # Normalize orientation
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': bbox,
                'original_category': category_prefix,
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
    if n_bad_points > 0:
        print(f'NOTE: {n_bad_points} rectangles with wrong number of points were skipped')
    if n_missing_image > 0:
        print(f'WARNING: {n_missing_image} annotation files had no matching image')

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
