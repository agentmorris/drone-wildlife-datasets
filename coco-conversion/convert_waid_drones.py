"""
Convert waid-drones dataset to COCO format.

Source: YOLO format with separate images/ and labels/ folders.
classes.txt: sheep, cattle, seal, camelus, kiang, zebra
All are mammals.
"""

import json
import os
import glob
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'waid-drones'
dataset_dir = os.path.join(DATA_ROOT, DATASET)
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
classes_file = os.path.join(dataset_dir, 'classes.txt')

CATEGORY_MAPPING = {
    'sheep': 'mammal',
    'cattle': 'mammal',
    'seal': 'mammal',
    'camelus': 'mammal',
    'kiang': 'mammal',
    'zebra': 'mammal',
}


def convert():
    # Read class names
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    class_id_to_name = {i: name for i, name in enumerate(class_names)}
    print(f'Classes: {class_id_to_name}')

    # Find all label files
    txt_files = glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True)
    print(f'Found {len(txt_files)} label files')

    # Count images on disk
    jpg_files = glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True)
    print(f'Found {len(jpg_files)} images on disk')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    n_missing = 0

    for txt_file in tqdm(txt_files, desc='Processing waid-drones'):
        # Map label path to image path
        rel_from_labels = os.path.relpath(txt_file, labels_dir)
        image_rel = os.path.splitext(rel_from_labels)[0] + '.jpg'
        image_file = os.path.join(images_dir, image_rel)

        if not os.path.isfile(image_file):
            n_missing += 1
            continue

        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            continue

        im = Image.open(image_file)
        w, h = im.size
        im.close()

        image_id += 1
        # Build path relative to data root
        image_rel_from_dataset = os.path.relpath(image_file, dataset_dir).replace('\\', '/')
        rel_path = f'{DATASET}/{image_rel_from_dataset}'

        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
        }
        images.append(img_entry)

        for line in lines:
            tokens = line.split()
            if len(tokens) != 5:
                continue

            orig_class_id = int(tokens[0])
            orig_class_name = class_id_to_name.get(orig_class_id, f'unknown_{orig_class_id}')
            category_name = CATEGORY_MAPPING.get(orig_class_name, 'other')
            cat_id = get_category_id(category_name)

            x_center_norm = float(tokens[1])
            y_center_norm = float(tokens[2])
            width_norm = float(tokens[3])
            height_norm = float(tokens[4])

            box_w = width_norm * w
            box_h = height_norm * h
            x = (x_center_norm - width_norm / 2.0) * w
            y = (y_center_norm - height_norm / 2.0) * h

            bbox = [x, y, box_w, box_h]

            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': bbox,
                'original_category': orig_class_name,
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
    if n_missing > 0:
        print(f'WARNING: {n_missing} label files had no matching image')

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco


if __name__ == '__main__':
    convert()
