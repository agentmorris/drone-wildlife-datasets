"""
Shared converter for YOLO-format datasets (mmla-opc, mmla-wilds, mmla-mpala, waid-drones).

YOLO format: class_id x_center_norm y_center_norm width_norm height_norm
with a classes.txt file mapping class IDs to names.
Empty .txt files are treated as explicitly empty images.
"""

import json
import os
import glob
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR


def convert_yolo_dataset(dataset_name, category_mapping, classes_file=None):
    """
    Convert a YOLO-format dataset to COCO.

    Args:
        dataset_name: shortcode / folder name under DATA_ROOT
        category_mapping: dict mapping original class name -> harmonized category name
        classes_file: path to classes.txt (default: dataset_dir/classes.txt)
    """
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)

    if classes_file is None:
        classes_file = os.path.join(dataset_dir, 'classes.txt')

    # Read class names
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    class_id_to_name = {i: name for i, name in enumerate(class_names)}
    print(f'Classes: {class_id_to_name}')

    # Find all annotation txt files (exclude classes.txt and README)
    txt_files = glob.glob(os.path.join(dataset_dir, '**', '*.txt'), recursive=True)
    txt_files = [f for f in txt_files if not os.path.basename(f) in ('classes.txt',)]

    # Find all image files
    jpg_files = set()
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for f in glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True):
            jpg_files.add(os.path.abspath(f))

    print(f'Found {len(txt_files)} annotation files, {len(jpg_files)} images on disk')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    n_empty = 0
    n_missing_image = 0

    for txt_file in tqdm(txt_files, desc=f'Processing {dataset_name}'):
        # Find corresponding image
        base = os.path.splitext(txt_file)[0]
        image_file = None
        for ext in ('.jpg', '.jpeg', '.png'):
            candidate = base + ext
            if os.path.isfile(candidate):
                image_file = candidate
                break

        if image_file is None:
            n_missing_image += 1
            continue

        # Read annotation lines
        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Get image dimensions
        im = Image.open(image_file)
        w, h = im.size
        im.close()

        image_id += 1
        rel_from_dataset = os.path.relpath(image_file, dataset_dir).replace('\\', '/')
        rel_path = f'{dataset_name}/{rel_from_dataset}'

        img_entry = {
            'id': image_id,
            'file_name': rel_path,
            'width': w,
            'height': h,
        }
        images.append(img_entry)

        if len(lines) == 0:
            # Explicitly empty image
            n_empty += 1
            cat_id = get_category_id('empty')
            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'original_category': 'empty',
            })
            continue

        for line in lines:
            tokens = line.split()
            if len(tokens) != 5:
                continue

            orig_class_id = int(tokens[0])
            orig_class_name = class_id_to_name.get(orig_class_id, f'unknown_{orig_class_id}')
            category_name = category_mapping.get(orig_class_name, 'other')
            cat_id = get_category_id(category_name)

            x_center_norm = float(tokens[1])
            y_center_norm = float(tokens[2])
            width_norm = float(tokens[3])
            height_norm = float(tokens[4])

            # Convert from YOLO center format to COCO top-left format (absolute pixels)
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

    output_path = os.path.join(OUTPUT_DIR, f'{dataset_name}.json')
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=1)

    print(f'Wrote {len(images)} images ({n_empty} empty), {len(annotations)} annotations to {output_path}')
    if n_missing_image > 0:
        print(f'WARNING: {n_missing_image} annotation files had no matching image')

    image_ids_with_anns = set(a['image_id'] for a in annotations)
    images_without_anns = [img for img in images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'WARNING: {len(images_without_anns)} images have no annotations')

    return coco
