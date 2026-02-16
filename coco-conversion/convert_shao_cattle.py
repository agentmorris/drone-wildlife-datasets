"""
Convert shao-cattle dataset to COCO format.

Source: Tab-delimited TXT files with variable-length lines.
Format: image_name\tn_boxes\tx\ty\tw\th\tquality\tid\tid_confidence\t...
Images in Dataset1/ and Dataset2/ subfolders.
Species: cattle (-> mammal)
Boxes are in absolute pixel coordinates (x, y, w, h).
"""

import json
import os
import glob
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'shao-cattle'
dataset_dir = os.path.join(DATA_ROOT, DATASET)

DATASET_TO_IMAGE_FOLDER = {'dataset1': 'Dataset1', 'dataset2': 'Dataset2'}
BOX_COLUMNS = ['x', 'y', 'w', 'h', 'quality', 'id', 'id_confidence']
N_COLUMNS_PER_BOX = len(BOX_COLUMNS)


def convert():
    annotation_files = glob.glob(os.path.join(dataset_dir, '*.txt'))
    print(f'Found {len(annotation_files)} annotation files')

    relative_path_to_annotations = {}

    for annotation_file in annotation_files:
        image_folder = None
        for dataset_name, folder in DATASET_TO_IMAGE_FOLDER.items():
            if dataset_name in os.path.basename(annotation_file).lower():
                image_folder = folder
                break

        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        lines = [s.strip() for s in lines]

        # Skip header
        assert lines[0].startswith('image')
        lines = lines[1:]

        for s in lines:
            tokens = s.split('\t')
            # tokens[0] may contain backslash subpaths like "auto1\DJI_0001.JPG"
            image_relative_path = f'{image_folder}/{tokens[0]}'.replace('\\', '/')

            n_boxes = int(tokens[1])
            n_annotation_columns = len(tokens) - 2

            if n_annotation_columns != (N_COLUMNS_PER_BOX * n_boxes):
                continue

            boxes = []
            for i_box in range(n_boxes):
                box_start = 2 + (i_box * N_COLUMNS_PER_BOX)
                ann = {}
                for i_col, col_name in enumerate(BOX_COLUMNS):
                    ann[col_name] = int(tokens[box_start + i_col])
                boxes.append(ann)

            relative_path_to_annotations[image_relative_path] = boxes

    total_anns = sum(len(v) for v in relative_path_to_annotations.values())
    print(f'Read {total_anns} annotations for {len(relative_path_to_annotations)} images')

    # Count images on disk (recursive, since Dataset1 has subdirectories)
    disk_images = set()
    for folder in DATASET_TO_IMAGE_FOLDER.values():
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for fn in files:
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel = os.path.relpath(os.path.join(root, fn), dataset_dir).replace('\\', '/')
                        disk_images.add(rel)
    print(f'Found {len(disk_images)} images on disk')

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    for rel_path in tqdm(sorted(relative_path_to_annotations.keys()), desc='Processing images'):
        full_path = os.path.join(dataset_dir, rel_path)
        if not os.path.isfile(full_path):
            continue

        boxes = relative_path_to_annotations[rel_path]
        if len(boxes) == 0:
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
        images.append(img_entry)

        for ann_data in boxes:
            cat_id = get_category_id('mammal')
            bbox = [float(ann_data['x']), float(ann_data['y']),
                    float(ann_data['w']), float(ann_data['h'])]

            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': bbox,
                'original_category': 'cattle',
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
