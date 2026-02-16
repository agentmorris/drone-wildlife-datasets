"""
Convert qian-penguins dataset to COCO format.

Source: LabelBox JSON files with tiny bounding boxes (effectively points).
Images in Jack/Luke/Maisie/Thomas subfolders.
Species: penguin (-> bird)
"""

import json
import os
import glob
from PIL import Image
from tqdm import tqdm

from conversion_config import CATEGORIES, get_category_id, DATA_ROOT, OUTPUT_DIR

DATASET = 'qian-penguins'
SUBFOLDERS = ['Jack', 'Luke', 'Maisie', 'Thomas']

dataset_dir = os.path.join(DATA_ROOT, DATASET)

SPECIES_TO_CATEGORY = {
    'penguin': 'bird',
}


def convert():
    # Map image filenames to subfolders
    image_to_subfolder = {}
    for subfolder in SUBFOLDERS:
        folder_path = os.path.join(dataset_dir, subfolder)
        for fn in os.listdir(folder_path):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_to_subfolder[fn] = subfolder

    print(f'Found {len(image_to_subfolder)} images on disk')

    # Read all JSON annotation files
    json_files = glob.glob(os.path.join(dataset_dir, '*.json'))
    filename_to_annotations = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            records = json.load(f)

        for rec in records:
            image_filename = rec['External ID']
            label = rec['Label']
            if not label:
                continue
            objects = label.get('objects', [])
            if image_filename not in filename_to_annotations:
                filename_to_annotations[image_filename] = []
            for obj in objects:
                bbox = obj['bbox']
                # Convert tiny box to center point
                cx = bbox['left'] + bbox['width'] / 2.0
                cy = bbox['top'] + bbox['height'] / 2.0
                filename_to_annotations[image_filename].append({
                    'species': obj['value'],
                    'point': [cx, cy],
                })

    total_anns = sum(len(v) for v in filename_to_annotations.values())
    print(f'Read {total_anns} annotations for {len(filename_to_annotations)} images')

    # Check coverage
    annotated_images = set(filename_to_annotations.keys())
    disk_images = set(image_to_subfolder.keys())
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

    for image_filename in tqdm(sorted(filename_to_annotations.keys()), desc='Processing images'):
        if image_filename not in image_to_subfolder:
            continue

        subfolder = image_to_subfolder[image_filename]
        rel_path = f'{DATASET}/{subfolder}/{image_filename}'
        full_path = os.path.join(dataset_dir, subfolder, image_filename)

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
        images.append(img_entry)

        for ann_data in filename_to_annotations[image_filename]:
            species = ann_data['species']
            category_name = SPECIES_TO_CATEGORY.get(species, 'bird')
            cat_id = get_category_id(category_name)

            ann_id += 1
            ann_entry = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'point': ann_data['point'],
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
