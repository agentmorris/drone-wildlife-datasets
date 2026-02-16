"""
Merge all individual dataset COCO files into a single harmonized COCO file.

Output: I:/data/drone-data/output/drone-wildlife-datasets.json
"""

import json
import os
import glob
from collections import defaultdict

from conversion_config import CATEGORIES, OUTPUT_DIR

OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'drone-wildlife-datasets.json')

# All dataset JSON files to merge (everything in output/ except the final merged file)
DATASET_FILES = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.json')))
DATASET_FILES = [f for f in DATASET_FILES if os.path.basename(f) != 'drone-wildlife-datasets.json']


def merge():
    merged_images = []
    merged_annotations = []
    next_image_id = 0
    next_ann_id = 0

    dataset_stats = []

    for dataset_file in DATASET_FILES:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]

        with open(dataset_file, 'r') as f:
            data = json.load(f)

        # Build mapping from old image IDs to new image IDs
        old_to_new_image_id = {}

        for im in data['images']:
            next_image_id += 1
            old_id = im['id']
            old_to_new_image_id[old_id] = next_image_id

            new_im = dict(im)
            new_im['id'] = next_image_id
            merged_images.append(new_im)

        for ann in data['annotations']:
            next_ann_id += 1
            new_ann = dict(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = old_to_new_image_id[ann['image_id']]
            merged_annotations.append(new_ann)

        n_images = len(data['images'])
        n_anns = len(data['annotations'])
        dataset_stats.append((dataset_name, n_images, n_anns))
        print(f'  {dataset_name}: {n_images} images, {n_anns} annotations')

    # Remove images with no annotations
    image_ids_with_anns = set(a['image_id'] for a in merged_annotations)
    orphan_images = [img for img in merged_images if img['id'] not in image_ids_with_anns]
    if orphan_images:
        print(f'\nRemoving {len(orphan_images)} images with no annotations:')
        for img in orphan_images:
            print(f'  {img["file_name"]}')
        merged_images = [img for img in merged_images if img['id'] in image_ids_with_anns]

    coco = {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': CATEGORIES,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(coco, f, indent=1)

    print(f'\nMerged {len(DATASET_FILES)} datasets')
    print(f'Total: {len(merged_images)} images, {len(merged_annotations)} annotations')
    print(f'Output: {OUTPUT_FILE}')

    # Print category distribution
    cat_counts = defaultdict(int)
    cat_id_to_name = {c['id']: c['name'] for c in CATEGORIES}
    for ann in merged_annotations:
        cat_name = cat_id_to_name[ann['category_id']]
        cat_counts[cat_name] += 1

    print('\nCategory distribution:')
    for cat_name, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f'  {cat_name}: {count}')

    # Validate: every image has at least one annotation
    image_ids_with_anns = set(a['image_id'] for a in merged_annotations)
    images_without_anns = [img for img in merged_images if img['id'] not in image_ids_with_anns]
    if images_without_anns:
        print(f'\nWARNING: {len(images_without_anns)} images have no annotations')
        # Show which datasets these belong to
        for img in images_without_anns[:5]:
            print(f'  {img["file_name"]}')

    return coco


if __name__ == '__main__':
    merge()
