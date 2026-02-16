"""
Pick 3 random images per dataset from the merged COCO file, render annotations,
and create an index.html for visual inspection.

Usage: python visualize_samples.py
"""

import json
import os
import random
from collections import defaultdict
from PIL import Image, ImageDraw

from conversion_config import DATA_ROOT, OUTPUT_DIR

MERGED_FILE = os.path.join(OUTPUT_DIR, 'drone-wildlife-datasets.json')
SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'sample_images')
SAMPLES_PER_DATASET = 3

CATEGORY_COLORS = {
    'bird': (0, 200, 255),
    'mammal': (255, 100, 0),
    'reptile': (0, 255, 100),
    'empty': (180, 180, 180),
    'other': (255, 255, 0),
}


def main():
    random.seed(42)

    print('Loading merged COCO file...')
    with open(MERGED_FILE, 'r') as f:
        coco = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # Build image_id -> annotations
    img_id_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    # Group images by dataset (first path component)
    dataset_to_images = defaultdict(list)
    for img in coco['images']:
        dataset = img['file_name'].split('/')[0]
        dataset_to_images[dataset].append(img)

    # Pick random samples
    selected = []
    for dataset in sorted(dataset_to_images.keys()):
        imgs = dataset_to_images[dataset]
        k = min(SAMPLES_PER_DATASET, len(imgs))
        selected.extend((dataset, img) for img in random.sample(imgs, k))

    os.makedirs(SAMPLE_DIR, exist_ok=True)

    html_entries = []
    current_dataset = None

    for dataset, img in selected:
        file_name = img['file_name']
        image_path = os.path.join(DATA_ROOT, file_name)
        if not os.path.isfile(image_path):
            print(f'  MISSING: {image_path}')
            continue

        im = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(im)

        anns = img_id_to_anns.get(img['id'], [])
        for ann in anns:
            cat_name = cat_id_to_name.get(ann['category_id'], 'other')
            color = CATEGORY_COLORS.get(cat_name, (255, 255, 255))

            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            elif 'point' in ann:
                px, py = ann['point']
                r = 4
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=color)

        out_name = file_name.replace('/', '_').replace('\\', '_')
        out_path = os.path.join(SAMPLE_DIR, out_name)
        im.save(out_path)
        print(f'  {out_name}')

        if dataset != current_dataset:
            html_entries.append(f'<h2>{dataset}</h2>')
            current_dataset = dataset

        html_entries.append(
            f'<div>'
            f'<p><code>{file_name}</code></p>'
            f'<img src="{out_name}" style="width:1000px;">'
            f'</div>'
        )

    # Write index.html
    html = (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
        '<title>Sample Images</title>\n'
        '<style>body{font-family:sans-serif;margin:20px;background:#111;color:#eee;}'
        'img{display:block;margin:8px 0 24px 0;} code{color:#8cf;}</style>\n'
        '</head><body>\n'
        '<h1>Drone Wildlife Datasets - Sample Images</h1>\n'
        + '\n'.join(html_entries)
        + '\n</body></html>'
    )
    index_path = os.path.join(SAMPLE_DIR, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html)

    print(f'\nWrote {len([e for e in html_entries if "<img" in e])} sample images to {SAMPLE_DIR}')
    print(f'Open {index_path} in a browser to inspect.')


if __name__ == '__main__':
    main()
