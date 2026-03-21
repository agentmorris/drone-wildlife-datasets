#
# Code to render sample images and count annotations in the Wilson et al.
# Big Bird dataset.
#
# https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.70059
#
# Annotations are polygons in per-image Labelme .json files.  Labels are
# string-encoded dicts with a 'name' key for the species common name.
#

#%% Constants and imports

import ast
import os
import json
import glob
import shutil
import operator

from collections import defaultdict
from tqdm import tqdm

from megadetector.visualization import visualization_utils as visutils
from megadetector.utils import path_utils

base_folder = r'I:\data\drone-data\wilson-bigbird'

output_file_annotated = r'g:\temp\wilson_bigbird_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\wilson_bigbird_sample_image_unannotated.jpg'

assert os.path.isdir(base_folder)


#%% Notes

"""
Data is in labelme format, which looks like...

{
  "version": "5.3.0a0",
  "flags": {},
  "shapes": [
    {
      "label": "{'name': 'great egret', 'age': 'adult', 'obscured': 'no', 'class': 'aves', 'order': 'pelecaniformes', 'family': 'ardeidae', 'genus': 'ardea', 'species': 'alba', 'sex': 'monomorphic', 'posture': 'resting'}",
      "points": [
        [
          94.0,
          1453.5
        ],
        [
          101.5,
          1449.0
        ],
        [
          102.0,
          1460.0
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "polygon",
      "flags": {}
    },
    ...
}

"""


#%% Read and summarize annotations

annotation_files = sorted(glob.glob(os.path.join(base_folder, '*.json')))

print('Found {} annotation files'.format(len(annotation_files)))

filename_to_annotations = {}
species_to_count = defaultdict(int)
shape_type_to_count = defaultdict(int)
n_annotations = 0
n_empty = 0


# annotation_file = annotation_files[0]
for annotation_file in tqdm(annotation_files):

    with open(annotation_file, 'r') as f:
        d = json.load(f)

    image_path = d['imagePath']
    image_full_path = os.path.join(base_folder, image_path)
    assert os.path.isfile(image_full_path), image_full_path

    shapes = d.get('shapes', [])

    if len(shapes) == 0:
        n_empty += 1
        continue

    filename_to_annotations[image_full_path] = []

    # shape = shapes[0]
    for shape in shapes:

        label_str = shape['label']
        assert "'name':" in label_str
        info = ast.literal_eval(label_str)
        species_name = info.get('name', label_str)

        species_to_count[species_name] += 1
        n_annotations += 1
        filename_to_annotations[image_full_path].append(shape)

        shape_type_to_count[shape['shape_type']] += 1

    # ...for each shape

# ...for each annotation file

print('Read {} annotations for {} images ({} empty)'.format(
    n_annotations, len(filename_to_annotations), n_empty))

print('\n{} species:'.format(len(species_to_count)))
for species_name in sorted(species_to_count.keys()):
    print('  {}: {}'.format(species_name, species_to_count[species_name]))

# defaultdict(int, {'polygon': 8653, 'rectangle': 41337})


#%% Find an image with a bunch of annotations

image_name_to_count = {}
for image_name in filename_to_annotations:
    image_name_to_count[image_name] = len(filename_to_annotations[image_name])

# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(),
                                 key=operator.itemgetter(1), reverse=True))

sorted_annotations = list(sorted_annotations)

# Pick an image near the top (lots of annotations)
# image_full_path = sorted_annotations[10]
image_full_path = sorted_annotations[10]

assert os.path.isfile(image_full_path)

image_annotations = filename_to_annotations[image_full_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations), image_full_path))


##%% Render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []
custom_strings = []

# Build a category mapping from species names in this image
category_name_to_id = {}
for shape in image_annotations:
    label_str = shape['label']
    assert "'name':" in label_str
    info = ast.literal_eval(label_str)
    species_name = info.get('name', label_str)
    if species_name not in category_name_to_id:
        category_name_to_id[species_name] = len(category_name_to_id)

# shape = image_annotations[0]
for shape in image_annotations:

    pts = shape.get('points', [])
    # print(shape['shape_type'])

    label_str = shape['label']
    assert "'name':" in label_str
    info = ast.literal_eval(label_str)

    # Compute bounding box from polygon (or rectangle) points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)

    box_w = x1 - x0
    box_h = y1 - y0

    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[species_name]
    det['bbox'] = [x0 / image_w,
                   y0 / image_h,
                   box_w / image_w,
                   box_h / image_h]

    detection_formatted_boxes.append(det)

# ...for each shape

"""
def draw_bounding_boxes_on_file(input_file, output_file, detections, confidence_threshold=0.0,
                                detector_label_map=DEFAULT_DETECTOR_LABEL_MAP,
                                thickness=DEFAULT_BOX_THICKNESS, expansion=0,
                                colormap=DEFAULT_COLORS,
                                custom_strings=None):

"""

category_id_to_name = {v: k for k, v in category_name_to_id.items()}

visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated,
                                     detection_formatted_boxes,
                                     confidence_threshold=0.0,
                                     detector_label_map=None, # category_id_to_name,
                                     thickness=1,
                                     expansion=1)

shutil.copyfile(image_full_path, output_file_unannotated)
path_utils.open_file(output_file_annotated)
