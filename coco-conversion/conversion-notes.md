# Drone Wildlife Datasets - COCO Conversion Notes

## Overview

Converting multiple aerial/drone wildlife datasets into a single harmonized COCO-format JSON file with five categories: **bird**, **mammal**, **reptile**, **empty**, **other**.

Output: `I:\data\drone-data\output\drone-wildlife-datasets.json`

## Usage

From the `coco-conversion/` directory:

```
python run_all.py
```

This runs all 17 individual dataset converters, then merges the per-dataset JSON files into the final output file.

## Final Output Summary

- **224,703 images** across 17 datasets
- **1,923,914 annotations**
- All images have at least one annotation
- All filenames are unique

### Category Distribution

| Category | Annotations |
|----------|------------|
| mammal   | 1,323,075  |
| bird     | 592,117    |
| other    | 3,838      |
| empty    | 3,792      |
| reptile  | 1,092      |

## Categories

| ID | Name    | Description |
|----|---------|-------------|
| 1  | bird    | All bird species (waterfowl, penguins, seabirds, etc.) |
| 2  | mammal  | All mammal species (elephants, zebras, cattle, seals, etc.) |
| 3  | reptile | Reptiles (sea turtles, etc.) |
| 4  | empty   | Explicitly empty images (no animals present, as indicated by the dataset) |
| 5  | other   | Non-wildlife (humans, vehicles, drones, shadows) and unclassifiable |

## Datasets Processed

| Shortcode | Format | Annotation Type | Notes |
|-----------|--------|-----------------|-------|
| eikelboom-savanna | CSV | boxes | |
| qian-penguins | JSON (LabelBox) | points | |
| gray-turtles | CSV | points | |
| aerial-elephants | CSV | points | |
| weinstein-birds | CSV | boxes | |
| hayes-seabirds | CSV | boxes | |
| shao-cattle | TXT | boxes | |
| naik-bucktales | COCO/YOLO | boxes | |
| koger-drones | COCO | boxes | |
| kabra-birds | CSV | boxes | |
| mmla-opc | YOLO | boxes | |
| mmla-wilds | YOLO | boxes | |
| mmla-mpala | YOLO | boxes | |
| waid-drones | YOLO | boxes | |
| delplanque-mammals | COCO | boxes | |
| reinhard-savmap | Parquet (HF) | boxes | Using Hugging Face version, not Zenodo version |
| price-zebras | JSON (Labelme) | boxes | |

## Datasets Skipped

| Shortcode | Reason |
|-----------|--------|
| hu-thermal | Thermal imagery is out of scope for this exercise |
| hodgson-counts | Count-only annotations (no spatial annotations) are out of scope |
| steller-sea-lion-count | Count-only annotations (no spatial annotations) are out of scope |
| right-whale-recognition | Individual-ID-only annotations (no spatial annotations) are out of scope |
| conservation-drones | Thermal imagery is out of scope for this exercise |
| kabr-behavior | Behavior-label-only annotations (no spatial annotations) are out of scope |

## Datasets Deferred

These datasets will be processed later; they are currently on different hard drives.

| Shortcode | Notes |
|-----------|-------|
| aerial-seabirds-west-africa | |
| nm-waterfowl | |
| noaa-arctic-seals | |
| weiser-waterfowl-lila | |

## TODO

- **qian-penguins**: 738 images found on disk (README says 753), but only 560 have annotations (137365 total annotations match README). 178 images on disk (24%) have no annotations. Currently we include only the 560 annotated images. Need to investigate whether the unannotated images are empty or from a different subset.
- **shao-cattle**: 670 images on disk, 663 in annotation files. Of those, 340 images have 0 boxes (explicitly empty), 7 have malformed annotation columns, and 323 have actual cattle annotations (1919 boxes). Currently only including the 323 annotated images. The 340 zero-box images could be added as "empty" if desired.
- **gray-turtles**: 1059 images on disk, all appear in the CSV, but only 357 have "Certain Turtle" labels (1092 annotations). The other 702 images only have label "0" (no turtle). README says 1902 point annotations, but the preview code only uses "Certain Turtle" (1092). The 702 no-turtle images are not included in the COCO file. Need to visually verify that those 702 images are truly empty.

## Dataset-Specific Notes

### price-zebras

1 image (`round2/video_2013/frame_002333.jpg`) was excluded from the merged output because its only annotation had a malformed rectangle (3 corner points instead of 2). 4 total annotations across the dataset had this issue.

### reinhard-savmap

Two versions exist (Zenodo with geojson polygons, Hugging Face with Parquet/boxes). We are using the Hugging Face version located at `reinhard-savmap/savmap-huggingface/`. The Zenodo version at `reinhard-savmap/savmap-zenodo/` is not processed.

