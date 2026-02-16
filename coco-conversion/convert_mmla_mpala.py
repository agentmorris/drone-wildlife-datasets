"""Convert mmla-mpala dataset (YOLO format). Categories: zebra, giraffe, onager, dog -> mammal."""

from convert_yolo_dataset import convert_yolo_dataset

CATEGORY_MAPPING = {
    'zebra': 'mammal',
    'giraffe': 'mammal',
    'onager': 'mammal',
    'dog': 'mammal',
}

if __name__ == '__main__':
    convert_yolo_dataset('mmla-mpala', CATEGORY_MAPPING)
