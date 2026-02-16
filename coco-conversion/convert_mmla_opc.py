"""Convert mmla-opc dataset (YOLO format). Categories: zebra -> mammal."""

from convert_yolo_dataset import convert_yolo_dataset

CATEGORY_MAPPING = {
    'zebra': 'mammal',
}

if __name__ == '__main__':
    convert_yolo_dataset('mmla-opc', CATEGORY_MAPPING)
