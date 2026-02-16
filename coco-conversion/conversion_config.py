"""
Shared configuration for COCO conversion scripts.
"""

import os

DATA_ROOT = r'I:\data\drone-data'
OUTPUT_DIR = os.path.join(DATA_ROOT, 'output')

CATEGORIES = [
    {'id': 1, 'name': 'bird'},
    {'id': 2, 'name': 'mammal'},
    {'id': 3, 'name': 'reptile'},
    {'id': 4, 'name': 'empty'},
    {'id': 5, 'name': 'other'},
]

CATEGORY_NAME_TO_ID = {c['name']: c['id'] for c in CATEGORIES}


def get_category_id(name):
    return CATEGORY_NAME_TO_ID[name]
