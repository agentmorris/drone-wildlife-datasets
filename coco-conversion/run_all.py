"""
Run all dataset conversions and merge into a single COCO file.

Usage: cd into the coco-conversion/ directory and run:
    python run_all.py
"""

import time

from convert_eikelboom_savanna import convert as convert_eikelboom_savanna
from convert_qian_penguins import convert as convert_qian_penguins
from convert_gray_turtles import convert as convert_gray_turtles
from convert_aerial_elephants import convert as convert_aerial_elephants
from convert_weinstein_birds import convert as convert_weinstein_birds
from convert_hayes_seabirds import convert as convert_hayes_seabirds
from convert_shao_cattle import convert as convert_shao_cattle
from convert_naik_bucktales import convert as convert_naik_bucktales
from convert_koger_drones import convert as convert_koger_drones
from convert_kabra_birds import convert as convert_kabra_birds
from convert_waid_drones import convert as convert_waid_drones
from convert_delplanque_mammals import convert as convert_delplanque_mammals
from convert_reinhard_savmap import convert as convert_reinhard_savmap
from convert_price_zebras import convert as convert_price_zebras
from convert_yolo_dataset import convert_yolo_dataset
from convert_mmla_opc import CATEGORY_MAPPING as MMLA_OPC_CATS
from convert_mmla_wilds import CATEGORY_MAPPING as MMLA_WILDS_CATS
from convert_mmla_mpala import CATEGORY_MAPPING as MMLA_MPALA_CATS
from merge_datasets import merge

CONVERTERS = [
    ('eikelboom-savanna', convert_eikelboom_savanna),
    ('qian-penguins', convert_qian_penguins),
    ('gray-turtles', convert_gray_turtles),
    ('aerial-elephants', convert_aerial_elephants),
    ('weinstein-birds', convert_weinstein_birds),
    ('hayes-seabirds', convert_hayes_seabirds),
    ('shao-cattle', convert_shao_cattle),
    ('naik-bucktales', convert_naik_bucktales),
    ('koger-drones', convert_koger_drones),
    ('kabra-birds', convert_kabra_birds),
    ('mmla-opc', lambda: convert_yolo_dataset('mmla-opc', MMLA_OPC_CATS)),
    ('mmla-wilds', lambda: convert_yolo_dataset('mmla-wilds', MMLA_WILDS_CATS)),
    ('mmla-mpala', lambda: convert_yolo_dataset('mmla-mpala', MMLA_MPALA_CATS)),
    ('waid-drones', convert_waid_drones),
    ('delplanque-mammals', convert_delplanque_mammals),
    ('reinhard-savmap', convert_reinhard_savmap),
    ('price-zebras', convert_price_zebras),
]


def run_all():
    t0 = time.time()
    for name, convert_fn in CONVERTERS:
        print(f'\n{"="*60}')
        print(f'Converting: {name}')
        print(f'{"="*60}')
        convert_fn()

    print(f'\n{"="*60}')
    print(f'Merging all datasets')
    print(f'{"="*60}')
    merge()

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f'\nDone in {minutes}m {seconds:.1f}s')


if __name__ == '__main__':
    run_all()
