import multiprocessing
import os

import psutil

from manifoldy.definitions import TARGET_DIMENSIONALITY


def setup_multiprocessing():
    process = psutil.Process(os.getpid())
    process.cpu_affinity(range(multiprocessing.cpu_count()))


def get_instance_name(instance):
    name = instance[0][0].upper() + instance[1][0].upper()
    name += (
        ("E" if instance[2] == 0.8 else "H")
        + ("E" if instance[3] == 0.8 else "H")
        + str(TARGET_DIMENSIONALITY)
    )
    return name
