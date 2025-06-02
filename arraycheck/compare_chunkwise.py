import zarr
import numpy as np
import time
import logging

from .util import get_chunk_slicing

logger = logging.getLogger(__name__)


def compare_arrays(
    array1_path: str,
    array2_path: str,
    chunk_width: int = 256
) -> bool:
    array1 = zarr.open(array1_path, mode='r')
    array2 = zarr.open(array2_path, mode='r')

    chunk_count = 0
    for sl in get_chunk_slicing(array1.shape, chunk_width):
        chunk1 = array1[sl]
        chunk2 = array2[sl]

        if (chunk1 != chunk2).any():
            return False

    return True
