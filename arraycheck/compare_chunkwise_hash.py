import zarr
import numpy as np
import logging
import hashlib

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

        hash1 = hashlib.md5(chunk1.tobytes()).hexdigest()
        hash2 = hashlib.md5(chunk2.tobytes()).hexdigest()

        if hash1 != hash2:
            return False

    return True
