import zarr
import numpy as np
import time
import logging
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .util import get_chunk_slicing

logger = logging.getLogger(__name__)


def compare_chunk_from_paths(array1_path: str, array2_path: str, slicing: Tuple[slice, slice, slice]) -> bool:
    array1 = zarr.open(array1_path, mode='r')
    array2 = zarr.open(array2_path, mode='r')
    
    chunk1 = array1[slicing]
    chunk2 = array2[slicing]
    return (chunk1 == chunk2).all()


def compare_arrays(
    array1_path: str,
    array2_path: str, 
    chunk_width: int = 256,
    max_workers: int = 4
) -> bool:
    array1 = zarr.open(array1_path, mode='r')
    chunk_slicings = list(get_chunk_slicing(array1.shape, chunk_width))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(compare_chunk_from_paths, array1_path, array2_path, sl): i
            for i, sl in enumerate(chunk_slicings)
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            chunk_index = future_to_index[future]
            is_equal = future.result(timeout=30)
            if not is_equal:
                return False
                
            completed += 1
    
    return True
