import zarr
import numpy as np
import time
from typing import Tuple, Iterator

def compare_arrays(
    array1_path: str,
    array2_path: str,
    chunk_width: int = 256
) -> bool:
    array1 = zarr.open(array1_path, mode='r')
    array2 = zarr.open(array2_path, mode='r')

    for sl in get_chunk_slicing(array1.shape, chunk_width):
        chunk1 = array1[sl]
        chunk2 = array2[sl]

        if (chunk1 != chunk2).any():
            return False

    return True


def get_chunk_slicing(shape: Tuple[int, int, int], chunk_width: int = 256) -> Iterator[Tuple[slice, slice, slice]]:
    """
    Generate slicings for iterating over chunks in a 3D array.
    
    Args:
        shape: 3D array shape (Z, Y, X)
        chunk_width: Size of chunks along each dimension
        
    Yields:
        Slices for (Z, Y, X) dimensions, to be used for extracting chunks from a 3D array.
    """
    shape = np.array(shape)
    grid_shape = (shape + chunk_width - 1) // chunk_width
    
    for grid_idx in np.ndindex(*grid_shape):
        starts = np.array(grid_idx) * chunk_width
        ends = np.minimum(starts + chunk_width, shape)

        z0, y0, x0 = starts
        z1, y1, x1 = ends

        yield np.s_[z0:z1, y0:y1, x0:x1]
