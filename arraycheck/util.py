import numpy as np
import pandas as pd
import zarr
import skimage.io
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import hashlib
from typing import Tuple, Optional, Iterator

from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_test_volumes(shape: Tuple[int, int, int], dtype=np.uint16, output_dir: str = "/tmp") -> Tuple[str, str]:
    """
    Create two test zarr volumes for demonstration.
    In practice, these would be existing large datasets.
    
    Args:
        shape: 3D shape tuple (Z, Y, X)
        dtype: Data type for the arrays
        output_dir: Directory to store the zarr arrays (default: /tmp)
        
    Returns:
        Tuple of two file paths (original, backup) to the zarr arrays on disk
    """
    logger.info(f"Creating test volumes with shape {shape} and dtype {dtype}")
    logger.info(f"Writing zarr arrays to {output_dir}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create two zarr arrays with disk storage
    array1_path = output_path / "test_array_1"
    array2_path = output_path / "test_array_2"
    
    store1 = zarr.DirectoryStore(str(array1_path))
    store2 = zarr.DirectoryStore(str(array2_path))
    
    chunk_z = min(128, shape[0])
    chunk_y = min(128, shape[1])
    chunk_x = min(128, shape[2])
    chunks = (chunk_z, chunk_y, chunk_x)
    
    # Array 1: Original data
    array1 = zarr.zeros(shape, dtype=dtype, store=store1, chunks=chunks)
    
    # Array 2: "Backup" data (identical for this demo)
    array2 = zarr.zeros(shape, dtype=dtype, store=store2, chunks=chunks)
    
    # Fill with some test data
    logger.info("Filling arrays with test data...")
    test_data = np.random.randint(0, 65535, size=shape, dtype=dtype)
    array1[:] = test_data
    array2[:] = test_data
    
    logger.info(f"Test volumes created at {array1_path} and {array2_path}")
    
    return str(array1_path), str(array2_path)


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
