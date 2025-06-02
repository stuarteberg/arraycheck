import zarr
import numpy as np
import time
import logging
import hashlib
import pandas as pd
import pyarrow.feather as feather

from .util import get_chunk_slicing

logger = logging.getLogger(__name__)


def compute_chunk_hashes(
    array_path: str,
    output_path: str,
    chunk_width: int = 256
) -> None:
    array = zarr.open(array_path, mode='r')

    chunk_data = []
    for sl in get_chunk_slicing(array.shape, chunk_width):
        chunk = array[sl]
        chunk_hash = hashlib.md5(chunk.tobytes()).hexdigest()
        
        z_start, z_end = sl[0].start, sl[0].stop
        y_start, y_end = sl[1].start, sl[1].stop
        x_start, x_end = sl[2].start, sl[2].stop
        
        chunk_data.append({
            'z_start': z_start,
            'z_end': z_end,
            'y_start': y_start,
            'y_end': y_end,
            'x_start': x_start,
            'x_end': x_end,
            'hash': chunk_hash
        })

    df = pd.DataFrame(chunk_data)
    feather.write_feather(df, output_path)
