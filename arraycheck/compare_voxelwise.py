import zarr
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


def compare_arrays(array1_path: str, array2_path: str) -> bool:
    array1 = zarr.open(array1_path, mode='r')
    array2 = zarr.open(array2_path, mode='r')
    
    Z, Y, X = array1.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if array1[z, y, x] != array2[z, y, x]:
                    return False

    return True
