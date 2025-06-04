import zarr
import numpy as np
import time


def compare_arrays(array1_path: str, array2_path: str) -> bool:
    array1 = zarr.open(array1_path, mode='r')
    array2 = zarr.open(array2_path, mode='r')
    
    data1 = array1[:]
    data2 = array2[:]
    return (data1 == data2).all()


data1.shape == (1000, 2000, 3000)

mask = (data1 .== data2)

mask.shape == (1000, 2000, 3000)

