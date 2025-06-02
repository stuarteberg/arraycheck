import logging
import numpy as np

from arraycheck.util import create_test_volumes
from arraycheck.naive_comparison import compare_arrays


def main():
    logging.basicConfig(level=logging.INFO)
    array1_path, array2_path = create_test_volumes((128, 500, 500), dtype=np.uint16)
    print(compare_arrays(array1_path, array2_path, chunk_width=128))


if __name__ == "__main__":
    main()
