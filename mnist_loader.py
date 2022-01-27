import struct
import numpy as np

def read_labels(labels_fn):
    with open(labels_fn, 'rb') as f:
        data = f.read()
    header_format = '>ii'
    magic, num_labels = struct.unpack_from(header_format, data)
    assert magic == 2049
    offset = struct.calcsize(header_format)
    labels = struct.unpack_from(f'>{num_labels}B', data, offset=offset)
    return np.array(labels)

def read_images(images_fn):
    with open(images_fn, 'rb') as f:
        data = f.read()
    header_format = '>iiii'
    magic, num_images, num_rows, num_columns = struct.unpack_from(header_format, data)
    assert magic == 2051
    offset = struct.calcsize(header_format)
    pixels = struct.unpack_from(f'>{num_images * num_rows * num_columns}B', data, offset=offset)
    img_matrices = np.array(pixels).reshape((num_images, num_rows, num_columns))
    return np.moveaxis(img_matrices, 0, -1)  # shape (num_rows, num_columns, num_images)
