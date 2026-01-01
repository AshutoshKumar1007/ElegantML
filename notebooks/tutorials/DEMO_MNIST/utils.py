# utils.py
import struct
import numpy as np
import os
def load_images(filename,mean = None,std = None):
    with open(filename, 'rb') as f:
        # Read the header
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols)
        # standardizing the image
        images = images/255.0
        if mean is None:
            mean = np.mean(images,0)
        if std is None:
            std = np.std(images,0)
        std[std == 0] = 1
        
        images = (images - mean) / std
    return images,mean,std

def load_labels(filename):
    num_classes = 10
    with open(filename, 'rb') as f:
        # Read the header
        magic, num = struct.unpack(">II", f.read(8))
        # Read label data
        labels = np.fromfile(f, dtype=np.uint8)
        labels = np.eye(num_classes)[labels]
    return labels

# Example usage
# train_images,Mean,Std = load_images(CURR_DIR + r'\train-images-idx3-ubyte\train-images-idx3-ubyte')
# train_labels = load_labels(CURR_DIR+ r'\train-labels-idx1-ubyte\train-labels-idx1-ubyte')
# print(Mean.shape,Std.shape)
