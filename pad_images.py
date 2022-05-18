import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    train_path = '../instance_version/train/'
    test_path = '../instance_version/val/'

    train_fname = os.listdir(train_path)
    test_fname = os.listdir(test_path)

    max_x = 0
    max_y = 0

    for count, i in enumerate(train_fname):
        im = plt.imread(os.path.join(train_path, i))
        x, y, _ = im.shape
        max_x = max(x, max_x)
        max_y = max(y, max_y)
        print('train max: ', count, '/', len(train_fname))

    for count, i in enumerate(test_fname):
        im = plt.imread(os.path.join(test_path, i))
        x, y, _ = im.shape
        max_x = max(x, max_x)
        max_y = max(y, max_y)
        print('val max: ', count, '/', len(test_fname))

    os.makedirs('../instance_version_padded/train/')
    os.makedirs('../instance_version_padded/val/')
    for count, i in enumerate(train_fname):
        im = plt.imread(os.path.join(train_path, i))
        x, y, _ = im.shape
        x_pad, y_pad = max_x - x, max_y - y
        padded = np.pad(im, ((0, x_pad), (0, y_pad), (0,0)), mode='constant', constant_values=((0,0), (0,0), (0,0)))
        plt.imsave(os.path.join('../instance_version_padded/train/', i), padded)
        print('train copy: ', count, '/', len(train_fname))

    for count, i in enumerate(test_fname):
        im = plt.imread(os.path.join(test_path, i))
        x, y, _ = im.shape
        x_pad, y_pad = max_x - x, max_y - y
        padded = np.pad(im, ((0, x_pad), (0, y_pad), (0,0)), mode='constant', constant_values=((0,0), (0,0), (0,0)))
        plt.imsave(os.path.join('../instance_version_padded/val/', i), padded)
        print('test copy: ', count, '/', len(test_fname))

if __name__ == '__main__':
    ## main()

    f_list = os.listdir('../instance_version/train/')
    s = set()
    for i in f_list:
        im = plt.imread(os.path.join('../instance_version/train', i))
        s.add(im.shape)

    print(s)