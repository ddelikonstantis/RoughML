import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.fft import fftfreq
import cv2
import math
import argparse


def polar_fft(image_filename):
    # load image
    img = cv2.imread(image_filename)
    # convert image to grayscale
    img = rgb2gray(img)
    # plot grayscale image and get shape
    cv2.imshow('grayscale image ' + str(img.shape), img)
    # get image rows
    rows = img.shape[0]
    # get image columns
    clmns = img.shape[1]
    # get image fft2D
    fft2d = np.fft.fft2(img)
    # plot image fft2D
    plt.imshow(np.log(abs(fft2d)), cmap='gray')
    plt.show()
    # make fourier values as column and convert complex values to real
    flat_fft2d = abs(fft2d.ravel())
    # computation of matrix indexes distance
    mid_idxrow = math.floor((rows / 2) + 1)
    mid_idxcol = math.floor((clmns / 2) + 1)
    dist = np.zeros(shape=(rows, clmns), dtype='float')
    # get matrix distances from middle index
    for i in range(rows):
            for j in range(clmns):
                dist[i][j] = math.sqrt(pow((i-mid_idxrow), 2) + pow((j-mid_idxcol), 2))
    # convert matrix distances as column
    flat_dist = dist.ravel()
    # construct new array first column -> distance and second column -> fft
    dist_fft = np.concatenate((flat_dist, flat_fft2d))

    return None

if __name__ == "__main__":
    plr = polar_fft("src/roughml/scripts/fake_00.png")