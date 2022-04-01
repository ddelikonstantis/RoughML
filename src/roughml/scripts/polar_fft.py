from matplotlib import projections
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import math
import argparse
import random


def load_image(image_filename):
    # read image
    img = cv2.imread(image_filename)
    # convert to grayscale
    img = rgb2gray(img)
    # plot grayscale image and get shape
    cv2.imshow('grayscale image ' + str(img.shape), img)

    return img

def get_polar_fft(img):
    # convert to pixel values
    img = (((img - img.min()) / (img.max() - img.min())) * 255.9)
    img = np.array(img, dtype=np.uint8)
    # get image dimensions
    rows, clmns = img.shape[0], img.shape[1]
    # get image fft2D, shift sums to the center and plot
    fft2d = np.fft.fftshift(np.fft.fft2(img))
    # get fft2d mean column values
    fft2d_mean_col = abs(np.mean(fft2d, axis=0))
    fft2d_mean_col_half = fft2d_mean_col[64:128]
    print('fft2d_mean_col: ','\n', fft2d_mean_col, '\n', fft2d_mean_col.shape, '\n')
    # get fft2d mean row values
    fft2d_mean_row = abs(np.mean(fft2d, axis=1))
    fft2d_mean_row_half = fft2d_mean_row[64:128]
    print('fft2d_mean_row: ','\n', fft2d_mean_row, '\n', fft2d_mean_row.shape, '\n')
    print('fft2d: ','\n', fft2d, '\n', fft2d.shape, '\n')
    plt.imshow(np.log(abs(fft2d)), cmap='gray')
    plt.show()
    # get absolute fourier values and flatten array
    flat_fft2d = abs(fft2d.ravel(order='F'))
    print('flat_fft2d: ','\n', flat_fft2d, '\n', flat_fft2d.shape, '\n')
    # get image center points
    mid_idxrow, mid_idxcol = math.floor(rows / 2), math.floor(clmns / 2)
    dist = np.zeros(shape=(rows, clmns), dtype='float')
    # get image distances from center point
    for i in range(rows):
            for j in range(clmns):
                dist[i][j] = math.sqrt(pow((i-mid_idxrow), 2) + pow((j-mid_idxcol), 2))
    print('dist: ','\n', dist, '\n', dist.shape, '\n')
    # convert matrix distances as 1d array
    flat_dist = dist.ravel(order='F')
    print('flat_dist: ','\n', flat_dist, '\n', flat_dist.shape, '\n')
    # construct 2d array (first column: distances from center point, second column: corresponding fft values)
    dist_fft = np.vstack((flat_dist, flat_fft2d)).T
    print('dist fft: ', '\n', dist_fft, '\n', dist_fft.shape, '\n')
    # Sort 2d array by distances from center point
    dist_fft_sorted = dist_fft[dist_fft[:,0].argsort()]
    print('dist_fft_sorted: ','\n', dist_fft_sorted, '\n')
    # get polar fft for surface
    minVal, maxVal = float(0.0), float(1.0)
    if rows < clmns:
        cycle_radius = math.floor(rows / 2)
    else:
        cycle_radius = math.floor(clmns / 2)
    polar_fft = np.zeros(shape=(cycle_radius, 2), dtype='float')
    step = int(1)
    for j in range(0, cycle_radius, step):
        sum_dis = float(0.0)
        sum_fft = float(0.0)
        count = int(0)
        for i in range(dist_fft.shape[0]):
            if (dist_fft_sorted[i][0] > minVal) and (dist_fft_sorted[i][0] <= maxVal):
                sum_dis= sum_dis + dist_fft_sorted[i][0]
                sum_fft= sum_fft + dist_fft_sorted[i][1]
                count = count + 1
        polar_fft[j][0] = sum_dis / count
        polar_fft[j][1] = sum_fft / count
        minVal = maxVal
        maxVal = maxVal + step
    print('polar_fft: ','\n', polar_fft, '\n')

    return polar_fft, fft2d_mean_row_half, fft2d_mean_col_half

def plot(x, y, title, xlabel, ylabel, axiscale, color=None):
    plt.xscale(axiscale)
    plt.yscale(axiscale)
    plt.plot(x, y)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    # argument parser
    # parser = argparse.ArgumentParser(description = 'Image polar fft')
    # parser.add_argument('image1', help = 'directory of first image to compare')
    # parser.add_argument('image2', help = 'directory of second image to compare', action='store_false')
    # parser.add_argument('image3', help = 'directory of third image to compare', action='store_false')
    # parser.add_argument('image4', help = 'directory of fourth image to compare', action='store_false')
    # parser.add_argument('image5', help = 'directory of fifth image to compare', action='store_false')
    # args = parser.parse_args()

    img = load_image("src/roughml/scripts/fake_00.png")
    # img = np.array([[9, 12, 24], [30, 2, 7], [20, 11, 14]])
    polar_fft, mean_row, mean_col = get_polar_fft(img)

    xlabel = "Spatial frequency (nm^{-1})"
    ylabel="Fourier amplitude (nm^{-1})"
    axiscale='log'
    fig1=plot(polar_fft[:,0], polar_fft[:,1], "polar fft", xlabel, ylabel, axiscale)
    fig2=plot(polar_fft[:,0], mean_row, "mean rows fft", xlabel, ylabel, axiscale)
    fig3=plot(polar_fft[:,0], mean_col, "mean columns fft", xlabel, ylabel, axiscale)
    