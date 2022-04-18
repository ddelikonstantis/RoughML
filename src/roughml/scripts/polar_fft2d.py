from genericpath import exists
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import argparse
from pathlib import Path
import os, errno
import sys


def polar_fft2d(*args):
    for iter, arg in enumerate(args):
        # get image alpha value
        alpha, alpha_val = None, ["0.5","0.6","0.7","0.8","0.9","1.0"]
        for a in alpha_val:
            if arg.find(str(a)) != -1:
                alpha = a
                break
        if alpha == None:
            print("Alpha cannot be established in loaded image")
            sys.exit()
        # read image as grayscale
        img = cv2.imread(arg, 0)
        # get image dimension
        rows, clmns = img.shape[0], img.shape[1]
        # get image FFT2D and shift sums to the center
        fft2d = np.fft.fftshift(np.fft.fft2(img))
        # get absolute fourier values and flatten array
        flat_fft2d = abs(fft2d.ravel(order='F'))
        # get image center points
        mid_idxrow, mid_idxcol = math.floor(rows / 2), math.floor(clmns / 2)
        # get FFT2D mean absolute column values
        fft2d_mean_col = np.mean(abs(fft2d), axis=0)
        # keep only second half of the mean FFT matrices
        fft2d_mean_col_half = fft2d_mean_col[mid_idxcol:clmns]
        # get FFT2D mean absolute row values
        fft2d_mean_row = np.mean(abs(fft2d), axis=1)
        # keep only second half of the mean FFT matrices
        fft2d_mean_row_half = fft2d_mean_row[mid_idxrow:rows]
        dist = np.zeros(shape=(rows, clmns), dtype='float')
        # get indexes of the middle element of a matrix
        for i in range(rows):
                for j in range(clmns):
                    dist[i][j] = math.sqrt(pow((i-mid_idxrow), 2) + pow((j-mid_idxcol), 2))
        # convert distance matrix as a column matrix
        flat_dist = dist.ravel(order='F')
        # construct 2d array (first column: distances from center point, second column: corresponding fft values)
        dist_fft = np.vstack((flat_dist, flat_fft2d)).T
        # Sort 2d array based on 1st column
        dist_fft_sorted = dist_fft[dist_fft[:,0].argsort()]
        # get polar fft for surface
        minVal, maxVal = float(0.0), float(1.0)
        # get loop cycle depending on lowest dimension
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
            # get radial sums of distance and fft2d
            for i in range(dist_fft.shape[0]):
                if (dist_fft_sorted[i][0] > minVal) and (dist_fft_sorted[i][0] <= maxVal):
                    sum_dis= sum_dis + dist_fft_sorted[i][0]
                    sum_fft= sum_fft + dist_fft_sorted[i][1]
                    count = count + 1
            polar_fft[j][0] = sum_dis / count
            polar_fft[j][1] = sum_fft / count
            minVal = maxVal
            maxVal = maxVal + step

        # prepare directory
        path = os.path.join(os.path.dirname( __file__ ), 'output/', alpha + '/').replace('\\', '/')
        try:
            os.mkdir(path)
        except FileExistsError:
            # directory already exists
            pass

        plt.figure()
        plt.xlabel('Spatial frequency (nm^{-1})')
        plt.ylabel('Fourier amplitude (nm^{-1})')
        plt.xscale('log'), plt.yscale('log')
        plt.title('Polar FFT2D')
        # plot polar fft2d
        plt.plot(polar_fft[:,0], polar_fft[:,1], label = "Polar FFT2D")
        # plot mean rows fft2d
        plt.plot(polar_fft[:,0], fft2d_mean_row_half, label = 'Mean rows FFT2D')
        # plot mean columns fft2d
        plt.plot(polar_fft[:,0], fft2d_mean_col_half, label = 'Mean columns FFT2D')
        plt.legend()
        filename = "image_" + str(iter)
        plt.savefig(path + filename)

        Polar_mean = np.average(polar_fft)

    return None



if __name__ == "__main__":
    # argument parser
    # parser = argparse.ArgumentParser(description = 'Image polar fft2d')
    # parser.add_argument('image1', help = 'directory of first image to compare')
    # parser.add_argument('image2', help = 'directory of second image to compare')
    # # parser.add_argument('image3', help = 'directory of third image to compare')
    # # parser.add_argument('image4', help = 'directory of fourth image to compare')
    # # parser.add_argument('image5', help = 'directory of fifth image to compare')
    # args = parser.parse_args()

    arg1=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.50\CNNGenerator_CNNDiscriminator\2022_04_17_13_40_13_880276\Plots\grayscale\fake_00.png"
    arg2=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.50\CNNGenerator_CNNDiscriminator\2022_04_17_13_40_13_880276\Plots\grayscale\fake_01.png"
    arg3=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.50\CNNGenerator_CNNDiscriminator\2022_04_17_13_40_13_880276\Plots\grayscale\fake_02.png"
    arg4=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.50\CNNGenerator_CNNDiscriminator\2022_04_17_13_40_13_880276\Plots\grayscale\fake_03.png"
    arg5=r"Output\Alpha effect\Standard\dataset_1000_128_03_00_03_04_04_0.50\CNNGenerator_CNNDiscriminator\2022_04_17_13_40_13_880276\Plots\grayscale\fake_05.png"

    polar_fft2d(arg1, arg2, arg3, arg4, arg5)