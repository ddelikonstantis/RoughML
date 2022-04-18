import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import argparse


def polar_fft2d(*args):
    for arg in args:
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
        dist = np.zeros(shape=(rows, clmns), dtype='float')
        # get FFT2D mean absolute column values
        fft2d_mean_col = np.mean(abs(fft2d), axis=0)
        # keep only second half of the mean FFT matrices
        fft2d_mean_col_half = fft2d_mean_col[mid_idxcol:clmns]
        # get FFT2D mean absolute row values
        fft2d_mean_row = np.mean(abs(fft2d), axis=1)
        # keep only second half of the mean FFT matrices
        fft2d_mean_row_half = fft2d_mean_row[mid_idxrow:rows]
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
        plt.show()

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

    arg1=r"src\roughml\scripts\Alpha effect\fake_00.png"
    arg2=r"src\roughml\scripts\Alpha effect\fake_01.png"
    arg3=r"src\roughml\scripts\Alpha effect\fake_02.png"
    arg4=r"src\roughml\scripts\Alpha effect\fake_03.png"
    arg5=r"src\roughml\scripts\Alpha effect\fake_04.png"
    arg6=r"src\roughml\scripts\Alpha effect\fake_05.png"

    polar_fft2d(arg1, arg2, arg3, arg4, arg5, arg6)