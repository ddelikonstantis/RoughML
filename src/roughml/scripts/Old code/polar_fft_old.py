import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import math
import argparse


def load_image(image_filename=None):
    # img = []
    # for i in image_filename:
    #     im = cv2.imread(i)
    #     im = rgb2gray(im)
    #     img.append(im)
    # read image
    img = cv2.imread(image_filename)
    # convert to grayscale
    img = rgb2gray(img)
    # plot grayscale image and get shape
    # cv2.imshow('grayscale image ' + str(img.shape), img)

    return img

def get_polar_fft(img=None, save_path=None):
    # convert to pixel values
    img = (((img - img.min()) / (img.max() - img.min())) * 255.9)
    img = np.array(img, dtype=np.uint8)
    # get image dimension
    rows, clmns = img.shape[0], img.shape[1]
    # get image FFT2D and shift sums to the center
    fft2d = np.fft.fftshift(np.fft.fft2(img))
    # print('fft2d: ','\n', fft2d, '\n', fft2d.shape, '\n')
    # plt.imshow(np.log(abs(fft2d)), cmap='gray')
    # plt.show()
    # get FFT2D mean column values
    fft2d_mean_col = np.mean(abs(fft2d), axis=0)
    # keep only second half of the mean FFT matrices
    fft2d_mean_col_half = fft2d_mean_col[64:128]
    # print('fft2d_mean_col: ','\n', fft2d_mean_col, '\n', fft2d_mean_col.shape, '\n')
    # get FFT2D mean row values
    fft2d_mean_row = np.mean(abs(fft2d), axis=1)
    # keep only second half of the mean FFT matrices
    fft2d_mean_row_half = fft2d_mean_row[64:128]
    # print('fft2d_mean_row: ','\n', fft2d_mean_row, '\n', fft2d_mean_row.shape, '\n')
    # get absolute fourier values and flatten array
    flat_fft2d = abs(fft2d.ravel(order='F'))
    # print('flat_fft2d: ','\n', flat_fft2d, '\n', flat_fft2d.shape, '\n')
    # get image center points
    mid_idxrow, mid_idxcol = math.floor(rows / 2), math.floor(clmns / 2)
    dist = np.zeros(shape=(rows, clmns), dtype='float')
    # get indexes of the middle element of a matrix
    for i in range(rows):
            for j in range(clmns):
                dist[i][j] = math.sqrt(pow((i-mid_idxrow), 2) + pow((j-mid_idxcol), 2))
    # print('dist: ','\n', dist, '\n', dist.shape, '\n')
    # convert distance matrix as a column matrix
    flat_dist = dist.ravel(order='F')
    # print('flat_dist: ','\n', flat_dist, '\n', flat_dist.shape, '\n')
    # construct 2d array (first column: distances from center point, second column: corresponding fft values)
    dist_fft = np.vstack((flat_dist, flat_fft2d)).T
    # print('dist fft: ', '\n', dist_fft, '\n', dist_fft.shape, '\n')
    # Sort 2d array based on 1st column
    dist_fft_sorted = dist_fft[dist_fft[:,0].argsort()]
    # print('dist_fft_sorted: ','\n', dist_fft_sorted, '\n')
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
    # print('polar_fft: ','\n', polar_fft, '\n')

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
    plt.savefig(save_path)
    plt.close

    Polar_mean = np.average(polar_fft)

    return None


if __name__ == "__main__":
    # argument parser
    # parser = argparse.ArgumentParser(description = 'Image polar fft')
    # parser.add_argument('image1', help = 'directory of first image to compare')
    # parser.add_argument('image2', help = 'directory of second image to compare')
    # # parser.add_argument('image3', help = 'directory of third image to compare')
    # # parser.add_argument('image4', help = 'directory of fourth image to compare')
    # # parser.add_argument('image5', help = 'directory of fifth image to compare')
    # args = parser.parse_args()

    # img = load_image(r"src\roughml\scripts\Alpha effect\alpha 0.5\fake_00.png")
    # polar_fft = get_polar_fft(img)

    img = load_image(r"src\roughml\scripts\Alpha effect\alpha 0.5\fake_00.png") 
    polar_fft = get_polar_fft(img, save_path=r"src\roughml\scripts\Alpha effect\Plots\figure.png")
