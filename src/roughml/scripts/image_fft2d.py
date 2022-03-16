import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.fft import fftfreq
import cv2
import math
import argparse


def get_image_fft(image_filename, show_original=False, show_grayscale=False, fft_2d_vector=False, show_fft=False):

    # load image
    original_image = cv2.imread(image_filename)

    if show_original:
        # plot original image and get shape
        cv2.imshow('original image ' + str(original_image.shape), original_image)
        # get original image fft2D
        fft2d_rgb = np.fft.fft2(original_image)
        # plot original image fft2D
        plt.imshow(np.log(abs(fft2d_rgb)), cmap='gray')
        plt.show()
    
    if show_grayscale:
        # convert image to grayscale
        grayscale_image = rgb2gray(original_image)
        # plot grayscale image and get shape
        cv2.imshow('grayscale image ' + str(grayscale_image.shape), grayscale_image)
        # get grayscale image fft and fft2D
        fft2d_gray = np.fft.fft2(grayscale_image)

        plt.imshow(np.log(abs(fft2d_gray)), cmap='gray')
        plt.title('FFT 2D')
        plt.show()

        # plot polar fft
        flat_fft = abs(fft2d_gray.ravel())
        mid_idxrow = (fft2d_gray.shape[0] / 2) + 1
        mid_idxcol = (fft2d_gray.shape[1] / 2) + 1
        dist=np.eye(fft2d_gray.shape[0],fft2d_gray.shape[1])
        for i in range(fft2d_gray.shape[0]):
            for j in range(fft2d_gray.shape[1]):
                dist[i][j] = math.sqrt(pow((i-mid_idxrow), 2) + pow((j-mid_idxcol), 2))
        
        new_dist = dist.ravel()
        dist_reshape = [new_dist, flat_fft]

        if fft2d_gray.shape[0] < fft2d_gray.shape[1]:
            cycle_radius = fft2d_gray.shape[0] / 2
        else:
            cycle_radius = fft2d_gray.shape[1] / 2
        
        minVal = 0
        maxVal = 1
        polar_ft = []
        for j in range(fft2d_gray.shape[0]):
            sum_dis = 0
            sum_fft = 0
            count = 0
            for i in range(dist_reshape):
                if dist_reshape[i,j] > minVal and dist_reshape[i,j] <= minVal:
                    sum_dis = sum_dis + dist_reshape[i]
                    sum_fft = sum_fft + dist_reshape[i]
                    count = count + 1
            
            polar_ft[j] = sum_dis / count
            polar_ft[j] = sum_fft / count
            minVal = maxVal
            maxVal = maxVal + j

    return None

if __name__ == "__main__":
    # argument parser
    # parser = argparse.ArgumentParser(description = 'Image fft2d plot')
    # parser.add_argument('image', help = 'directory of image')
    # parser.add_argument('original', help = 'Plot fft for original image', action='store_false')
    # parser.add_argument('grayscale', help = 'Plot fft for grayscale image', action='store_true')
    # parser.add_argument('fft2d_vector', help = 'Plot fft vector', action='store_true')
    # # parser.add_argument('fft', help = 'Plot fft')
    # args = parser.parse_args()

    # fft2d = get_image_fft(args.image, args.original, args.grayscale, args.fft2d_vector)
    fft2d = get_image_fft("src/roughml/scripts/fake_00.png", False, True, False)