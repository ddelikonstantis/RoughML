import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import argparse

from sympy import false, true


def get_image_fft(image_filename, show_original=False, show_grayscale=False, show_fft=False):

    # load image
    original_image = cv2.imread(image_filename)

    if show_original:
        # plot original image and get shape
        cv2.imshow('original image ' + str(original_image.shape), original_image)
        # get original image fft
        fft2d_rgb = np.fft.fft2(original_image)
        # plot original image fft
        plt.imshow(np.log(abs(fft2d_rgb)), cmap='gray')
        plt.show()
    
    if show_grayscale:
        # convert image to grayscale
        grayscale_image = rgb2gray(original_image)
        # plot grayscale image and get shape
        cv2.imshow('grayscale image ' + str(grayscale_image.shape), grayscale_image)
        fft2d_gray = np.fft.fft2(grayscale_image)
        # plot original image fft
        plt.imshow(np.log(abs(fft2d_gray)), cmap='gray')
        plt.show()

    return None


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description = 'Image fft2d plot')
    parser.add_argument('image', help = 'directory of image')
    parser.add_argument('original', help = 'Plot fft for original image', action='store_false')
    parser.add_argument('grayscale', help = 'Plot fft for grayscale image', action='store_true')
    # parser.add_argument('fft', help = 'Plot fft')
    args = parser.parse_args()

    fft2d = get_image_fft(args.image, args.original, args.grayscale)