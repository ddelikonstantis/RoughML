import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import argparse


# argument parser
parser = argparse.ArgumentParser(description = 'Image fft plot')
parser.add_argument('image', help = 'directory of image')
args = parser.parse_args()

# load preferred image via command line
image = cv2.imread(args.image)

# plot original image and get shape
cv2.imshow('original image ' + str(image.shape), image)

# convert image to grayscale
image = rgb2gray(image)
# plot grayscale image and get shape
cv2.imshow('grayscale image ' + str(image.shape), image)

# get fft
# fft2d_1 = fftpack.fft2(image)
fft2d_2 = np.fft.fftshift(np.fft.fft2(image))

# plot fft
plt.imshow(np.log(abs(fft2d_2)), cmap='gray')
plt.show()