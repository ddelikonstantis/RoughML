import numpy as np
from pathlib import Path
from scipy import fftpack
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

# get current working dir
cwd = Path.cwd()
# complete path to scripts folder
cwd = str(cwd) + "/src/" + "roughml/" + "scripts/"

# load preferred image
image = "fake_00.png"
image = cv2.imread(cwd + image)

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