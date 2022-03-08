import numpy as np
import sys
import cv2
import argparse


def mse(img1, img2):
	img1 = cv2.imread(img1)
	img2 = cv2.imread(img2)
	# get difference of images by subtracting the pixel intensities
	# square this difference and get the sum
	error = np.sum((img1 - img2) ** 2)
	# divide the sum of squares by the total number of pixels
	error = error / (img1.shape[0] * img1.shape[1])

	return error

# argument parser
parser = argparse.ArgumentParser(description = 'Image similarity comparison')
parser.add_argument('image1', help = 'directory of first image to compare')
parser.add_argument('image2', help = 'directory of second image to compare')
args = parser.parse_args()

# get mean square error
mymse = mse(args.image1, args.image2)
# print("Mean Square Error: ", mymse)

# if mean square error has a value images are different, otherwise they are equal
if mymse > 0:
	sys.exit("Images are different")
else:
	sys.exit("0")