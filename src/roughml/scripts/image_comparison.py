import numpy as np
import cv2
from sewar.full_ref import mse

def mymse(img1, img2):
	# get difference of images by subtracting the pixel intensities
	# square this difference and get the sum
	error = np.sum((img1 - img2) ** 2)
	# divide the sum of squares by the total number of pixels
	error = error / (img1.shape[0] * img1.shape[1])

	return error

# load images
image1 = cv2.imread("fake_00.png")
image2 = cv2.imread("fake_00.png")

# get mean square error
mymse1 = mymse(image1, image2)
print("Mean Square Error: ", mymse1)

# if either of the calculated mean square errors has a value
# images are different, otherwise they are equal
if mymse1 > 0:
	print("Images are different")
else:
	print("Images are equal")