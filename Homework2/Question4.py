"""
Vincenzo Macri
Digital Image Processing
Homework 2, Question 4:

Take any image and apply the Fourier Transform to this image and the following
filters:
(a) Butterworth filters
(b) Gaussian filters
--

Below please find my implementation of the Butterworth and Gaussian filters.

"""

import numpy as np
import cv2
from scipy import fftpack

# read the image
img = cv2.imread('dog.png', 0)  # read as grayscale

# compute fourier transform
f = fftpack.fft2(img)
fshift = fftpack.fftshift(f)

# get dimensions and center
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2

# create meshgrid
y, x = np.ogrid[:rows, :cols]
y, x = y - center_row, x - center_col

# butterworth filter
def butterworth_filter(d0, n):
    d = np.sqrt(x*x + y*y)
    return 1 / (1 + (d / d0)**(2*n))

# gaussian filter
def gaussian_filter(d0):
    d = np.sqrt(x*x + y*y)
    return np.exp(-(d*d) / (2*d0*d0))


d0 = 50  
n = 2    

# apply butterworth
h_butterworth = butterworth_filter(d0, n)
g_butterworth = fshift * h_butterworth
img_back_butterworth = np.abs(fftpack.ifft2(fftpack.ifftshift(g_butterworth)))

cv2.imwrite(f'outputs//butterworth_d0_{d0}_n_{n}.png', img_back_butterworth)
    
cv2.imshow('Butterworth Filter', img_back_butterworth / 255.0)
cv2.waitKey(0)

# apply gaussian
h_gaussian = gaussian_filter(d0)
g_gaussian = fshift * h_gaussian
img_back_gaussian = np.abs(fftpack.ifft2(fftpack.ifftshift(g_gaussian)))

cv2.imwrite(f'outputs//gaussian_d0_{d0}.png', img_back_gaussian)

cv2.imshow('Gaussian Filter', img_back_gaussian / 255.0)
cv2.waitKey(0)

cv2.destroyAllWindows()
