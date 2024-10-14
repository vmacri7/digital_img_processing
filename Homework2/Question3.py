"""
Vincenzo Macri
Digital Image Processing
Homework 2, Question 3:

Explain what a Kuwahara filter is, and apply it to the image using either Python or
MATLAB to demonstrate its effect.
--

The Kuwahara filter is an edge-preserving smoothing filter which gives images
it is applied to a kind of "water color painting" effect. It works by dividing 
the area around each pixel into four overlapping windows, calculating the mean 
and variance of each window, and then replacing the central pixel with the mean 
of the window that has the smallest variance. This approach, while computationally 
intensive, helps to reduce noise while preserving edges, as it tends to average 
pixels from the most homogeneous region around each pixel. The result is a smoothed
image that maintains sharp edges and doesn't blur across different regions.  For the
image of a dog I ran this filter on, the edges and major features of the dog remained
sharp, while the fur and grass became much smoother.

"""

import numpy as np
import cv2
from tqdm import tqdm

def kuwahara_filter(image, window_size):
    height, width = image.shape[:2]
    pad = window_size // 2
    # pad the image to handle border pixels
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    output = np.zeros_like(image)
    
    for y in tqdm(range(height), desc="Applying Kuwahara filter"):
        for x in range(width):
            # define four sub_windows
            sub_windows = [
                padded_image[y:y+window_size, x:x+window_size],
                padded_image[y:y+window_size, x+1:x+window_size+1],
                padded_image[y+1:y+window_size+1, x:x+window_size],
                padded_image[y+1:y+window_size+1, x+1:x+window_size+1]
            ]
            
            # calculate mean and variance for each sub_window
            means = [np.mean(window) for window in sub_windows]
            variances = [np.var(window) for window in sub_windows]
            
            # select the mean of the sub_window with minimum variance
            min_var_index = np.argmin(variances)
            output[y, x] = means[min_var_index]
    
    return output


image = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)
filtered_image = kuwahara_filter(image, window_size=5)

cv2.imshow('Original Image', image)
cv2.imshow('Kuwahara Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('outputs//dog_kuwahara_filtered.png', filtered_image)
