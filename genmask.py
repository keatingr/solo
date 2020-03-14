"""
Take white bckgrnd solo logo and generate a mask; both will be simultaneously transformed in augmentation step
"""
import cv2
import numpy as np

img = cv2.imread('./solo.png', cv2.IMREAD_GRAYSCALE)
w, h = img.shape[0], img.shape[1]

maskout = np.zeros((w, h, 3))


for col in range(w):
    for row in range(h):
        if img[col][row] < 250:  # TODO revisit thresh for white in original png logo
            maskout[col][row] = 255

cv2.imwrite('./traindata/', maskout)
# a=1
