# https://www.youtube.com/watch?v=uKcuOweBnKE
"""
Perform simple discrete cosine transformation on an image

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

B=8
fn3 = './logo.png'
img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
h, w = np.array(img1.shape[:2])/B * B
print(h, w)
h= int(h)
w=int(w)
img1=img1[:h, :w]

blocksV=h//B
blocksH=w//B
vis0 =  np.zeros((h,w), np.float32)
Trans = np.zeros((h,w), np.float32)
vis0[:h, :w] = img1
for row in range(blocksV):
    for col in range(blocksH):
        currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
        Trans[row*B:(row+1)*B, col*B:(col+1)*B]=currentblock
cv2.imwrite('t.jpg', Trans)