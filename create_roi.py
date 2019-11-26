
# Miscelaneous of useful detection processing functions

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
#from skimage.filters import threshold_mean
from random import randint
import math



# Actual system
img = cv2.imread('frame8.jpg') #Command to read the image


#These are the images corners, top right, bottom left, bottom right and top right 
pts1 = np.array([[0,30],[0,210],[452,642],[715,502]], np.int32) #First mask
pts2 = np.array([[1277,543],[1556,682],[1920,338],[1920,120]], np.int32) #Second mask

pts_b1 = np.array([[1255,543],[1565,711],[1572, 677], [1291,529]], np.int32) #bounding 1
pts_b2 = np.array([[711,488],[425,630],[449,676],[767,510]], np.int32) #Bounding 2

pts1= pts1.reshape((-1,1,2))
pts2= pts2.reshape((-1,1,2))

pts_b1= pts_b1.reshape((-1,1,2))
pts_b2= pts_b2.reshape((-1,1,2))

#Create a mask of zeros of the image size
mask1 = np.zeros_like(img) #First mask
mask2 = np.zeros_like(img) #Second mask

mask3 = np.zeros_like(img) #Second mask
mask4 = np.zeros_like(img) #Limit line 2

# filling pixels inside the polygon defined by "vertices" with the fill color (If i want to change the color, just have to set the RGB sequence (R,G,B)
cv2.fillPoly(mask1, [pts1], 255) #First mask
cv2.fillPoly(mask2, [pts2], 255) #Second mask

cv2.fillPoly(mask3, [pts_b1], (255,0,255)) #Limit line 1
cv2.fillPoly(mask4, [pts_b2], (255,0,255)) #Limit line 1
#Add polylines in the points chosen by the user
cv2.polylines(img,[pts1],True,(0,255,255))
cv2.polylines(img,[pts2],True,(0,255,255))
#this blends the mask into the main image
#The inputs are first input array, weight, source 2, weight of the second image, Scallar add to each sum, and the output of the image
cv2.addWeighted(mask1, 1, img, 1, 0, img)
cv2.addWeighted(mask2, 1, img, 1, 0, img)
cv2.addWeighted(mask3, 1, img, 1, 0, img)
cv2.addWeighted(mask4, 1, img, 1, 0, img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
