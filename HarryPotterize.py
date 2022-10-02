import numpy as np
import cv2
import skimage.io 
import skimage.color
import matplotlib.pyplot as plt
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')




height = cv_cover.shape[0]
width = cv_cover.shape[1]
# resize cited from https://www.geeksforgeeks.org/resize-multiple-images-using-opencv-python/
hp_cover_new = cv2.resize(hp_cover, (width,height))


# find the matches to the desk 
matches, locs1, locs2 = matchPics(cv_cover, cv_desk )

x1=locs1[matches[:,0],0:2]
x2=locs2[matches[:,1],0:2]


bestH2to1, inliers=computeH_ransac(x1,x2)



img = compositeH(bestH2to1,hp_cover_new, cv_desk)

plt.imshow(img)
plt.show()




#Write script for Q3.9
