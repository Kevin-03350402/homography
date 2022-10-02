import numpy as np
import cv2
from matchPics import matchPics
import scipy
import matplotlib.pyplot as plt
cover = cv2.imread('../data/cv_cover.jpg')

#Q3.5
#Read the image and convert to grayscale, if necessary
hist = []
for i in range(36):
	#Rotate Image
	rotatecv = scipy.ndimage.rotate(cover,i*10)
	
	#Compute features, descriptors and Match features
	nmatch,loc1,loc2 = matchPics(cover, rotatecv)
	#Update histogram
	hist.append(len(nmatch))
x = list(range(0,360,10))
plt.bar(x, hist,width = 5)
plt.show()
#Display histogram

