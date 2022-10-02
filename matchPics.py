import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
	
	
            
	#I1, I2 : Images to match
	

	#Convert Images to GrayScale
	img1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
	
	
	#Detect Features in Both Images
	locs1 = corner_detection(img1,0.15)
	
	locs2 = corner_detection(img2,0.15)
	
	
	
	#Obtain descriptors for the computed feature locations
	desc1,locs1 = computeBrief(img1, locs1)
	desc2,locs2 = computeBrief(img2, locs2)
	

	#Match features using the descriptors
	matches = briefMatch(desc1,desc2,ratio = 0.65)
	

	
	return matches, locs1, locs2
