import numpy as np
import cv2
import skimage.io 
import skimage.color
import matplotlib.pyplot as plt
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches
import os
from loadVid import loadVid
#Import necessary functions




p1 = loadVid('../data/ar_source.mov')
p2 = loadVid('../data/ar_source.mov')
panda = np.concatenate((p1, p2), axis=0)

book = loadVid('../data/book.mov')








def cutimage(input):
    upper = 43
    lower = 312
    left = 200
    right = 410
    center = input[upper:lower,left:right,:]
    return center


def hp(cv_cover,book_desk,hp_panda):
    height = cv_cover.shape[0]
    width = cv_cover.shape[1]
    # resize cited from https://www.geeksforgeeks.org/resize-multiple-images-using-opencv-python/
    hp_cover_new = cv2.resize(hp_panda, (width,height))
    # find the matches to the desk 
    matches, locs1, locs2 = matchPics(cv_cover, book_desk )
    x1=locs1[matches[:,0],0:2]
    x2=locs2[matches[:,1],0:2]

    bestH2to1, inliers=computeH_ransac(x1,x2)
    hp_cover_new = cv2.transpose(hp_cover_new)
    img = compositeH(bestH2to1,hp_cover_new, book_desk)
    return img

def existmatch(cv_cover,book_desk):
    height = cv_cover.shape[0]
    width = cv_cover.shape[1]
    # resize cited from https://www.geeksforgeeks.org/resize-multiple-images-using-opencv-python/
    hp_cover_new = cv2.resize(hp_panda, (width,height))
    # find the matches to the desk 
    matches, locs1, locs2 = matchPics(cv_cover, book_desk )
    x1=locs1[matches[:,0],0:2]
    x2=locs2[matches[:,1],0:2]
    # chech if there exist sufficient matches
    if len(x1) <4:
        return False
    return True

time = book.shape[0]
cv_cover = cv2.imread('../data/cv_cover.jpg')

# cv2 Videowrite cited : https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
rows=book[1,:,:,:].shape[0]
cols = book[1,:,:,:].shape[1]
path = '../result'
res=cv2.VideoWriter(os.path.join(path , 'ar.avi'),cv2.VideoWriter_fourcc('X','V','I','D'),28,(cols,rows))

for i in range(0,time):
    
    book_desk = book[i,:,:,:]
    hp_panda = panda[i,:,:,:]
    if (existmatch(cv_cover,book_desk)):
        hp_panda = cutimage(hp_panda)

        slide = hp(cv_cover,book_desk,hp_panda)
        print(i)
        res.write(slide)

cv2.destroyAllWindows()
res.release()


    




# since the length of the book is longer, double the panda size




#Write script for Q4.1
