import numpy as np
import cv2
import math
def computeH(x1, x2):

    a = []
    for i in range(x1.shape[0]):
        x = x2[i,0]
        y = x2[i,1]
        xp = x1[i,0]
        yp = x1[i,1]

        a.append([-x,-y,-1,0,0,0,x*xp,y*xp,xp])
        a.append([0,0,0,-x,-y,-1,x*yp,y*yp,yp])
    u, s, v = np.linalg.svd(np.array(a))
    H2to1 = v[-1, :].reshape(3, 3)
    return H2to1

def computeH_norm(x1, x2):
    #Q3.7
    #Compute the centroid of the points

    p1xmean = np.mean(x1[:,0])
    p1ymean = np.mean(x1[:,1])
    p2xmean = np.mean(x2[:,0])
    p2ymean = np.mean(x2[:,1])

    dist1 = []
    dist2 = []
    length = x1.shape[0]
    for i in range (0,length):
        d1 = math.sqrt((x1[i,0]-p1xmean)**2+(x1[i,1]-p1ymean)**2)
        dist1.append(d1)
    
    for i in range (0,length):
        d2 = math.sqrt((x2[i,0]-p2xmean)**2+(x2[i,1]-p2ymean)**2)
        dist2.append(d2)
    
    max1 = np.max(np.array(dist1))
    if max1 == 0:
        max1 = math.sqrt(2)
    max2 = np.max(np.array(dist2))
    if max2 == 0:
        max2 = math.sqrt(2)

    ratio1 = math.sqrt(2)/max1
    ratio2 = math.sqrt(2)/max2
    translation1 = np.array([[1,0,-p1xmean],[0,1,-p1ymean],[0,0,1]])
    scale1 = np.array([[ratio1,0,0],[0,ratio1,0],[0,0,1]])
    T1 = np.matmul(scale1,translation1)
    translation2 = np.array([[1,0,-p2xmean],[0,1,-p2ymean],[0,0,1]])
    scale2 = np.array([[ratio2,0,0],[0,ratio2,0],[0,0,1]])
    T2 = np.matmul(scale2,translation2)

    stx1 = np.vstack((x1.T,np.ones((x1.shape[0]))))
    normal_x1 = T1@stx1
    stx2 = np.vstack((x2.T,np.ones((x2.shape[0]))))
    normal_x2 = T2@stx2
    x1in = normal_x1.T
    x2in = normal_x2.T
    x1in = x1in[:,0:2]
    x2in = x2in[:,0:2]
    homoH = computeH(x1in, x2in)
    H2to1=np.matmul(np.linalg.inv(T1),homoH)
    H2to1=np.matmul(H2to1,T2)
    return H2to1





    


    

    



def computeH_ransac(x1, x2):
    #Q3.8
    #Compute the best fitting homography given a list of matching points
    stx1 = np.vstack((x1.T,np.ones((x1.shape[0]))))
    stx2 = np.vstack((x2.T,np.ones((x2.shape[0]))))
    epsilon = 0.5 
    iteration = 1000
    length = x1.shape[0]
    maxinliner = np.zeros((length,1))
    maxcount = -math.inf
    bestmodel = np.ones((3,3))
    for i in range (0,iteration):
        rows = np.random.choice(length, size=4) 
        select1 = x1[rows,:]
        select2 = x2[rows,:]
        model = computeH_norm(select1, select2)
        # check the number of matches
        # initializa a inliers
        predictx1 = np.matmul(model,stx2)
        inliners = np.zeros((length,1))
        count = 0
        for j in range (0, length):
            # select the row
            realp1x = stx1[0,j]
            realp1y = stx1[1,j]
            
            
            prep1x = predictx1[0,j]
            prep1y = predictx1[1,j]
            z = predictx1[2,j]
            # test if we have a match
            
            error1 = realp1x-prep1x/z
            error2 = realp1y-prep1y/z
            diff = error1**2+error2**2
            if np.sqrt(diff) <= epsilon:
            # find a match
                inliners[j] = 1
                count+=1
        if count > maxcount:
            maxinliner = inliners
            maxcount = count
            
    
    il = maxinliner.reshape((1,maxinliner.shape[0])).flatten()
    
    index = []
    for i in range(len(il)):
        if il[i] == 1:
            index.append(i)
    
    inliers = maxinliner
    matchx1 = x1[index,:]
    matchx2 = x2[index,:]
    bestH2to1 = computeH_norm(matchx1,matchx2)
    

                




    return bestH2to1, inliers




def compositeH(H2to1, template, img):
     
     #Create a composite image after warping the template image on top
     #of the image using the homography

     #Note that the homography we compute is from the image to the template;
     #x_template = H2to1*x_photo
     #For warping the template to the image, we need to invert it.

        covertodesk = np.linalg.inv(H2to1)
        outputheight = img.shape[0]
        outputwidth = img.shape[1]
        inputheight = template.shape[0]
        inputwidth = template.shape[1]
        mask = np.ones((inputheight,inputwidth))
        mask = cv2.transpose(mask)

        t = cv2.transpose(template)
        # transpose function cited from https://www.geeksforgeeks.org/resize-multiple-images-using-opencv-python/
    

        # find the matches of hp cover on the output
        
        temp = cv2.warpPerspective(mask,covertodesk,(outputheight,outputwidth))
        temp = cv2.transpose(temp)
        hpindex = np.where(temp!=0)
        holder = cv2.warpPerspective(t,covertodesk,(outputheight,outputwidth))
        holder = cv2.transpose(holder)
        img[hpindex] = holder[hpindex]
        return img


     # first map the template to the mask 
 
   



