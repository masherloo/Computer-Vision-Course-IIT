#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import time


# In[4]:


def mouse_pos(event,x,y,flags,param):
    global points
    global gray, winName
    if event == cv.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv.circle(gray,(x, y),2,(0,0,0))
        cv.imshow(winName, gray)
        cv.waitKey(1)

def nothing(x):
    pass

def EnCal(i, lenPoints):
    global points, d
    En = np.zeros((9, 3))
    g = 0
    for j in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            En_cont = [0] * lenPoints
            En_cur = [0] * lenPoints
            En_img = [0] * lenPoints
            temp = points
            temp[i][0] += j
            temp[i][1] += k
            En[g][0] = j
            En[g][1] = k
            for h in range(-1, lenPoints-1):
                En_cont[h] += ((np.sqrt((temp[h+1][0] - temp[h][0])**2 + (temp[h+1][1] - temp[h][1])**2))**2)*alpha
                En_cur[h] += ((temp[h+1][0] - 2*(temp[h][0]) + temp[h-1][0])**2 + (temp[h+1][1] - 2*(temp[h][1]) + temp[h-1][1])**2)*beta
                En_img[h] += (gradient[temp[h][1]][temp[h][0]]**2)*gamma
            En[g][2] = sum(En_cont) + sum(En_cur) - sum(En_img)
            g += 1
    return En


# In[ ]:


winName = "Image"
points = list()
image = cv.imread("test.jpg")
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
ret, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)
cv.namedWindow(winName)
cv.setMouseCallback(winName,mouse_pos)
cv.createTrackbar("Alpha", winName, 0, 10, nothing)
cv.createTrackbar("Beta", winName, 0, 10, nothing)
cv.createTrackbar("Gamma", winName, 0, 10, nothing)
cv.createTrackbar("Iteration", winName, 0, 300, nothing)
cv.createTrackbar("Delay(ms)", winName, 0, 1000, nothing)
while(1):
    gradient = cv.Sobel(binary,cv.CV_64F,1,0,ksize=3)
    cv.imshow(winName, gray)
    KEY = cv.waitKey(0)
    if KEY == ord('s'):        
        alpha = cv.getTrackbarPos("Alpha", winName) / 10
        beta = cv.getTrackbarPos("Beta", winName) / 10
        gamma = cv.getTrackbarPos("Gamma", winName) / 10
        delay = cv.getTrackbarPos("Delay(ms)", winName) / 1000
        ite = cv.getTrackbarPos("Iteration", winName)
        for iteration in range(ite):
            if len(points) == 0:
                print("Please select points first")
                break
            E = 0
            Econt = [0] * len(points)
            Ecur = [0] * len(points)
            Eimg = [0] * len(points)
            En = np.zeros((9, 3))
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            for i in range(-1, len(points)-1):
                cv.line(gray,(points[i][0], points[i][1]),(points[i+1][0], points[i+1][1]),(0,155,0),2)
            time.sleep(delay)
            cv.imshow(winName, gray)
            cv.waitKey(1)
            t = 0
            for i in range(-1, len(points)-1):
                t += np.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)
            d = t/len(points)
            for i in range(-1, len(points)-1):
                Econt[i] += ((np.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2))**2)*alpha
                Ecur[i] += ((points[i+1][0] - 2*points[i][0] + points[i-1][0])**2 + (points[i+1][1] - 2*points[i][1] + points[i-1][1])**2)*beta
                Eimg[i] += (gradient[points[i][1]][points[i][0]]**2)*gamma
            E = sum(Econt) + sum(Ecur) - sum(Eimg) 
            for i in range(-1, len(points)-1):
                En = EnCal(i, len(points))
                En = En[np.argsort(En[:,2])]
                if En[0][2] < E:
                    points[i][0] += int(En[0][0])
                    points[i][1]+= int(En[0][1])
    if KEY == ord('h'):
        print("Select desired point by mouse and press 's' when you are finished to start the active contour algorithm")
    if KEY == ord('q'):
        break
cv.destroyAllWindows()

