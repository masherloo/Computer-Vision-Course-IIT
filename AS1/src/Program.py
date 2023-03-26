#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
from convolution2D import convolve2d
def nothing(x):
    pass
def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
def slideHandler(n):
    global image
    img = cv2.imread("1.jpg")
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(winName, image_gray)
    if n==0:
        cv2.imshow(winName, image_gray)
        image = image_gray
    else:
        kernel = np.ones((n,n), np.float32)/(n*n)
        dst = cv2.filter2D(image_gray, -1, kernel)
        cv2.imshow(winName, dst)
        image = dst
def slideHandlerOwnConvolution(n):
    global image
    img = cv2.imread("1.jpg")
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(winName, image_gray)
    if n==0:
        cv2.imshow(winName, image_gray)
        image = image_gray
    else:
        kernel = np.ones((n,n), np.float32)/(n*n)
        dst = convolve2d(image_gray, kernel)
        cv2.imshow(winName, dst)
        image = dst
def rotate(n):
    global image
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, n, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    cv2.imshow(winName, rotated_mat)


# In[2]:


winName = "Image"
if len(sys.argv) == 2:
    filename = sys.argv[1]
    image = cv2.imread(filename)
    cv2.imshow(winName, image)
    i = 0
    while True:
        key = cv2.waitKey(0)
        if key == ord('i'):
            cv2.destroyAllWindows()
            image = cv2.imread(filename)
            cv2.imshow(winName, image)
        if key == ord('w'):
            cv2.imwrite("out.jpg", image)
        if key == ord('g'):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image_gray
            cv2.imshow(winName, image_gray)
        if key == ord('s'):
            cv2.createTrackbar("s", winName, 0, 255, slideHandler)
        if key == ord('d'):
            dwos = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            image = dwos
            cv2.imshow(winName, dwos)
        if key == ord('D'):
            dws = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            kernel = np.ones((5,5), np.float32)/(25)
            dst = cv2.filter2D(dws, -1, kernel)
            image = dst
            cv2.imshow(winName, dst)
        if key == ord('x'):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
            dst = cv2.normalize(cv2.filter2D(image_gray, -1, kernel), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = dst
            cv2.imshow(winName, dst)
        if key == ord('y'):
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
            dst = cv2.normalize(cv2.filter2D(image_gray, -1, kernel), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = dst
            cv2.imshow(winName, dst)
        if key == ord('c'):
            image_oc = np.zeros(shape=image.shape, dtype=np.uint8)
            image_oc[:,:,i] = image[:,:,i]
            cv2.imshow(winName, image_oc)
            if i==2:
                i = 0
            else:
                i = i+1
        if key == ord('G'):
            image_gray = rgb2gray(image)
            image = image_gray
            cv2.imshow(winName, image_gray)
        if key == ord('S'):
            cv2.createTrackbar("s", winName, 0, 255, slideHandlerOwnConvolution)
        if key == ord('m'):
            sx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=5)
            sy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=5)
            sobel = np.hypot(sx,sy)/255
            image = sobel
            cv2.imshow(winName, sobel)
        if key == ord('r'):
            cv2.createTrackbar("r", winName, 0, 360, rotate)
        if key == ord('h'):
            print("This program is developed by Mohammadreza Asherloo, a PhD student at IIT. You can process the image with buttons below:")
            print("n: gray and rotate")
            print("m: magnitude of gradient")
            print("y: y derivative")
            print("x: x derivative")
            print("i: reload original image")
            print("g and G: graysclae")
            print("s: smoothing")
            print("d: downsample without smoothing")
            print("D: downsample wit smoothing")
            print("c: color channel change")
            print("w: save processed image")
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
elif len(sys.argv) < 2:
    winName = "Image"
    cam = cv2.VideoCapture(0)
    i = 0
    while True:
        retval, frame = cam.read()
        quit = -1
        keyy = -1
        s_key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            cv2.imwrite("out.jpg", frame)
        if s_key == ord('s') or s_key == ord('S'):
            cv2.createTrackbar("s", winName, 0, 255, nothing)
        if s_key == ord('r'):
            cv2.createTrackbar("r", winName, 0, 360, nothing)
        while s_key == ord('s'):
            retval, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            n = cv2.getTrackbarPos("s", winName)
            if n==0:
                pass
            if n!=0:
                kernel = np.ones((n,n), np.float32)/(n*n)
                frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('g'):
            retval, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('x'):
            retval, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
            frame = cv2.normalize(cv2.filter2D(frame, -1, kernel), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('y'):
            retval, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])
            frame = cv2.normalize(cv2.filter2D(frame, -1, kernel), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('d'):
            retval, frame = cam.read()
            frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('D'):
            retval, frame = cam.read()
            frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            kernel = np.ones((5,5), np.float32)/(25)
            frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('c'):
            retval, frame = cam.read()
            frame_oc = np.zeros(shape=frame.shape, dtype=np.uint8)
            frame_oc[:,:,i] = frame[:,:,i]
            cv2.imshow(winName, frame_oc)
            keyy = cv2.waitKey(1)
            if keyy == ord('c'):
                if i==2:
                    i = 0
                else:
                    i = i+1

            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('G'):
            retval, frame = cam.read()
            frame = rgb2gray(frame)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('S'):
            retval, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            n = cv2.getTrackbarPos("s", winName)
            if n==0:
                pass
            if n!=0:
                kernel = np.ones((n,n), np.float32)/(n*n)
                frame = convolve2d(frame, kernel)
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('m'):
            retval, frame = cam.read()
            sx = cv2.Sobel(frame,cv2.CV_32F,1,0,ksize=5)
            sy = cv2.Sobel(frame,cv2.CV_32F,0,1,ksize=5)
            frame = np.hypot(sx,sy)/255
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        while s_key == ord('r'):
            retval, frame = cam.read()
            n = cv2.getTrackbarPos("r", winName)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            image_center = (width/2, height/2)
            rotation_mat = cv2.getRotationMatrix2D(image_center, n, 1.)
            abs_cos = abs(rotation_mat[0,0]) 
            abs_sin = abs(rotation_mat[0,1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)
            rotation_mat[0, 2] += bound_w/2 - image_center[0]
            rotation_mat[1, 2] += bound_h/2 - image_center[1]
            frame = cv2.warpAffine(frame, rotation_mat, (bound_w, bound_h))
            cv2.imshow(winName, frame)
            keyy = cv2.waitKey(1)
            if keyy == ord('i'):
                break
            if keyy == ord('q'):
                quit = ord('q')
                break
        if s_key == ord('h'):
            print("This program is developed by Mohammadreza Asherloo, a PhD student at IIT. You can process the image with buttons below:")
            print("n: gray and rotate")
            print("m: magnitude of gradient")
            print("y: y derivative")
            print("x: x derivative")
            print("i: reload original image")
            print("g and G: graysclae")
            print("s: smoothing")
            print("d: downsample without smoothing")
            print("D: downsample wit smoothing")
            print("c: color channel change")
            print("w: save processed image")
        if s_key == ord('q') or quit == ord('q'):
            break
        cv2.imshow(winName, frame)
    cam.release()
    cv2.destroyAllWindows()
# In[ ]:




