import argparse
import cv2
import numpy as np
import os
import glob
import pandas as pd
import scipy.linalg as sp
from sklearn.metrics import mean_squared_error
import sys
pd.set_option('precision', 4)

def mouse_click(event, x, y, flags, param):
    global mouse_point, img
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),4,(255,255,255),-1)
        cv2.imshow('img',img)
        mouse_point.append([x, y])

if len(sys.argv) == 2:
    filename = sys.argv[1]
    if filename == 'test_points.txt':
        pass
    else:
        objpoints = [[0,0,0],[5,0,0],[10,0,0],[10,0,3],[10,5,3],[10,10,3],[5,10,3],[0,10,3],[0,5,3],[0,0,3],[0,5,0],[0,10,0]]
        imgpoints = []
        cv2.namedWindow('img')
        mouse_point = []
        img = cv2.imread('test.jpg')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('img',img)
        cv2.setMouseCallback("img", mouse_click)
        h = cv2.waitKey(0)
        if h == ord('q'):
            with open("points.txt", 'w') as file:
                for i in range(len(mouse_point)):
                    s = " ".join(map(str, objpoints[i])) + " " + " ".join(map(str, mouse_point[i]))[1:-1]
                    file.write(s+'\n')
                    cv2.destroyAllWindows()



points = pd.read_csv(filename, delimiter='\s+', header = None, dtype = float)
objpoints_file = points[[0, 1, 2]]
imgpoints_file = points[[3, 4]]
A = np.zeros((objpoints_file.shape[0]*2, 12))
j = 0
for i in range(0, objpoints_file.shape[0]*2, 2):
    A[i][0] = objpoints_file[0][j]
    A[i][1] = objpoints_file[1][j]
    A[i][2] = objpoints_file[2][j]
    A[i][3] = 1
    A[i][8] = objpoints_file[0][j] * -1 * imgpoints_file[3][j]
    A[i][9] = objpoints_file[1][j] * -1 * imgpoints_file[3][j]
    A[i][10] = objpoints_file[2][j] * -1 * imgpoints_file[3][j]
    A[i][11] = 1 * -1 * imgpoints_file[3][j]
    A[i+1][4] = objpoints_file[0][j]
    A[i+1][5] = objpoints_file[1][j]
    A[i+1][6] = objpoints_file[2][j]
    A[i+1][7] = 1
    A[i+1][8] = objpoints_file[0][j] * -1 * imgpoints_file[4][j]
    A[i+1][9] = objpoints_file[1][j] * -1 * imgpoints_file[4][j]
    A[i+1][10] = objpoints_file[2][j] * -1 * imgpoints_file[4][j]
    A[i+1][11] = 1 * -1 * imgpoints_file[4][j]
    j += 1
u, d, v = np.linalg.svd(A)
x = np.transpose(v)[:,-1]
M = x.reshape(3,4)


a1 = M[0][0:3].reshape(3,1)
a2 = M[1][0:3].reshape(3,1)
a3 = M[2][0:3].reshape(3,1)
b = M[:, -1]

rho_abs = 1/(np.sqrt(a3[0]**2+a3[1]**2+a3[2]**2))
U0 = rho_abs**2 * (np.dot(np.squeeze(np.asarray(a1)),np.squeeze(np.asarray(a3))))
V0 = rho_abs**2 * (np.dot(np.squeeze(np.asarray(a2)),np.squeeze(np.asarray(a3))))
alpha_v = np.sqrt((rho_abs**2 * (np.dot(np.squeeze(np.asarray(a2)),np.squeeze(np.asarray(a2))))) - V0**2)

s = (1/alpha_v) * rho_abs**4 * (np.dot(np.squeeze(np.asarray(np.cross(a1,a3,axis=0))),np.squeeze(np.asarray(np.cross(a2,a3,axis=0)))))

rho_sign = np.sign(b[2])

alpha_u = np.sqrt((rho_abs**2 * (np.dot(np.squeeze(np.asarray(a1)),np.squeeze(np.asarray(a1))))) - s**2 - U0**2)

kstar = np.zeros((3,3))
kstar[0][0] = alpha_u; kstar[1][1] = alpha_v; kstar[0][1] = s; kstar[0][2] = U0; kstar[1][2] = V0; kstar[2][2] = 1;

Tstar = rho_sign * rho_abs * (np.linalg.inv(kstar) @ b)
r3 = rho_sign * rho_abs * a3
r1 = rho_abs**2 / alpha_v * np.cross(a2,a3,axis=0)
r2 = np.cross(r3, r1, axis = 0)
Rstar = np.concatenate([r1,r2,r3]).T.reshape(3,3)

M_computed = kstar @ np.concatenate([Rstar, Tstar.reshape(3,1)], axis = 1)


a1 = M[0][0:3].reshape(3,1)
a2 = M[1][0:3].reshape(3,1)
a3 = M[2][0:3].reshape(3,1)
b = M[:, -1]

rho_abs = 1/(np.sqrt(a3[0]**2+a3[1]**2+a3[2]**2))
U0 = rho_abs**2 * (np.dot(np.squeeze(np.asarray(a1)),np.squeeze(np.asarray(a3))))
V0 = rho_abs**2 * (np.dot(np.squeeze(np.asarray(a2)),np.squeeze(np.asarray(a3))))
alpha_v = np.sqrt((rho_abs**2 * (np.dot(np.squeeze(np.asarray(a2)),np.squeeze(np.asarray(a2))))) - V0**2)

s = (1/alpha_v) * rho_abs**4 * (np.dot(np.squeeze(np.asarray(np.cross(a1,a3,axis=0))),np.squeeze(np.asarray(np.cross(a2,a3,axis=0)))))

rho_sign = np.sign(b[2])

alpha_u = np.sqrt((rho_abs**2 * (np.dot(np.squeeze(np.asarray(a1)),np.squeeze(np.asarray(a1))))) - s**2 - U0**2)

kstar = np.zeros((3,3))
kstar[0][0] = alpha_u; kstar[1][1] = alpha_v; kstar[0][1] = s; kstar[0][2] = U0; kstar[1][2] = V0; kstar[2][2] = 1;

Tstar = rho_sign * rho_abs * (np.linalg.inv(kstar) @ b)
r3 = rho_sign * rho_abs * a3
r1 = rho_abs**2 / alpha_v * np.cross(a2,a3,axis=0)
r2 = np.cross(r3, r1, axis = 0)
Rstar = np.concatenate([r1,r2,r3]).T.reshape(3,3)

M_computed = kstar @ np.concatenate([Rstar, Tstar.reshape(3,1)], axis = 1)

predicted_points = imgpoints_file.copy()
predicted_points[2] = 1
object = objpoints_file.copy()
object[3] = 1
for i in range(predicted_points.shape[0]):
    predict = M_computed @ np.array(object.iloc[[i]]).reshape(4,1)
    predicted_points.iloc[[i]] = predict.reshape(1,3) / predict[2]

predicted_points = predicted_points.drop([2], axis = 1)
rms = mean_squared_error(imgpoints_file, predicted_points)

print("T* = {}".format(Tstar))
print("\nR* = {}".format(Rstar))
print("\nK* = {}".format(kstar))
print("\n(u0, v0) = ({:.2f}, {:.2f})".format(U0[0], V0[0]))
print("\n(alpha_u, alpha_v) = ({:.2f}, {:.2f})".format(alpha_u[0], alpha_v[0]))
print("\nMSE = {:.18f}".format(rms))