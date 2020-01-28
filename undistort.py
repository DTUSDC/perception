import numpy as np 
import cv2
import pickle

def cal_undistort(img, objpoints, imgpoints):
    dist = np.array([-0.423352,0.143247,0.003812,-0.008234,0.000000])
    mtx = np.array([[459.982148,0.000000,313.695275],[0.000000,463.201561,235.310658],[0.000000,0.000000,1.000000]])
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def cal_undistort1(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    undst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    return undst

def cal_undistort2(img, objpoints, imgpoints):
    dist = np.array([-0.361749,0.104000,-0.003157,0.000011,0.000000])
    mtx = np.array([[453.097111,0.000000,320.636246],[0.000000,458.793546,242.712218],[0.000000,0.000000,1.000000]])
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    undst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return undst

with open('objpoints.pkl','rb') as f:
    objpoints = pickle.load(f)
with open('imgpoints.pkl','rb') as f:
    imgpoints = pickle.load(f)
    
cap = cv2.VideoCapture(0)
if cap.isOpened()== False: 
    print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        undistorted = cal_undistort2(frame, objpoints, imgpoints)
        cv2.imshow('frame',frame)
        cv2.imshow('undistorted',undistorted)
        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()