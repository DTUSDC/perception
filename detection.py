from __future__ import print_function
import numpy as np
import os
import glob                                                                                                                                                                  
import sys
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.cluster import DBSCAN,KMeans
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import scipy.interpolate as interpolate
from collections import Counter

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

video_path = r"C:\Users\aayaa\Desktop\lane-videos\vid50_800,448.mp4"
weight_path = r'\Users\aayaa\Desktop\Lane Detection\New folder\unet_lane2.hdf5'
source = cv2.imread('green.jpg')

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def testGenerator(img1,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = trans.resize(img1,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def color_transfer(source, target):
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
 
    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
 
    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
 
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
 
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
 
    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    # return the color transferred image
    return transfer

def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
 
    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def processImageOld(im1) :  

    im1 = np.array(255*(im1/255)**1.3,dtype='uint8')
    im1 = cv2.GaussianBlur(im1, (7 , 7) , 0 )
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    h1 = im1[:,:,0]
    s1 = im1[:,:,1]
    v1 = im1[:,:,2]
    v1 = cv2.bitwise_not(v1)
    kernel =  np.ones((3,3), np.uint8) 
    v1 = cv2.erode(v1, kernel , iterations=2) 
    v1 = cv2.GaussianBlur(v1, (11 , 11) , 0 )
    v1 = cv2.adaptiveThreshold(v1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 157, 19) 
    v1 = cv2.bitwise_not(v1)
    vertices1 = np.array([[(0,v1.shape[0]-50),(0, v1.shape[0]-450 ), (v1.shape[1], im1.shape[0]-450), (v1.shape[1],v1.shape[0]-50)]], dtype=np.int32)    
    mask1 = np.zeros_like(v1) 
    ignore_mask_color = 255 
    cv2.fillPoly(mask1 , vertices1 , ignore_mask_color ) 
    v1 = cv2.bitwise_and(v1 , mask1)
    kernel =  np.ones((3,3), np.uint8) 
    v1 = cv2.erode(v1, kernel , iterations=2) 
    #kernel =  np.ones((3,3), np.uint8) 
    #v1 = cv2.erode(v1, kernel , iterations=2) 

    return v1 

def processImage5(im1) :
   
    im1 = np.array(255*(im1/255)**1.3,dtype='uint8')
    im1 = cv2.GaussianBlur(im1, (7 , 7) , 0 )
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    h1 = im1[:,:,0]
    s1 = im1[:,:,1]
    v1 = im1[:,:,2]
    v1 = cv2.bitwise_not(v1)
    kernel =  np.ones((3,3), np.uint8)
    v1 = cv2.dilate(v1, kernel , iterations=2)
    v1 = cv2.GaussianBlur(v1, (5, 5) , 0)
    v1 = cv2.adaptiveThreshold(v1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 157, 19)
    v1 = cv2.bitwise_not(v1)
    return v1

def processImage6(im1) :
    
    im1 = color_transfer(source , im1)
    im1 = np.array(255*(im1/255)**1.3,dtype='uint8')
    im1 = cv2.GaussianBlur(im1, (7 , 7) , 0 )   
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    v1 = im1[:,:,2]
    v1 = cv2.bitwise_not(v1)
    kernel =  np.ones((3,3), np.uint8)
    v1 = cv2.erode(v1, kernel , iterations=2)
    v1 = cv2.GaussianBlur(v1, (5, 5) , 0)
    v1 = cv2.adaptiveThreshold(v1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 157, 19)
    v1 = cv2.bitwise_not(v1)   
    return v1

def pos(x):
    if x>0 and x<=255:
        return x
    elif x>255:
        return 255
    else:
        return 0

model = unet()
model.load_weights(weight_path)
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(r'detected\output.mp4', fourcc, 20.0, (256,256))
k=1
s=400
interval_pts = 50
save = True
if cap.isOpened()== False: 
    print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")    
while cap.isOpened():
    ret,frame = cap.read()
    if ret==True:
        if cv2.waitKey(25) & 0xFF == 27:
            break
        frame = cv2.resize(frame,(256,256))
        img1 = frame.copy()
        img = processImageOld(frame)
        cv2.imshow('Before',img)
        #img[0:80,:] = 0
        testGene = testGenerator(img)
        img = (model.predict(testGene,steps=1).reshape((256,256)))*255
        _,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        cv2.imshow('thresh',img)
        
        _,contours, _=cv2.findContours(np.int32(img),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        minArea = 50
        cont1 = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > minArea:
                cont1.append(cnt)

        image = np.zeros((256,256,3))
        cv2.drawContours(image,cont1,-1,(255,255,255),3);
        cv2.imshow('contours',image)
        final = []
        for cnt in cont1:
            for i in cnt:
                final.append(i)
        final = np.array(final)
        final = np.reshape(final,[-1,2])
        if len(final) == 0:
            continue
        
        db = DBSCAN(eps=30,min_samples=30,algorithm='auto',metric='euclidean')
        db.fit(final)
        labels = list(db.labels_)
        final = list(final)

        if -1 in labels:
            final = [final[i] for i in range(len(labels)) if labels[i] != -1]
            labels = [labels[i] for i in range(len(labels)) if labels[i] != -1]
        assert -1 not in labels
        l = len(set(labels))
        print(l)
        start_y = 256
        end_y = 0

        if l >= 3:
            final1 = []
            final2 = []
            for i,j in zip(labels,final):
                if i==sorted(zip(Counter(labels).values(),Counter(labels).keys()),reverse=True)[0][1]:
                    final1.append(j)
                elif i==sorted(zip(Counter(labels).values(),Counter(labels).keys()),reverse=True)[1][1]:
                    final2.append(j)
                else:
                    continue
            km1 = KMeans(n_clusters=1).fit(final1)
            km2 = KMeans(n_clusters=1).fit(final2)

            if np.ravel(km1.cluster_centers_)[0] > np.ravel(km2.cluster_centers_)[0]:
                final1,final2 = final2,final1
            
            final1_x = [i[0] for i in final1]
            final1_y = [i[1] for i in final1]
            final2_x = [i[0] for i in final2]
            final2_y = [i[1] for i in final2]

            a = [i for i in sorted(list(zip(final1_x,final1_y)))]
            b = [i for i in sorted(list(zip(final2_x,final2_y)))]

            a1,a2 = [],[]
            for i in range(1,len(a)+1):
                if i == len(a):
                    break
                if a[i-1][0] == a[i][0]:
                    continue
                else:
                    a1.append(a[i])
            final1_x,final1_y = [i[0] for i in a1],[i[1] for i in a1]
            a1 = [i for i in sorted(list(zip(final1_y,final1_x)))]
            for i in range(1,len(a1)+1):
                if i == len(a1):
                    break
                if a1[i-1][0] == a1[i][0]:
                    continue
                else:
                    a2.append(a1[i])
            a3 = sorted([(i[1],i[0]) for i in a2])
            left_x,left_y = [i[0] for i in a3],[i[1] for i in a3]

            b1,b2 = [],[]
            for i in range(1,len(b)+1):
                if i == len(b):
                    break
                if b[i-1][0] == b[i][0]:
                    continue
                else:
                    b1.append(b[i])
            final2_x,final2_y = [i[0] for i in b1],[i[1] for i in b1]
            b1 = [i for i in sorted(list(zip(final2_y,final2_x)))]
            for i in range(1,len(b1)+1):
                if i == len(b1):
                    break
                if b1[i-1][0] == b1[i][0]:
                    continue
                else:
                    b2.append(b1[i])
            b3 = sorted([(i[1],i[0]) for i in b2])
            right_x,right_y = [i[0] for i in b3],[i[1] for i in b3]

            t1, c1, k1 = interpolate.splrep(left_x,left_y, s=s, k=k)
            t2, c2, k2 = interpolate.splrep(right_x,right_y, s=s, k=k)
            
            spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)
            spline2 = interpolate.BSpline(t2, c2, k2, extrapolate=False)

            x1min, x1max = np.array(left_x).min(), np.array(left_x).max()
            x2min, x2max = np.array(right_x).min(), np.array(right_x).max()

            xx1 = np.linspace(x1min, x1max, interval_pts)
            xx2 = np.linspace(x2min, x2max, interval_pts)

            left_lane = [[i,j] for i,j in zip(xx1,spline1(xx1))]
            right_lane = [[i,j] for i,j in zip(xx2,spline2(xx2))]

            cv2.polylines(img1, np.int32([left_lane]), isClosed=False, color=(255,0,0),thickness=3)
            cv2.polylines(img1, np.int32([right_lane]), isClosed=False, color=(0,0,255),thickness=3)
            '''left_coeff = np.polyfit(final1_x,final1_y,5)
            right_coeff = np.polyfit(final2_x,final2_y,5)

            left_coeff_inv = np.polyfit(final1_y,final1_x,5)
            right_coeff_inv = np.polyfit(final2_y,final2_x,5)

            left_coeff = np.polyfit(final1_x,final1_y,2)
            right_coeff = np.polyfit(final2_x,final2_y,2)

            left_coeff_inv = np.polyfit(final1_y,final1_x,2)
            right_coeff_inv = np.polyfit(final2_y,final2_x,2)

            left_coeff = np.polyfit(final1_x,final1_y,3)
            right_coeff = np.polyfit(final2_x,final2_y,3)

            left_coeff_inv = np.polyfit(final1_y,final1_x,3)
            right_coeff_inv = np.polyfit(final2_y,final2_x,3)

            left_start_x = int(pos((start_y**5) * left_coeff_inv[0]  + (start_y**4)*left_coeff_inv[1] + (start_y**3) * left_coeff_inv[2]  + (start_y**2)*left_coeff_inv[3]+ start_y*left_coeff_inv[4] + left_coeff_inv[5]))
            left_end_x = int(pos((end_y**5) * left_coeff_inv[0]  + (end_y**4)*left_coeff_inv[1] + (end_y**3) * left_coeff_inv[2]  + (end_y**2)*left_coeff_inv[3]+ end_y*left_coeff_inv[4] + left_coeff_inv[5]))
            right_end_x = int(pos((start_y**5) * right_coeff_inv[0]  + (start_y**4)*right_coeff_inv[1] + (start_y**3) * right_coeff_inv[2]  + (start_y**2)*right_coeff_inv[3]+ start_y*right_coeff_inv[4] + right_coeff_inv[5]))
            right_start_x = int(pos((end_y**5) * right_coeff_inv[0]  + (end_y**4)*right_coeff_inv[1] + (end_y**3) * right_coeff_inv[2]  + (end_y**2)*right_coeff_inv[3]+ end_y*right_coeff_inv[4] + right_coeff_inv[5]))

            left_x = np.linspace(left_start_x,left_end_x,50,endpoint=True,dtype=np.int32)
            right_x = np.linspace(right_start_x,right_end_x,50,endpoint=True,dtype=np.int32)

            left_y = [(i**5) * left_coeff[0] + (i**4)*left_coeff[1] + (i**3) * left_coeff[2] + (i**2)*left_coeff[3] + i*left_coeff[4] + left_coeff[5] for i in left_x]
            right_y = [(i**5) * right_coeff[0] + (i**4)*right_coeff[1] + (i**3) * right_coeff[2] + (i**2)*right_coeff[3] + i*right_coeff[4] + right_coeff[5] for i in right_x]

            left_start_x = int(pos((start_y**2) * left_coeff_inv[0]  + start_y*left_coeff_inv[1] + left_coeff_inv[2]))
            left_end_x = int(pos((end_y**2) * left_coeff_inv[0]  + end_y*left_coeff_inv[1] + left_coeff_inv[2]))
            right_end_x = int(pos((start_y**2) * right_coeff_inv[0]  + start_y*right_coeff_inv[1] + right_coeff_inv[2]))
            right_start_x = int(pos((end_y**2) * right_coeff_inv[0]  + end_y*right_coeff_inv[1] + right_coeff_inv[2]))

            left_x = np.linspace(left_start_x,left_end_x,50,endpoint=True,dtype=np.int32)
            right_x = np.linspace(right_start_x,right_end_x,50,endpoint=True,dtype=np.int32)

            left_y = [(i**2) * left_coeff[0] + i*left_coeff[1] + left_coeff[2] for i in left_x]
            right_y = [(i**2) * right_coeff[0] + i*right_coeff[1] + right_coeff[2] for i in right_x]

            left_start_x = int(pos((start_y**3) * left_coeff_inv[0]  + (start_y**2)*left_coeff_inv[1] + start_y*left_coeff_inv[2]+left_coeff_inv[3]))
            left_end_x = int(pos((end_y**3) * left_coeff_inv[0]  + (end_y**2)*left_coeff_inv[1] + end_y*left_coeff_inv[2]+left_coeff_inv[3]))
            right_end_x = int(pos((start_y**3) * right_coeff_inv[0]  + (start_y**2)*right_coeff_inv[1] + start_y*right_coeff_inv[2]+right_coeff_inv[3]))
            right_start_x = int(pos((end_y**3) * right_coeff_inv[0]  + (end_y**2)*right_coeff_inv[1] + end_y*right_coeff_inv[2]+right_coeff_inv[3]))

            left_x = np.linspace(left_start_x,left_end_x,50,endpoint=True,dtype=np.int32)
            right_x = np.linspace(right_start_x,right_end_x,50,endpoint=True,dtype=np.int32)

            left_y = [(i**3) * left_coeff[0] + (i**2)*left_coeff[1] + i*left_coeff[2] + left_coeff[3] for i in left_x]
            right_y = [(i**3) * right_coeff[0] + (i**2)*right_coeff[1] + i*right_coeff[2] + right_coeff[3] for i in right_x]

            left_lane = [[i,j] for i,j in zip(left_x,left_y)]
            right_lane = [[i,j] for i,j in zip(right_x,right_y)]

            frame = cv2.resize(image,(256,256))
            cv2.polylines(img1, np.int32([left_lane]), isClosed=False, color=(0,0,255),thickness=3)
            cv2.polylines(img1, np.int32([right_lane]), isClosed=False, color=(0,0,255),thickness=3)'''
            cv2.imshow('image',img1)
            if save:
                out.write(img1)

        elif l == 2:
            final1 = []
            final2 = []
            for i,j in zip(labels,final):
                if i==0:
                    final1.append(j)
                elif i==1:
                    final2.append(j)
                else:
                    continue

            km1 = KMeans(n_clusters=1).fit(final1)
            km2 = KMeans(n_clusters=1).fit(final2)
            if np.ravel(km1.cluster_centers_)[0] > np.ravel(km2.cluster_centers_)[0]:
                final1,final2 = final2,final1

            final1_x = [i[0] for i in final1]
            final1_y = [i[1] for i in final1]
            final2_x = [i[0] for i in final2]
            final2_y = [i[1] for i in final2]

            a = [i for i in sorted(list(zip(final1_x,final1_y)))]
            b = [i for i in sorted(list(zip(final2_x,final2_y)))]

            a1,a2 = [],[]
            for i in range(1,len(a)+1):
                if i == len(a):
                    break
                if a[i-1][0] == a[i][0]:
                    continue
                else:
                    a1.append(a[i])
            final1_x,final1_y = [i[0] for i in a1],[i[1] for i in a1]
            a1 = [i for i in sorted(list(zip(final1_y,final1_x)))]
            for i in range(1,len(a1)+1):
                if i == len(a1):
                    break
                if a1[i-1][0] == a1[i][0]:
                    continue
                else:
                    a2.append(a1[i])
            a3 = sorted([(i[1],i[0]) for i in a2])
            left_x,left_y = [i[0] for i in a3],[i[1] for i in a3]

            b1,b2 = [],[]
            for i in range(1,len(b)+1):
                if i == len(b):
                    break
                if b[i-1][0] == b[i][0]:
                    continue
                else:
                    b1.append(b[i])
            final2_x,final2_y = [i[0] for i in b1],[i[1] for i in b1]
            b1 = [i for i in sorted(list(zip(final2_y,final2_x)))]
            for i in range(1,len(b1)+1):
                if i == len(b1):
                    break
                if b1[i-1][0] == b1[i][0]:
                    continue
                else:
                    b2.append(b1[i])
            b3 = sorted([(i[1],i[0]) for i in b2])
            right_x,right_y = [i[0] for i in b3],[i[1] for i in b3]

            t1, c1, k1 = interpolate.splrep(left_x,left_y, s=s, k=k)
            t2, c2, k2 = interpolate.splrep(right_x,right_y, s=s, k=k)
            
            spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)
            spline2 = interpolate.BSpline(t2, c2, k2, extrapolate=False)

            x1min, x1max = np.array(left_x).min(), np.array(left_x).max()
            x2min, x2max = np.array(right_x).min(), np.array(right_x).max()

            xx1 = np.linspace(x1min, x1max, interval_pts)
            xx2 = np.linspace(x2min, x2max, interval_pts)

            left_lane = [[i,j] for i,j in zip(xx1,spline1(xx1))]
            right_lane = [[i,j] for i,j in zip(xx2,spline2(xx2))]

            cv2.polylines(img1, np.int32([left_lane]), isClosed=False, color=(255,0,0),thickness=3)
            cv2.polylines(img1, np.int32([right_lane]), isClosed=False, color=(0,0,255),thickness=3)
            '''left_coeff = np.polyfit(final1_x,final1_y,5)
            right_coeff = np.polyfit(final2_x,final2_y,5)

            left_coeff_inv = np.polyfit(final1_y,final1_x,5)
            right_coeff_inv = np.polyfit(final2_y,final2_x,5)

            left_coeff = np.polyfit(final1_x,final1_y,2)
            right_coeff = np.polyfit(final2_x,final2_y,2)

            left_coeff_inv = np.polyfit(final1_y,final1_x,2)
            right_coeff_inv = np.polyfit(final2_y,final2_x,2)

            left_coeff = np.polyfit(final1_x,final1_y,3)
            right_coeff = np.polyfit(final2_x,final2_y,3)

            left_coeff_inv = np.polyfit(final1_y,final1_x,3)
            right_coeff_inv = np.polyfit(final2_y,final2_x,3)

            left_start_x = int(pos((start_y**5) * left_coeff_inv[0]  + (start_y**4)*left_coeff_inv[1] + (start_y**3) * left_coeff_inv[2]  + (start_y**2)*left_coeff_inv[3]+ start_y*left_coeff_inv[4] + left_coeff_inv[5]))
            left_end_x = int(pos((end_y**5) * left_coeff_inv[0]  + (end_y**4)*left_coeff_inv[1] + (end_y**3) * left_coeff_inv[2]  + (end_y**2)*left_coeff_inv[3]+ end_y*left_coeff_inv[4] + left_coeff_inv[5]))
            right_end_x = int(pos((start_y**5) * right_coeff_inv[0]  + (start_y**4)*right_coeff_inv[1] + (start_y**3) * right_coeff_inv[2]  + (start_y**2)*right_coeff_inv[3]+ start_y*right_coeff_inv[4] + right_coeff_inv[5]))
            right_start_x = int(pos((end_y**5) * right_coeff_inv[0]  + (end_y**4)*right_coeff_inv[1] + (end_y**3) * right_coeff_inv[2]  + (end_y**2)*right_coeff_inv[3]+ end_y*right_coeff_inv[4] + right_coeff_inv[5]))

            left_x = np.linspace(left_start_x,left_end_x,50,endpoint=True,dtype=np.int32)
            right_x = np.linspace(right_start_x,right_end_x,50,endpoint=True,dtype=np.int32)

            left_y = [(i**5) * left_coeff[0] + (i**4)*left_coeff[1] + (i**3) * left_coeff[2] + (i**2)*left_coeff[3] + i*left_coeff[4] + left_coeff[5] for i in left_x]
            right_y = [(i**5) * right_coeff[0] + (i**4)*right_coeff[1] + (i**3) * right_coeff[2] + (i**2)*right_coeff[3] + i*right_coeff[4] + right_coeff[5] for i in right_x]

            left_start_x = int(pos((start_y**2) * left_coeff_inv[0]  + start_y*left_coeff_inv[1] + left_coeff_inv[2]))
            left_end_x = int(pos((end_y**2) * left_coeff_inv[0]  + end_y*left_coeff_inv[1] + left_coeff_inv[2]))
            right_end_x = int(pos((start_y**2) * right_coeff_inv[0]  + start_y*right_coeff_inv[1] + right_coeff_inv[2]))
            right_start_x = int(pos((end_y**2) * right_coeff_inv[0]  + end_y*right_coeff_inv[1] + right_coeff_inv[2]))

            left_x = np.linspace(left_start_x,left_end_x,50,endpoint=True,dtype=np.int32)
            right_x = np.linspace(right_start_x,right_end_x,50,endpoint=True,dtype=np.int32)

            left_y = [(i**2) * left_coeff[0] + i*left_coeff[1] + left_coeff[2] for i in left_x]
            right_y = [(i**2) * right_coeff[0] + i*right_coeff[1] + right_coeff[2] for i in right_x]

            left_start_x = int(pos((start_y**3) * left_coeff_inv[0]  + (start_y**2)*left_coeff_inv[1] + start_y*left_coeff_inv[2]+left_coeff_inv[3]))
            left_end_x = int(pos((end_y**3) * left_coeff_inv[0]  + (end_y**2)*left_coeff_inv[1] + end_y*left_coeff_inv[2]+left_coeff_inv[3]))
            right_end_x = int(pos((start_y**3) * right_coeff_inv[0]  + (start_y**2)*right_coeff_inv[1] + start_y*right_coeff_inv[2]+right_coeff_inv[3]))
            right_start_x = int(pos((end_y**3) * right_coeff_inv[0]  + (end_y**2)*right_coeff_inv[1] + end_y*right_coeff_inv[2]+right_coeff_inv[3]))

            left_x = np.linspace(left_start_x,left_end_x,50,endpoint=True,dtype=np.int32)
            right_x = np.linspace(right_start_x,right_end_x,50,endpoint=True,dtype=np.int32)

            left_y = [(i**3) * left_coeff[0] + (i**2)*left_coeff[1] + i*left_coeff[2] + left_coeff[3] for i in left_x]
            right_y = [(i**3) * right_coeff[0] + (i**2)*right_coeff[1] + i*right_coeff[2] + right_coeff[3] for i in right_x]

            left_lane = [[i,j] for i,j in zip(left_x,left_y)]
            right_lane = [[i,j] for i,j in zip(right_x,right_y)]

            frame = cv2.resize(image,(256,256))
            cv2.polylines(img1, np.int32([left_lane]), isClosed=False, color=(0,0,255),thickness=3)
            cv2.polylines(img1, np.int32([right_lane]), isClosed=False, color=(0,0,255),thickness=3)'''
            cv2.imshow('image',img1)
            if save:
                out.write(img1)

        elif l == 1:
            final_x = [i[0] for i in final]
            final_y = [i[1] for i in final]

            a = [i for i in sorted(list(zip(final_x,final_y)))]
            a1,a2 = [],[]
            for i in range(1,len(a)+1):
                if i == len(a):
                    break
                if a[i-1][0] == a[i][0]:
                    continue
                else:
                    a1.append(a[i])
            final_x,final_y = [i[0] for i in a1],[i[1] for i in a1]
            a1 = [i for i in sorted(list(zip(final_y,final_x)))]
            for i in range(1,len(a1)+1):
                if i == len(a1):
                    break
                if a1[i-1][0] == a1[i][0]:
                    continue
                else:
                    a2.append(a1[i])
            a3 = sorted([(i[1],i[0]) for i in a2])
            left_x,left_y = [i[0] for i in a3],[i[1] for i in a3]

            t1, c1, k1 = interpolate.splrep(left_x,left_y, s=s, k=k)    
            spline1 = interpolate.BSpline(t1, c1, k1, extrapolate=False)
            x1min, x1max = np.array(left_x).min(), np.array(left_x).max()
            xx1 = np.linspace(x1min, x1max, interval_pts)
            lane = [[i,j] for i,j in zip(xx1,spline1(xx1))]
            km = KMeans(n_clusters=1).fit(final)
            if np.ravel(km.cluster_centers_)[0] >= 128:
                cv2.polylines(img1, np.int32([lane]), isClosed=False, color=(0,0,255),thickness=3)
            else:
                cv2.polylines(img1, np.int32([lane]), isClosed=False, color=(255,0,0),thickness=3)
            '''coeff = np.polyfit(final_x,final_y,2)
            
            coeff_inv = np.polyfit(final_y,final_x,2)

            start_x = int(pos((start_y**5) * coeff_inv[0]  + (start_y**4)*coeff_inv[1] + (start_y**3) * coeff_inv[2]  + (start_y**2)*coeff_inv[3]+ start_y*coeff_inv[4] + coeff_inv[5]))
            end_x = int(pos((end_y**5) * coeff_inv[0]  + (end_y**4)*coeff_inv[1] + (end_y**3) * coeff_inv[2]  + (end_y**2)*coeff_inv[3]+ end_y*coeff_inv[4] + coeff_inv[5]))
            
            x = np.linspace(start_x,end_x,50,endpoint=True,dtype=np.int32)
            y = [(i**5) * coeff[0] + (i**4)*coeff[1] + (i**3) * coeff[2] + (i**2)*coeff[3] + i*coeff[4] + coeff[5] for i in x]
            
            start_x = int(pos((start_y**2) * coeff_inv[0]  + start_y*coeff_inv[1] + coeff_inv[2]))
            end_x = int(pos((end_y**2) * coeff_inv[0]  + end_y*coeff_inv[1] + coeff_inv[2]))
            
            x = np.linspace(start_x,end_x,50,endpoint=True,dtype=np.int32)
            y = [(i**2) * coeff[0] + i*coeff[1] + coeff[2] for i in x]
            
            lane = [[i,j] for i,j in zip(x,y)]

            frame = cv2.resize(frame,(256,256))
            cv2.polylines(img1, np.int32([lane]), isClosed=False, color=(0,0,255),thickness=3)'''
            cv2.imshow('image',img1)
            if save:
                out.write(img1)            
    else:
        print('Cannot capture input!!')
        break
cap.release()
cv2.destroyAllWindows()
