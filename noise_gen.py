import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir, mkdir
from os.path import isfile, join, exists, splitext
import re

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def noise(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = sorted_aphanumeric(files)
    kernel = np.ones((5,5),np.uint8)
    if not exists('noise_images'):
        mkdir('noise_images')
        print('Directory Created!')    
    for j,file in enumerate(files):
        img = plt.imread(join(path,file))*255
        img1 = img
        for i in range(5):
            np.random.seed(i)
            img1 = cv2.erode(img,kernel,iterations=1)
            img1[600-30*i:650-40*i,:,:] = cv2.dilate(img1[600-30*i:650-40*i,:,:],kernel,iterations=2)
            img1[450+30*i:500+30*i,:,:] = cv2.dilate(img1[450+30*i:500+30*i,:,:],kernel,iterations=2)
            img1[300+10*i:350+10*i,:,:] = cv2.dilate(img1[300+10*i:350+10*i,:,:],kernel,iterations=2)
            img1[300+10*i:450+10*i,200-20*i:300-20*i] = img1[300:550,350+20*i:450+20*i] = img1[250+10*i:550+20*i,800+30*i:900+30*i] = 0
            img1[300+10*i:600,1050+10*i:1150+10*i] = 0
            img1[450-10*i:500-10*i,1+20*i:50+20*i] = img1[(650-20*i):,250+20*i:350+20*i] = 0
            img1[300:650,1200:] = 0
            noise2 = (np.random.normal(0,2,img[320:,:,:].shape)*255).astype(np.uint8)
            noise2 = cv2.threshold(noise2,252,255,cv2.THRESH_BINARY)[1]
            noise2 = np.array(noise2[:,:,2])
            noise2 = noise2.reshape([noise2.shape[0],noise2.shape[1],-1])
            noise2 = np.concatenate([noise2,noise2,noise2],axis=2)
            noise2[100:350,100+200*i:200+200*i,:] = noise2[100:250,1000-100*i:1150-100*i:,:] = 0
            noise2[0+50*i:150+50*i,400:450:,:] = noise2[300:400,650:850:,:] = 0
            noise2[150:250,400+50*i:500+50*i,:] = noise2[0:150,700-150*i:850-150*i,:] = 0
            copy = np.zeros(img[:320,:,:].shape)
            new2 = np.concatenate([copy,noise2],axis=0)
            final2 = new2 + img1
            final2[400-50*i:430-50*i,400+100*i:430+100*i] = final2[460+50*i:490+50*i,600+100*i:630+100*i] = 255
            final2[450-20*i:600-10*i,1100-150*i:1130-150*i] = 255
            final2[680-30*i:710-30*i,30+150*i:60+150*i] = 255
            img1 = img
            result = Image.fromarray(final2.astype(np.uint8))
            result.save(join('noise_images/noise'+str(j)+'_'+str(i)+'.png'))
        print('created noise for ' + str(j) + ' images')
        
noise(r"gt_images")
