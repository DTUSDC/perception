import cv2
import numpy as np
import json
from os import listdir, mkdir
from os.path import isfile, join, exists
from PIL import Image

def ground_truth_image(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for file in files:
        json_data = [json.loads(line) for line in open(join(path,file),errors='ignore').readlines()]
        for row in json_data:
            data.append(row)
    print(len(data))
    if not exists('gt_images1'):
        mkdir('gt_images1')
        print('Directory Created!')
    kernel = np.ones((5,5),np.uint8)
    for j,gt in enumerate(data):
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']
        img = cv2.imread(raw_file)
        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
        img_vis = img * 0
        for lane in gt_lanes_vis:
            for point in lane:
                cv2.circle(img_vis, point,4,color=(255,255,255), thickness=-1)
        img1 = img_vis
        for i in range(5):
            np.random.seed(i)
            img1 = cv2.erode(img1,kernel,iterations=1)
            img1[600-30*i:650-40*i,:,:] = cv2.dilate(img1[600-30*i:650-40*i,:,:],kernel,iterations=2)
            img1[450+30*i:500+30*i,:,:] = cv2.dilate(img1[450+30*i:500+30*i,:,:],kernel,iterations=2)
            img1[300+10*i:350+10*i,:,:] = cv2.dilate(img1[300+10*i:350+10*i,:,:],kernel,iterations=2)
            img1[300+10*i:450+10*i,200-20*i:300-20*i] = img1[300:550,350+20*i:450+20*i] = img1[250+10*i:550+20*i,800+30*i:900+30*i] = 0
            img1[300+10*i:600,1050+10*i:1150+10*i] = 0
            img1[450-10*i:500-10*i,1+20*i:50+20*i] = img1[(650-20*i):,250+20*i:350+20*i] = 0
            img1[300:650,1200:] = 0
            noise2 = (np.random.normal(0,2,img_vis.shape)*255).astype(np.uint8)
            noise2 = cv2.threshold(noise2,252,255,cv2.THRESH_BINARY)[1]
            #noise2 = np.array(noise2[:,:,2])
            #noise2 = noise2.reshape([noise2.shape[0],noise2.shape[1],-1])
            #noise2 = np.concatenate([noise2,noise2,noise2],axis=2)
            noise2[300:550,100+200*i:200+200*i,:] = noise2[300:550,1000-100*i:1150-100*i:,:] = 0
            noise2[200+50*i:350+50*i,400:450:,:] = noise2[500:600,650:850:,:] = 0
            noise2[350:450,400+50*i:500+50*i,:] = noise2[200:350,700-150*i:850-150*i,:] = 0
            #copy = np.zeros(img[:320,:,:].shape)
            #new2 = np.concatenate([copy,noise2],axis=0)
            final2 = noise2 + img1
            final2[400-50*i:430-50*i,400+100*i:430+100*i] = final2[460+50*i:490+50*i,600+100*i:630+100*i] = 255
            final2[450-20*i:600-10*i,1100-150*i:1130-150*i] = 255
            final2[680-30*i:710-30*i,30+150*i:60+150*i] = 255
            final2[600-50*i:680-50*i,600+30*i:700+30*i] = 255
            img1 = img_vis
            result = Image.fromarray(final2.astype(np.uint8))
            result.save(join('noise_images1/'+str(j)+'_'+str(i)+'.jpg'))
        print('created noise for ' + str(j) + ' images')     
        #cv2.imwrite(join('gt_images1/',str(i)+'.jpg'),img_vis)
    print('Ground Truth Images Generated!')
    return

ground_truth_image('train_set')

