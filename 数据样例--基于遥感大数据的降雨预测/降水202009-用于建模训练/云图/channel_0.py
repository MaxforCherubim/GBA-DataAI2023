from osgeo import gdal
import matplotlib.pyplot as plt
import cv2
import numpy as np

fast=cv2.FastFeatureDetector_create()
sift=cv2.SIFT_create()
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

com_name='AHI8_OBI_2000M_NOM_'
time_point='20200901'
name_list=['_0000','_0020','_0040']
img=[]
kp=[]
des=[]

for i,name in zip([0,1,2],name_list):
    raster=gdal.Open(com_name+time_point+name+'.tif')
    odata=raster.ReadAsArray()
    array0=odata[13]
    #TODO 储存为图片
    plt.imsave(time_point+name+'[13]'+'.png',array0)
    #TODO 转化为灰度图img
    img.append(cv2.imread(time_point+name+'[13]'+'.png',0))
    #TODO 展示灰度图
    cv2.imshow('Image',img[i])
    cv2.waitKey(0)
    #TODO 储存灰度图
    cv2.imwrite(time_point+name+'[13]'+'_gray'+'.png',img[i])
    kps=fast.detect(img[i],None)
    img_key=cv2.drawKeypoints(img[i],kps,img[i])
    cv2.imshow('Image_key',img_key)
    cv2.waitKey(0)
    cv2.imwrite(time_point+name+'[13]'+'_key'+'.png',img_key)
    _,__=sift.compute(img[i],kps)
    kp.append(_)
    des.append(__.astype(np.uint8))
    
    if (i>=1):
        matches=bf.match(des[i-1],des[i])
        matches=sorted(matches,key=lambda x:x.distance)
        #! flags=4
        contrast=cv2.drawMatches(img[i-1],kp[i-1],img[i],kp[i],matches,img[i-1],flags=4)
        cv2.imshow('contrast',contrast)
        cv2.waitKey(0)
        cv2.imwrite(time_point+name+'[13]'+'_contrast'+'.png',contrast)