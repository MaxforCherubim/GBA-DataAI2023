from osgeo import gdal
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

fast=cv2.FastFeatureDetector_create()
sift=cv2.SIFT_create()
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def line_match(array,list):
    row_index=[]
    array1=array
    array2=np.array(list)
    for i in range(array1.shape[0]):
        for j in range(array2.shape[0]):
            if (array1[i]==array2[j]).all():
                row_index.append(j)
    return row_index

com_name='AHI8_OBI_2000M_NOM_'
time_point='20200901'
name_list=['_0000','_0020','_0040']
input_channel=input('Channel=')
channel=int(input_channel)
img=[]
kp=[]
des=[]
desc=[]
locs=[]
trace_list=[]

for i,name in zip([0,1,2],name_list):

    row_id=1001
    col_id=601

    raster=gdal.Open(com_name+time_point+name+'.tif')
    odata=raster.ReadAsArray()
    array0=odata[channel]
    
    #TODO 储存为图片
    plt.imsave(time_point+name+input_channel+'.png',array0)
    #TODO 转化为灰度图img
    img.append(cv2.imread(time_point+name+input_channel+'.png',0))
    #TODO 储存灰度图
    cv2.imwrite(time_point+name+input_channel+'_gray'+'.png',img[i])
    
    kps=fast.detect(img[i],None)
    img_rawkey=cv2.drawKeypoints(img[i],kps,img[i])
    #TODO 储存特征点图
    cv2.imwrite(time_point+name+input_channel+'_rawkey'+'.png',img_rawkey)
    
    _,__=sift.compute(img[i],kps)
    kp.append(_)
    #TODO _.pt返回关键点的坐标
    locs.append([(lambda x:x.pt) (x) for x in _])
    #TODO 输出des是128维关键点向量
    des.append(__.astype(np.uint8))
    #TODO 对每个128维关键点向量单位化
    desc.append(np.array([d/np.linalg.norm(d) for d in des[i]]))
    
    if (i>=1):

        matchscores_list=[]

        matches=bf.match(des[i-1],des[i])
        matches=sorted(matches,key=lambda x:x.distance)
        #! flags=4
        contrast=cv2.drawMatches(img[i-1],kp[i-1],img[i],kp[i],matches,img[i-1],flags=4)
        cv2.imwrite(time_point+name+input_channel+'_rawcontrast'+'.png',contrast)
        #! 开始水平连线
        matchscores=np.zeros((desc[i-1].shape[0]),'int')
        for j in range(desc[i-1].shape[0]):
            #TODO dotprods.shape=(desc[i].shape[0],)
            dotprods=np.dot(desc[i-1][j,:],desc[i].T)
            #TODO desc都是单位向量，所以内积即cos(theta)，计算arccos即计算theta角
            index_angles=np.argsort(np.arccos(dotprods))
            if np.arccos(dotprods)[index_angles[0]]<0.95*np.arccos(dotprods)[index_angles[1]]:
                #TODO 记录图1，2的关键点，图1中单独一点对应另外全部点构成的角度
                #TODO 如果前两个最小角够接近，比值大于等于0.95则记为0，反之则直接记录最小角索引
                #TODO matchscores[i]中元素index与value有关系
                matchscores[j]=int(index_angles[0])
        matchscores_list.append(matchscores)

        matchscores=np.zeros((desc[i].shape[0]),'int')
        for j in range(desc[i].shape[0]):
            dotprods=np.dot(desc[i][j,:],desc[i-1].T)
            index=np.argsort(np.arccos(dotprods))
            if np.arccos(dotprods)[index[0]]<0.95*np.arccos(dotprods)[index[1]]:
                matchscores[j]=int(index[0])
        matchscores_list.append(matchscores)
        #TODO nonzero()以元组形式返回矩阵中非零元素的行列位置，[0]行[1]列
        #TODO 对于列向量.nonzero()直接返回[0]列位置，[1]空
        ndx=matchscores_list[0].nonzero()[0]
        
        for n in ndx:
            #TODO 1to2非零行位置n
            if matchscores_list[1][int(matchscores_list[0][n])] != n:
                matchscores_list[0][n]=0
        good_match=matchscores_list[0]

        #TODO 检索locs[i-1]中图一满足条件的关键点坐标
        src_pts=np.asarray([locs[i-1][j] for j in np.where(good_match>0)[0]])
        #TODO 检索locs[i]中图二满足条件的关键点坐标
        dst_pts=np.asarray([locs[i][j] for j in good_match[good_match>0]])
        disp=dst_pts-src_pts
        #TODO 检索匹配关键点x，y方向绝对变动和不超过20
        remove_ind=np.where(np.sum(abs(disp),axis=1)<20)
        src_pts=src_pts[remove_ind]
        dst_pts=dst_pts[remove_ind]
        disp=disp[remove_ind]
        
        line_match1=line_match(src_pts,locs[i-1])
        kps1=tuple([kp[i-1][j] for j in line_match1])
        #img_key1=cv2.drawKeypoints(img[i-1],kps1,img[i-1])
        #cv2.imwrite(time_point+name+input_channel+'_key1'+'.png',img_key1)
        kp1,des1=sift.compute(img[i-1],kps1)
        des1=des1.astype(np.uint8)
        line_match2=line_match(dst_pts,locs[i])
        kps2=tuple([kp[i][j] for j in line_match2])
        kp2,des2=sift.compute(img[i],kps2)
        des2=des2.astype(np.uint8)
        matches=bf.match(des1,des2)
        matches=sorted(matches,key=lambda x:x.distance)
        #! flags=2
        contrast=cv2.drawMatches(img[i-1],kp1,img[i],kp2,matches,img[i-1],flags=2)
        cv2.imwrite(time_point+name+input_channel+'_contrast'+'.png',contrast)

        #%%
        for j in range(3):
            trace_list.append([row_id,col_id])
            dist=np.sqrt(np.sum(([row_id,col_id]-dst_pts)**2,axis=1))
            useful_index=dist.argsort()[0:9]
            [row_shift,col_shift]=np.median(disp[useful_index],axis=0)
            #TODO math.isnan()返回布尔值，非数字返回True否则返回Flase
            if ((math.isnan(row_shift))|(math.isnan(col_shift))):
                row_shift,col_shift=0,0
            row_id=row_id-row_shift
            col_id=col_id-col_shift
        trace_list=np.asarray(trace_list,dtype='int')