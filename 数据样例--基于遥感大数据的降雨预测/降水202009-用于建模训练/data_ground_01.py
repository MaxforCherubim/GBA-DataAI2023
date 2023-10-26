txtfile=r'/Users/zhangyingtan/Desktop/同步空间/iCloud禁止同步文件夹.nosync/GBA-DataAI/数据样例--基于遥感大数据的降雨预测/降水202009-用于建模训练/pre_20200901.csv'
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
raw=pd.read_csv(txtfile,encoding='gb2312')
Station_num=int(raw.shape[0])//24
ground=np.zeros((24,Station_num,4+7))
for t in range(24):
    for i in range(Station_num):
        ground[t,i]=raw.drop(['Station_Name','Year','Mon','Day','Hour'],axis=1).iloc[t*Station_num+i,:]
print(ground.shape)
print(raw.Lat.max(),raw.Lat.min())
print(raw.Lon.max(),raw.Lon.min())
LLA=pd.DataFrame(ground[0][:,1:4],columns=['Lat','Lon','Alti'])
print(LLA)
LLA.to_csv('751_LLAs.csv',index=False)