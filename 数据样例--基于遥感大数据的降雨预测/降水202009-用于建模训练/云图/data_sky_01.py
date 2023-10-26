from osgeo import gdal
raster=gdal.Open('AHI8_OBI_2000M_NOM_20200901_0000.tif')
odata=raster.ReadAsArray()
print('数据维度',odata.shape)
print('矩阵行列',raster.RasterYSize,raster.RasterXSize)
transform=raster.GetGeoTransform()
print("空间信息",transform)
#!左上角经度，X方向分辨率，图像旋转系数，左上角纬度，图像旋转系数，Y方向分辨率（负数） 