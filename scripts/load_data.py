import os
import pandas as pd
from scipy.io import loadmat

# 拿到所有.mat文件
def get_file_name(file_dir):
    fileList = []
    dirList = []
    for root, dirs, files in os.walk(file_dir):
        for dir in dirs:
            dirList.append(dir)
        for file in files:
            if os.path.splitext(file)[1] == '.mat':  # os.path.splitext()函数将路径拆分为文件名+扩展名
                fileList.append(os.path.join(root, file))
    return dirList, fileList


_dirlist, _filelist = get_file_name(r"/home/xtc/文档/FFT+CNN+Trans/data_ultra")
print(_dirlist)
print(_filelist)
# sample
data0 = loadmat(f'{_filelist[0]}')  # 读取MAT文件
print(list(data0.keys()))
data0_datacount = data0['DataCount']
data0_linecount = data0['LineCount']
data0_data = data0['Data']
# 采用驱动端数据

data_df = pd.DataFrame()
for index in range(len(_filelist)):
    # 读取MAT文件
    data = loadmat(f'{_filelist[index]}')
    dataList = data['Data'].reshape(-1)
    data_df[_dirlist[index]] = dataList[:50000]
print(data_df.shape)
data_df.set_index(_dirlist[0],inplace=True)#由于csv会把df的index设为第一列
data_df.to_csv('data_all.csv')

