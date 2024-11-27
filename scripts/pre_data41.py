from joblib import dump
import numpy as np
import torch
import matplotlib.pyplot as plt
# 对于上述表格数据，要重新制作数据集，制作思路：
# 1. 对于上述表格每一列 数据，信号长度为2000,每一类区500次测试信号
# 2. 信号数据放在另外的表格中，一行对应一个信号，而且在末尾添加分类标签

import os
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 拿到所有.csv文件
def get_file_name(file_dir):
    fileList = []
    dirList = []
    for root, dirs, files in os.walk(file_dir):
        for dir in dirs:
            dirList.append(dir)
        for file in files:
            if os.path.splitext(file)[1] == '.csv':  # os.path.splitext()函数将路径拆分为文件名+扩展名
                fileList.append(os.path.join(root, file))
    return dirList, fileList

def normalize(data):
    ''' (0,1)归一化
        参数:一维时间序列数据
    '''
    s = (data - min(data)) / (max(data) - min(data))
    return s

def make_data_labels(dataframe):
    '''
        参数 dataframe: 数据框
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    # 信号值
    x_data = dataframe.iloc[:-1,:].T
    # 标签值
    y_label = dataframe.iloc[-1,:]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64')) # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return x_data, y_label

split_rate=[0.7,0.2,0.1]
_dirlist, _filelist = get_file_name(r"./data3")
sample_num=700
sampele_length=512
def make_and_split_data(start,end,sampele_num=sample_num,sampele_length=sampele_length):
    #起始点的判定
    data0=pd.read_csv(_filelist[0], header=None).iloc[:sampele_num,start:end]
    data0.columns=range(data0.shape[1])
    sample_start = 0
    ndata0=pd.DataFrame(columns=data0.columns)
    for i in data0.columns:
        sample_start=150
        ndata0[i] =normalize(data0.iloc[sample_start:sample_start + sampele_length, i]).tolist()
    shape0 = ndata0.shape
    label0 = pd.DataFrame([0] * shape0[1]).T
    label0.columns = list(ndata0)
    ndata0 = pd.concat([ndata0, label0], ignore_index=True, axis=0)
    all_data=ndata0

    for k in range(1,len(_filelist)):
        data = pd.read_csv(_filelist[k], header=None).iloc[:sampele_num,start:end]
        data.columns = range(data.shape[1])
        ndata = pd.DataFrame(columns=data.columns)
        sample_start = 0
        for x in data.columns:
            sample_start=150
            ndata[x] = normalize(data.iloc[sample_start:sample_start + sampele_length, x]).tolist()
        shape = ndata.shape
        label = pd.DataFrame([k] * shape[1]).T
        label.columns = list(ndata)
        ndata = pd.concat([ndata, label], ignore_index=True, axis=0)
        all_data=pd.concat([all_data,ndata],ignore_index=True,axis=1)
        all_data = all_data.dropna(axis=1, how='any')
    print(all_data)
    return make_data_labels(all_data)
num_per_class=30
train_xdata,train_ylabel=make_and_split_data(0,int(num_per_class*split_rate[0]))
val_xdata,val_ylabel=make_and_split_data(int(num_per_class*split_rate[0]),int(num_per_class*split_rate[0])+int(num_per_class*split_rate[1]))
test_xdata,test_ylabel=make_and_split_data(int(num_per_class*split_rate[0])+int(num_per_class*split_rate[1]),num_per_class)

# for i in range(9):
#     sample=test_xdata[i*3]
#     plt.plot(sample)
#     plt.xlabel("time")
#     plt.ylabel("Voltage")
#     plt.show()
#     fft_result1 = np.fft.fft(sample)#fft之后的单位问题
#     plt.plot(np.abs(fft_result1))
#     plt.show()


# sample=test_xdata[8*3]
# plt.plot(sample)
# plt.xlabel("time")
# plt.ylabel("Voltage")
# plt.savefig('glass.svg',dpi=600)
# plt.show()
# fft_result1 = np.fft.fft(sample)#fft之后的单位问题
# plt.plot(np.abs(fft_result1))
# plt.show()
# dump(train_xdata, './data41/trainX_1024_10c')
# dump(val_xdata, './data41/valX_1024_10c')
# dump(test_xdata, './data41/testX_1024_10c')
# dump(train_ylabel, './data41/trainY_1024_10c')
# dump(val_ylabel, './data41/valY_1024_10c')
# dump(test_ylabel, './data41/testY_1024_10c')

# import matplotlib.pyplot as plt
# from pyts.image import MarkovTransitionField
# '''
# 读取时间序列的数据
# 怎么读取需要你自己写
# X为ndarray类型数据
# '''
# X=train_xdata[0].reshape(1,-1)
# # MTF transformation
# mtf = MarkovTransitionField(image_size=512)
# X_mtf = mtf.fit_transform(X)

# # Show the image for the first time series
# plt.figure(figsize=(5, 5))
# plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
# plt.title('Markov Transition Field', fontsize=18)
# plt.colorbar(fraction=0.0457, pad=0.04)
# plt.tight_layout()
# plt.show()
# plt.savefig(r"./MKFdata/24",dpi=600)