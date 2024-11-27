# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df1=pd.read_csv(r"./heatmap/bxg0.csv")
# weights=df1["weights"].values

# data1=pd.read_csv(r"./databxg/BXGa.csv").iloc[:,0].values

# def normalize(data):
#     ''' (0,1)归一化
#         参数:一维时间序列数据
#     '''
#     s = (data - min(data)) / (max(data) - min(data))
#     return s
# data1=normalize(data1[150:150+512])

# plt.plot(data1)





# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable

# # 读取权重和数据
# df1 = pd.read_csv(r"./heatmap/bxg0.csv")
# weights = df1["weights"].values

# data1 = pd.read_csv(r"./databxg/BXGa.csv").iloc[:, 0].values

# def normalize(data):
#     ''' (0,1)归一化
#         参数:一维时间序列数据
#     '''
#     s = (data - min(data)) / (max(data) - min(data))
#     return s

# data1 = normalize(data1[150:150+512])

# # 确保weights的长度与data1相同
# weights = weights[:len(data1)]  # 调整weights长度以匹配data1

# # 创建归一化对象和颜色映射
# norm = Normalize(vmin=min(weights), vmax=max(weights))
# cmap = plt.get_cmap('coolwarm')
# sm = ScalarMappable(norm=norm, cmap=cmap)

# # 绘制带有热力图颜色的散点图
# plt.figure(figsize=(10, 4))
# plt.scatter(range(len(data1)), data1, c=weights, cmap='coolwarm', s=10)  # s控制点的大小
# plt.colorbar(sm, label='Weight')
# plt.title('Time Series Data with Heatmap Colors')
# plt.xlabel('Time')
# plt.ylabel('Normalized Value')
# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.colors import Normalize

# # 读取权重和数据
# df1 = pd.read_csv(r"./heatmap/bxg0.csv")
# weights = df1["weights"].values

# data1 = pd.read_csv(r"./databxg/BXGa.csv").iloc[:, 0].values

# def normalize(data):
#     ''' (0,1)归一化
#         参数:一维时间序列数据
#     '''
#     s = (data - min(data)) / (max(data) - min(data))
#     return s

# data1 = normalize(data1[150:150+512])

# # 确保weights的长度与data1相同
# weights = weights[:len(data1)]  # 调整weights长度以匹配data1

# # 创建点对以绘制线段
# points = np.array([np.arange(len(data1)), data1]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)

# # 创建LineCollection对象
# norm = Normalize(vmin=min(weights), vmax=max(weights))
# lc = LineCollection(segments, cmap='coolwarm', norm=norm)
# lc.set_array(weights)  # 设置颜色映射的权重

# # 绘图
# plt.figure(figsize=(10, 4))
# plt.gca().add_collection(lc)
# plt.xlim(0, len(data1) - 1)
# plt.ylim(min(data1), max(data1))
# plt.colorbar(lc, label='Weight')
# plt.title('Time Series Data with Weighted Colors')
# plt.xlabel('Time')
# plt.ylabel('Normalized Value')
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# 读取权重和数据
df1 = pd.read_csv(r"./heatmap/bxg0.csv")
weights = df1["weights"].values




data1 = pd.read_csv(r"./databxg/BXGa.csv").iloc[:, 0].values

def normalize(data):
    ''' (0,1)归一化
        参数:一维时间序列数据
    '''
    s = (data - min(data)) / (max(data) - min(data))
    return s

data1 = normalize(data1[150:150+512])
weights=normalize(weights)
# 设置窗口长度
window_length = 128

# 计算所有可能窗口的权重之和
window_sums = np.convolve(weights, np.ones(window_length), 'valid')

# 找到权重之和最大的窗口的起始索引
max_sum_index = np.argmax(window_sums)

# 确保weights的长度与data1相同
weights = weights[:len(data1)]  # 调整weights长度以匹配data1

# 创建点对以绘制线段
points = np.array([np.arange(len(data1)), data1]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 创建LineCollection对象
norm = Normalize(vmin=min(weights), vmax=max(weights))
lc = LineCollection(segments, cmap='coolwarm', norm=norm)
lc.set_array(weights)  # 设置颜色映射的权重

# 绘图
plt.figure(figsize=(10, 4))
plt.gca().add_collection(lc)
plt.xlim(0, len(data1) - 1)
plt.ylim(min(data1), max(data1))

# 添加两个竖直线来标示窗口
plt.axvline(x=max_sum_index, color='green', linestyle='--', label='Window Start')
plt.axvline(x=max_sum_index + window_length, color='red', linestyle='--', label='Window End')

plt.colorbar(lc, label='Weight')
plt.title('Time Series Data with Weighted Colors')
plt.xlabel('Time')
plt.ylabel('Normalized Value')

plt.savefig(r"./heat0.svg",dpi=600)