# 画二维坐标图
# 读取csv并作图
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np

readData = pd.read_csv("./data/计算数据.csv", header=None)  # 读取csv数据

data_x = readData.iloc[:, 0].tolist()  # 产生横轴坐标
data_y1 = readData.iloc[:, 1].tolist()  # 获取dataFrame中的第3列，并将此转换为list
data_y2 = readData.iloc[:, 2].tolist()  # 获取dataFrame中的第3列，并将此转换为list
data_y3 = readData.iloc[:, 3].tolist()  # 获取dataFrame中的第3列，并将此转换为list
data_y4 = readData.iloc[:, 4].tolist()  # 获取dataFrame中的第3列，并将此转换为list

img1 = plt.figure(figsize=(8.5, 6), dpi=300)
img1.add_axes([0.05, 0.05, 0.94, 0.94])  # [距离左边，下边，坐标轴宽度，坐标轴高度] 范围(0, 1)
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
config = {
            "font.family": 'serif',
            "font.size": 14,
            "mathtext.fontset": 'stix',
            # "font.serif": ['SimSun'],
            "font.serif": ['Times New Roman'],
         }
rcParams.update(config)


x_ticks = np.arange(0, 210, 10)
y_ticks = np.arange(0, 500, 50)

plt.xticks(x_ticks, size=12)
plt.yticks(y_ticks, size=12)

plt.plot(data_x, data_y1, linewidth=1, linestyle='-')
plt.plot(data_x, data_y2, linewidth=1, linestyle='-')
plt.plot(data_x, data_y3, linewidth=1, linestyle='-')
plt.plot(data_x, data_y4, linewidth=1, linestyle='-')
# plt.title("奖励和配餐分数变化")  # 设置标题
# plt.xlabel("步数")  # 横轴名称
# plt.ylabel("分数")  # 纵轴名称

# lab2 = ["最小损失值", "最大损失值", "top10平均损失值", "平均损失值"]
lab2 = ["Minimum loss value", "Maximum loss value", "Top10 average loss value", "Average loss value"]
plt.legend(lab2, labelspacing=0.5, columnspacing=0.1, fontsize=12)  # 按照顺序添加图例

img1.savefig("./img/02.png")
# plt.show()  # 画图
