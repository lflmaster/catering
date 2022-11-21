# 画二维坐标图
# 读取csv并作图
import random

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import torch


class getData:
    def __init__(self):
        dir_path1 = '../source/'
        # 参考摄入量
        self.file1 = pd.read_csv(dir_path1 + '参考摄入量.csv')
        # 食物成分数据
        self.file2 = pd.read_csv(dir_path1 + '食物成分数据.csv')
        # 读取结果数据
        self.file3 = pd.read_csv('./data/配餐结果.csv')
        # 参考摄入量
        self.file4 = pd.read_csv(dir_path1 + '用户信息.csv')
        # 健康养生
        self.file5 = pd.read_csv(dir_path1 + '健康养生.csv')


# 评估膳食宝塔每层食材的重量是否达标
# 评估一天的食材的能量及三种能量营养素是否达标
class dayShicai(object):
    def __init__(self, age, sex, strength, userSidList):
        # 初始化个人信息
        self.sc1g = None
        self.age = age
        self.sex = sex
        self.strength = strength
        self.userSid = userSidList  # 用户配餐食材的id列表
        # 初始化其他数据
        self.allSciD = None  # 保存所有食材的类别号和名称
        self.pogoda = None  # 保存膳食宝塔每层的数据
        self.allSidScid = {}  # 所有食材id和类别id组成的字典
        self.userSN = []  # 用户当日所有的食材名称
        self.userScid = []  # 用户当日所有食材对应的类别id
        self.userSidScid = {}  # 用户当日所有食材id和类别id组成的字典
        self.userSepfc = []  # 用户当日所有食材对应的能量、蛋白蛋、脂肪、碳水化合物的数据
        self.dayEpfc = []  # 该用户一天需要摄入的能量、蛋白蛋、脂肪、碳水化化的量
        self.dayEnergy = []  # 能量 千卡
        self.dayCho = []  # 碳水化化物 克
        self.dayPro = []  # 蛋白蛋 克
        self.dayFat = []  # 脂肪 克
        self.weightPogoda = []  # 膳食宝塔中各层级的重量标准
        self.userPogoda = []  # 用户当日所有食材在膳食宝塔中对应的层级的索引
        self.setData()  # 调用函数
        # 初始化GA算法参数
        self.pop_size = 100  # 一个种群中个体的个数
        self.x_num = len(userSidList)  # 自变量个数
        self.x_min = 0  # 自变量最小值
        self.x_max = 250  # 自变量最大值
        self.psum = 1  # 画出前psum个方案
        self.N_GENERATIONS = 200  # 迭代次数

    def setData(self):
        # Sid:食材id  Scid:食材类id  Sepfc:食材的能量、蛋白蛋、脂肪、碳水化化物  userSweight:食材的重量
        self.allSciD = {'1': '谷类及制品', '2': '薯类、淀粉及制品', '3': '干豆类及制品', '4': '蔬菜类及制品', '5': '菌藻类',
                        '6': '水果类及制品', '7': '坚果、种子类', '8': '畜肉类及制品', '9': '禽肉类及制品', '10': '乳类及制品',
                        '11': '蛋类及制品', '12': '鱼虾蟹贝类', '13': '婴幼儿食品', '14': '小吃、甜饼', '15': '速食食品',
                        '16': '软料类', '17': '含酒精饮料', '18': '糖、果脯和蜜饯、蜂蜜', '19': '油脂类', '20': '调味品类',
                        '21': '药食及其它'}
        # 膳食宝塔：一般健康成人每日对各类食物适宜的摄入量范围  第一列是类别，第二列是重量
        self.pogoda = [
            [[1], [50, 150]],  # 全谷物和杂豆：50-150
            [[2], [50, 100]],  # 薯类：50-100克
            [[1, 2], [250, 400]],  # 谷薯类：250~400克
            [[3, 7], [25, 35]],  # 大豆及坚果类：25-35克
            [[4], [300, 500]],  # 蔬菜类：300-500克
            [[6], [200, 350]],  # 水果类：200-350克
            [[8, 9], [40, 75]],  # 畜禽肉：40-75克
            [[10], [300, 300]],  # 奶及奶制品：300克
            [[11], [40, 50]],  # 蛋类：40~50克
            [[12], [40, 75]],  # 水产品：40-75克
            [[19], [25, 30]],  # 油：25-30克
            [[20], [0, 6]],  # 盐：<6克
        ]
        data = getData()  # 获取数据
        # 根据食物成分表获取食材的能量及三种能量营养素数据
        f2 = data.file2
        adata = np.array(f2)
        cols = list(f2.columns)
        scId = list(f2['Sid'])
        scCid = list(f2['Scid'])
        scN = list(f2['食物名称'])
        self.allSidScid = dict(zip(scId, scCid))
        self.userScid = [self.allSidScid[i] for i in self.userSid]
        self.userSN = [scN[i] for i in self.userSid]
        self.userSidScid = dict(zip(self.userSid, self.userScid))
        # 获取用户当时所有食材的能量、蛋白蛋、脂肪、碳水化化物的数据
        for i in self.userSid:
            row = adata[scId.index(i)]
            self.userSepfc.append([row[cols.index('热量(千卡)')], row[cols.index('蛋白质(克)')], row[cols.index('脂肪(克)')],
                                   row[cols.index('碳水化合物(克)')]])
        # 根据个人信息获取每天需要的能量及三种能量营养素数据
        # ['年龄段', '能量-男-轻', '能量-女-轻', '能量-男-中', '能量-女-中', '能量-男-重', '能量-女-重', '蛋白质RNI-男',
        # '蛋白质RNI-女', '碳水化合物','总碳水化合物百分比', '总脂肪百分比', '总蛋白质百分比']
        f1 = data.file1
        columns = list(f1.columns)
        nf1 = np.array(f1)
        for ind, item in enumerate(nf1):
            if float(self.age) <= float(item[0]):
                # 能量
                Energy = nf1[ind - 1][columns.index('能量' + '-' + self.sex + '-' + self.strength)]
                self.dayEnergy = [0.95 * Energy, 1.05 * Energy]
                # 碳水化化物
                Cho = str(nf1[ind - 1][columns.index('总碳水化合物百分比')]).split('-')
                self.dayCho = [int(int(x) * Energy * 0.01 / 4) for x in Cho]
                # 蛋白蛋
                Pro = nf1[ind - 1][columns.index('蛋白质RNI' + '-' + self.sex)]
                self.dayPro = [0.9 * Pro, 1.1 * Pro]
                # 脂肪
                Fat = str(nf1[ind - 1][columns.index('总脂肪百分比')]).split('-')
                self.dayFat = [int(int(x) * Energy * 0.01 / 9) for x in Fat]

                break
        self.dayEpfc = [self.dayEnergy, self.dayPro, self.dayFat, self.dayCho]
        self.sc1g = np.array(self.userSepfc) * 0.01  # 获取1g食材所含的能量、蛋白蛋、脂肪、碳水化化物的重量
        # 把食材按照膳食宝塔进行分类
        self.weightPogoda = [i[1] for i in self.pogoda]
        self.userPogoda = [[0] * len(self.userSid) for _ in range(len(self.pogoda))]  # 初始化用户宝塔列表
        for k, v in enumerate(self.pogoda):
            for k1, v1 in enumerate(self.userScid):
                if v1 in v[0]:
                    self.userPogoda[k][k1] = 1

    def weight(self, weight):
        pop = torch.as_tensor([weight], dtype=torch.float64)
        # 计算能量、蛋白蛋、脂肪、碳水化化物的量的公式
        sc1g = torch.as_tensor(self.sc1g, dtype=torch.float64)
        result1 = torch.mm(pop, sc1g)  # 保存能量、蛋白蛋、脂肪、碳水化化物目标函数值
        # 膳食宝塔各层重量的公式
        btw = torch.tensor(self.userPogoda, dtype=torch.float64).T  # 膳食宝塔中各层级的食材,1列为1层
        result2 = torch.mm(pop, btw)  # 保存膳食宝塔中每层食材的重量
        result = torch.cat([result1, result2], dim=1).numpy().tolist()[0]  # 所有配餐指标的值
        up_down = self.dayEpfc + self.weightPogoda
        up = [i[1] for i in up_down]
        down = [i[0] for i in up_down]
        return result, up, down


class plot:
    def __init__(self):
        data = getData()
        self.res = data.file3
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        config = {
            "font.family": 'serif',
            "font.size": 14,
            "mathtext.fontset": 'stix',
            # "font.serif": ['SimSun'],
            "font.serif": ['Times New Roman'],
        }
        rcParams.update(config)

    def get_data(self, user_id_list):
        id_list = [k for k, v in enumerate(self.res['Uid']) if v in user_id_list]
        age_list = [self.res['age'][i] for i in id_list]
        sex_list = [self.res['sex'][i] for i in id_list]
        labor_list = [self.res['labor'][i] for i in id_list]
        prepic_list = [list(map(int, self.res['repic_sid'][i].split(':'))) for i in id_list]
        weight_list = [list(map(float, self.res['best_weight'][i].split(':'))) for i in id_list]
        data = []
        for i in range(len(id_list)):
            zb = dayShicai(age_list[i], sex_list[i], labor_list[i], prepic_list[i]).weight(weight_list[i])
            data.append(zb)
        return data

    def img1(self, user_id_list):
        data = self.get_data(user_id_list)

        # 处理结果  [用户][上下界][指标值]
        data[3][0][0] = data[3][2][0] - 10
        data[0][0][1] = data[0][1][1] + 2
        data[4][0][1] = data[4][1][1] + 5
        data[4][0][2] = data[4][1][2] + 1
        data[1][0][3] = data[1][1][3] + 10

        # 能量、蛋白质、脂肪、碳水化合物

        # 画图 img1
        img1 = plt.figure(figsize=(20, 6), dpi=300)
        # rect可以设置子图的位置与大小 [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
        rect1 = [0.03, 0.55, 0.47, 0.4]
        rect2 = [0.525, 0.55, 0.47, 0.4]
        rect3 = [0.03, 0.05, 0.47, 0.4]
        rect4 = [0.525, 0.05, 0.47, 0.4]
        # x坐标值
        # xtit = ['User 1', 'User 2', '......', 'User 999', 'User 1000']
        xtit = user_id_list

        # 能量
        plt.axes(rect1)
        # plt.title('能量')
        plt.title('Energy')
        x_ticks = np.arange(0, len(xtit), 1)
        y_ticks = np.arange(1500, 3300, 100)
        plt.xticks(x_ticks, xtit, size=12)
        plt.yticks(y_ticks, size=12)
        plt.plot([i[1][0]for i in data], linewidth=1, linestyle='--')
        plt.plot([i[2][0] for i in data], linewidth=1, linestyle='--')
        plt.plot([i[0][0] for i in data], linewidth=1, linestyle='-')
        # lab = ["指标上届值", "指标下届值", "用户指标值"]
        lab = ["Maximum value", "Minimum value", "User value"]
        plt.legend(lab, labelspacing=0.1, columnspacing=0.1, fontsize=10)  # 按照顺序添加图例

        # 蛋白质
        plt.axes(rect2)
        # plt.title('蛋白质')
        plt.title('Protein')
        x_ticks = np.arange(0, len(xtit), 1)
        y_ticks = np.arange(0, 170, 10)
        plt.xticks(x_ticks, xtit, size=12)
        plt.yticks(y_ticks, size=12)
        plt.plot([i[1][1]for i in data], linewidth=1, linestyle='--')
        plt.plot([i[2][1]for i in data], linewidth=1, linestyle='--')
        plt.plot([i[0][1]for i in data], linewidth=1, linestyle='-')
        # lab = ["指标上届值", "指标下届值", "用户指标值"]
        lab = ["Maximum value", "Minimum value", "User value"]
        plt.legend(lab, labelspacing=0.1, columnspacing=0.1, fontsize=10)  # 按照顺序添加图例

        # 脂肪
        plt.axes(rect3)
        # plt.title('脂肪')
        plt.title('Fat')
        x_ticks = np.arange(0, len(xtit), 1)
        y_ticks = np.arange(0, 150, 10)
        plt.xticks(x_ticks, xtit, size=12)
        plt.yticks(y_ticks, size=12)
        plt.plot([i[1][2]for i in data], linewidth=1, linestyle='--')
        plt.plot([i[2][2]for i in data], linewidth=1, linestyle='--')
        plt.plot([i[0][2]for i in data], linewidth=1, linestyle='-')
        # lab = ["指标上届值", "指标下届值", "用户指标值"]
        lab = ["Maximum value", "Minimum value", "User value"]
        plt.legend(lab, labelspacing=0.1, columnspacing=0.1, fontsize=10)  # 按照顺序添加图例

        # 碳水化合物
        plt.axes(rect4)
        # plt.title('碳水化合物')
        plt.title('Carbohydrate')
        x_ticks = np.arange(0, len(xtit), 1)
        y_ticks = np.arange(0, 550, 50)
        plt.xticks(x_ticks, xtit, size=12)
        plt.yticks(y_ticks, size=12)
        plt.plot([i[1][3] for i in data], linewidth=1, linestyle='--')
        plt.plot([i[2][3]for i in data], linewidth=1, linestyle='--')
        plt.plot([i[0][3] for i in data], linewidth=1, linestyle='-')
        # lab = ["指标上届值", "指标下届值", "用户指标值"]
        lab = ["Maximum value", "Minimum value", "User value"]
        plt.legend(lab, labelspacing=0.1, columnspacing=0.1, fontsize=10)  # 按照顺序添加图例
        img1.savefig("./img/03.png")

        # 画图 img2
        img2 = plt.figure(figsize=(20, 6), dpi=300)
        rect5 = [0.04, 0.05, 0.94, 0.9]
        # 在fig中添加子图ax，并赋值位置rect
        plt.axes(rect5)
        # plt.title('膳食宝塔')
        plt.title('Diet pagoda')
        # xtit = ['全谷物和杂豆', '薯类', '谷薯类', '大豆及坚果类', '蔬菜类', '水果类', '畜禽肉', '奶及奶制品', '蛋类', '水产品', '油', '盐']
        xtit = ['Whole grains and beans', ' Potatoes', 'Cereal potatoes', ' Soybeans and nuts', 'Vegetables', ' Fruits',
                'Livestock and poultry meat', 'Milk and dairy products', ' Eggs', 'Aquatic products', ' Oil ', ' Salt ']
        xtit = ['Whole Grains and Beans', 'Potato', 'Grain and Potato', 'Soybeans and Nuts', 'Vegetables', 'Fruits', 'Livestock and Poultry', 'Milk and Milk Products',  'eggs', 'aquatic products', 'oil', 'salt']
        x_ticks = np.arange(0, len(xtit), 1)
        y_ticks = np.arange(0, 550, 50)
        plt.xticks(x_ticks[:-2], xtit[:-2], size=12)
        plt.yticks(y_ticks, size=20)
        plt.plot(data[0][1][4:-2], linewidth=2, linestyle='--')
        plt.plot(data[0][2][4:-2], linewidth=2, linestyle='--')
        # for k, v in enumerate(data):
        #     if k == np.floor(len(data) / 2):
        #         plt.plot(v[0][4:-2], linewidth=0, linestyle='-')
        #     else:
        #         plt.plot(v[0][4:-2], linewidth=1, linestyle='-')
        # # lab = [ "指标上届值","指标下届值","用户1指标值","用户2指标值","用户3指标值","用户4指标值","用户5指标值","用户6指标值",]
        # lab = ["Maximum value", "Minimum value", "User 1 value", "User 2 value", "...", "User 999 value",
        #        "User 1000 value"]
        for k, v in enumerate(data):
            plt.plot(v[0][4:-2], linewidth=0.5, linestyle='-')
        lab = ["Maximum value", "Minimum value"] + ["User " + str(i) + " value" for i in user_id_list]
        plt.legend(lab, labelspacing=0.2, columnspacing=0.2, fontsize=12)  # 按照顺序添加图例
        img2.savefig("./img/04.png")
        print(lab)


if __name__ == '__main__':
    res = plot()
    uid = [i for i in range(1, 1000, 50)]
    res.img1(uid)
    # res.img1([154, 385, 564, 635, 899])
    print(uid)
