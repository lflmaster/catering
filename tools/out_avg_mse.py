# 画二维坐标图
# 读取csv并作图
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import torch
import csv


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
    def __init__(self, age, sex, strength):
        # 初始化个人信息
        self.age = age
        self.sex = sex
        self.strength = strength
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
        self.data = getData()  # 根据食物成分表获取食材的能量及三种能量营养素数据
        # 根据个人信息获取每天需要的能量及三种能量营养素数据
        # ['年龄段', '能量-男-轻', '能量-女-轻', '能量-男-中', '能量-女-中', '能量-男-重', '能量-女-重', '蛋白质RNI-男',
        # '蛋白质RNI-女', '碳水化合物','总碳水化合物百分比', '总脂肪百分比', '总蛋白质百分比']
        dataset = self.data.file1
        columns = list(dataset.columns)
        data1 = np.array(dataset)
        for ind, item in enumerate(data1):
            if float(self.age) <= float(item[0]):
                # 能量
                Energy = data1[ind - 1][columns.index('能量' + '-' + self.sex + '-' + self.strength)]
                self.dayEnergy = [0.95 * Energy, 1.05 * Energy]
                # 碳水化化物
                Cho = str(data1[ind - 1][columns.index('总碳水化合物百分比')]).split('-')
                self.dayCho = [int(int(x) * Energy * 0.01 / 4) for x in Cho]
                # 蛋白蛋
                Pro = data1[ind - 1][columns.index('蛋白质RNI' + '-' + self.sex)]
                self.dayPro = [0.95 * Pro, 1.05 * Pro]
                # 脂肪
                Fat = str(data1[ind - 1][columns.index('总脂肪百分比')]).split('-')
                self.dayFat = [int(int(x) * Energy * 0.01 / 9) for x in Fat]

                break
        # 该用户一天需要摄入的能量、蛋白蛋、脂肪、碳水化化的量
        self.dayEpfc = [self.dayEnergy, self.dayPro, self.dayFat, self.dayCho]
        self.weightPogoda = [i[1] for i in self.pogoda]  # 膳食宝塔中各层级的重量标准
        self.userSepfc = []  # 用户当日所有食材对应的能量、蛋白蛋、脂肪、碳水化合物的数据
        self.userPogoda = []  # 用户当日所有食材在膳食宝塔中对应的层级的索引

    def call(self, repic_sid, repic_weight):
        data2 = self.data.file2
        cols = list(data2.columns)
        adata = np.array(data2)
        scId = list(data2['Sid'])
        scCid = list(data2['Scid'])
        allSidScid = dict(zip(scId, scCid))
        userScid = [allSidScid[i] for i in repic_sid]  # 用户当日所有食材对应的类别id
        # 获取用户当日所需食材的能量、蛋白蛋、脂肪、碳水化化物的数据
        userSepfc = []  # 用户当日所有食材对应的能量、蛋白蛋、脂肪、碳水化合物的数据
        for i in repic_sid:
            row = adata[scId.index(i)]
            userSepfc.append([row[cols.index('热量(千卡)')], row[cols.index('蛋白质(克)')], row[cols.index('脂肪(克)')],
                              row[cols.index('碳水化合物(克)')]])
        sc1g = np.array(userSepfc) * 0.01  # 获取1g食材所含的能量、蛋白蛋、脂肪、碳水化化物的重量
        # 把食材按照膳食宝塔进行分类
        self.userPogoda = [[0] * len(repic_sid) for _ in range(len(self.pogoda))]  # 初始化用户宝塔列表
        for k, v in enumerate(self.pogoda):
            for k1, v1 in enumerate(userScid):
                if v1 in v[0]:
                    self.userPogoda[k][k1] = 1
        pop = torch.as_tensor([repic_weight], dtype=torch.float64)
        # 计算能量、蛋白蛋、脂肪、碳水化化物的量的公式
        sc1g = torch.as_tensor(sc1g, dtype=torch.float64)
        result1 = torch.mm(pop, sc1g)  # 保存能量、蛋白蛋、脂肪、碳水化化物目标函数值
        # 膳食宝塔各层重量的公式
        btw = torch.as_tensor(self.userPogoda, dtype=torch.float64).T  # 膳食宝塔中各层级的食材,1列为1层
        result2 = torch.mm(pop, btw)  # 保存膳食宝塔中每层食材的重量
        result = torch.cat([result1, result2], dim=1)  # 所有配餐指标的值
        return result.numpy()[0], self.dayEpfc + self.weightPogoda


# 配餐结果分析
class resultAnalysis:
    def __init__(self):
        self.file3 = getData().file3
        self.file4 = getData().file4
        self.file5 = getData().file5

    def all_info(self):
        sig_table = np.array([[0] * 16 for i in range(len(self.file3))])
        for i in range(len(self.file3)):
            one_data = self.file3.iloc[i, :]
            user = dayShicai(one_data.age, one_data.sex, one_data.labor)
            sid = list(map(int, self.file3.iloc[i, :].repic_sid.split(':')))
            wight = list(map(float, self.file3.iloc[i, :].best_weight.split(':')))
            res = user.call(sid, wight)
            # print(i, res[0])
            for j in range(len(res[0])):
                if 0.95 * res[1][j][0] <= res[0][j] <= 1.05 * res[1][j][1]:
                    sig_table[i][j] = 1
            # break

        # 平均
        a1 = np.sum(sig_table[:, :-2]) / (len(self.file3) * 14)  # 综合指标达标率
        a2 = np.sum(sig_table[:, :4]) / (len(self.file3) * 4)  # 营养素达标率
        a3 = np.sum(sig_table[:, 4:-2]) / (len(self.file3) * 10)  # 膳食宝塔达标率
        # 均方差
        b1 = np.sum(sig_table[:, :-2], axis=1) / 14  # 综合指标达标率
        b2 = np.sum(sig_table[:, :4], axis=1) / 4  # 营养素达标率
        b3 = np.sum(sig_table[:, 4:-2], axis=1) / 10  # 膳食宝塔达标率
        # c1 = np.sqrt(np.sum((b1 - a1) ** 2) / len(b1))
        c1 = np.sum((b1 - a1) ** 2) / len(b1)
        # c2 = np.sqrt(np.sum((b2 - a2) ** 2) / len(b2))
        c2 = np.sum((b2 - a2) ** 2) / len(b2)
        # c3 = np.sqrt(np.sum((b3 - a3) ** 2) / len(b3))
        c3 = np.sum((b3 - a3) ** 2) / len(b3)

        print('>=13 指标用户id：', [k+1 for k, v in enumerate(np.sum(sig_table[:, :-2], axis=1)) if v >= 13])
        print('14 指标用户id：', [k+1 for k, v in enumerate(np.sum(sig_table[:, :-2], axis=1)) if v == 14])
        # print('综合指标达标率：', np.sum(sig_table[:, :-2], axis=0) / len(self.file3))
        print('平均')
        print('综合指标达标率：', a1)
        print('营养素达标率：', a2)
        print('膳食宝塔达标率：', a3)
        print('均方差')
        print('综合指标达标率:', c1)
        print('营养素达标率:', c2)
        print('膳食宝塔达标率:', c3)

    def percentage(self):
        u_repic = self.file3['repic_sid']
        u_like = self.file4['favourite']
        u_zz = self.file4['symptoms']
        u_last = self.file4['last_recipe']
        yc = self.file5['YCZid']
        lp1 = []
        for k, v in enumerate(u_repic):
            pre = 0
            a = u_like[k].split(':')
            b = v.split(':')
            for i in a:
                if i in b:
                    pre += 1
            lp1.append(pre / len(b))
        # lp1_mean = np.mean(lp1)
        lp2 = []
        for k, v in enumerate(u_repic):
            pre = 0
            a = u_zz[k].split(':')
            b = v.split(':')
            sc = []
            for i in a:
                it = yc[int(i)].split(':')
                sc += it
            sc = np.unique(sc)
            for j in sc:
                if j in b:
                    pre += 1
            lp2.append(pre / len(b))
        # lp2_mean = np.mean(lp2)
        lp3 = []
        for k, v in enumerate(u_repic):
            pre = 0
            a = u_last[k].split(':')
            b = v.split(':')
            for i in a:
                if i in b:
                    pre += 1
            lp3.append(pre / len(b))
        # lp3_mean = np.mean(lp3)

        # 平均
        lp1_mean = np.mean(lp1)  # 喜好食材百分比
        lp2_mean = np.mean(lp2)  # 症状食材百分比
        lp3_mean = np.mean(lp3)  # 上次食材百分比
        # 均方差
        # c1 = np.sqrt(np.sum((lp1 - lp1_mean) ** 2) / len(lp1))
        c1 = np.sum((lp1 - lp1_mean) ** 2) / len(lp1)
        # c2 = np.sqrt(np.sum((lp2 - lp2_mean) ** 2) / len(lp2))
        c2 = np.sum((lp2 - lp2_mean) ** 2) / len(lp2)
        # c3 = np.sqrt(np.sum((lp3 - lp3_mean) ** 2) / len(lp3))
        c3 = np.sum((lp3 - lp3_mean) ** 2) / len(lp3)
        print('平均')
        print('喜好食材百分比：', lp1_mean)
        print('症状食材百分比：', lp2_mean)
        print('上次食材百分比：', lp3_mean)
        print('均方差')
        print('喜好食材百分比:', c1)
        print('症状食材百分比:', c2)
        print('上次食材百分比:', c3)

    def one_user(self, user_id):
        all_Uid = self.file3['Uid']
        one_Uid = [k for k, v in enumerate(all_Uid) if v is user_id]
        one_res = []
        one_std = []
        if len(one_Uid) > 0:
            for i in one_Uid:
                one_data = self.file3.iloc[i, :]
                user = dayShicai(one_data.age, one_data.sex, one_data.labor)
                sid = list(map(int, self.file3.iloc[i, :].repic_sid.split(':')))
                wight = list(map(float, self.file3.iloc[i, :].best_weight.split(':')))
                result = user.call(sid, wight)
                one_res.append(result[0])
                one_std.append(result[1])
            r = np.mean(np.array(one_res), axis=0)
            s = np.mean(np.array(one_std), axis=0)
            res_tf = []
            for j in range(len(r)):
                if 0.95 * s[j][0] <= r[j] <= 1.05 * s[j][1]:
                    # if s[j][0] <= r[j]:
                    res_tf.append(1)
                else:
                    res_tf.append(0)
            print(user_id, r, s, res_tf)
            return res_tf
        else:
            print('用户不存在！')
            return 0

    def all_user(self):
        all_Uid = list(map(int, np.unique(self.file3['Uid'])))
        # print(all_Uid)
        user_table = []
        for i in all_Uid:
            rest = self.one_user(i)
            user_table.append(rest)
        user_table = np.array(user_table)
        print(user_table)
        aa = np.sum(user_table, axis=0)
        print(aa)


if __name__ == '__main__':
    res = resultAnalysis()
    # res.one_user(1)
    res.all_info()
    # res.all_user()
    res.percentage()
