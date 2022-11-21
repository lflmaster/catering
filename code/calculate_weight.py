import numpy as np
import torch
import pandas as pd


class getData:
    def __init__(self):
        dir_path1 = './source/'
        # 食物成分数据
        self.file1 = pd.read_csv(dir_path1 + '食物成分数据.csv')
        # 参考摄入量
        self.file2 = pd.read_csv(dir_path1 + '参考摄入量.csv')


# 评估膳食宝塔每层食材的重量是否达标
# 评估一天的食材的能量及三种能量营养素是否达标
class dayShicai(object):
    def __init__(self, age, sex, strength, userSidList):
        # 初始化个人信息
        self.age = age
        self.sex = sex
        self.strength = strength
        self.userSid = userSidList  # 用户配餐食材的id列表
        # 初始化其他数据
        self.allSciD = None  # 保存所有食材的类别号和名称
        self.pogoda = None   # 保存膳食宝塔每层的数据
        self.allSidScid = {}  # 所有食材id和类别id组成的字典
        self.userSN = []  # 用户当日所有的食材名称
        self.userScid = []  # 用户当日所有食材对应的类别id
        self.userSidScid = {}  # 用户当日所有食材id和类别id组成的字典
        self.userSepfc = []  # 用户当日所有食材对应的能量、蛋白蛋、脂肪、碳水化合物的数据
        self.dayEpfc = []  # 该用户一天需要摄入的能量、蛋白蛋、脂肪、碳水化化的量
        self.dayEnergy = []   # 能量 千卡
        self.dayCho = []  # 碳水化化物 克
        self.dayPro = []   # 蛋白蛋 克
        self.dayFat = []  # 脂肪 克
        self.weightPogoda = []  # 膳食宝塔中各层级的重量标准
        self.userPogoda = []  # 用户当日所有食材在膳食宝塔中对应的层级的索引
        self.setData()  # 调用函数
        # 初始化GA算法参数
        self.pop_size = 100     # 一个种群中个体的个数
        self.x_num = len(userSidList)   # 自变量个数
        self.x_min = 10          # 自变量最小值
        self.x_max = 500       # 自变量最大值
        self.N_GENERATIONS = 1000  # 迭代次数

    def setData(self):
        # Sid:食材id  Scid:食材类id  Sepfc:食材的能量、蛋白蛋、脂肪、碳水化化物  userSweight:食材的重量
        self.allSciD = {'1': '谷类及制品', '2': '薯类、淀粉及制品', '3': '干豆类及制品', '4': '蔬菜类及制品', '5': '菌藻类',
                        '6': '水果类及制品', '7': '坚果、种子类', '8': '畜肉类及制品', '9': '禽肉类及制品', '10': '乳类及制品',
                        '11': '蛋类及制品', '12': '鱼虾蟹贝类', '13': '婴幼儿食品', '14': '小吃、甜饼', '15': '速食食品',
                        '16': '软料类', '17': '含酒精饮料', '18': '糖、果脯和蜜饯、蜂蜜', '19': '油脂类', '20': '调味品类',
                        '21': '药食及其它'}
        # 膳食宝塔：一般健康成人每日对各类食物适宜的摄入量范围  第一列是类别，第二列是重量
        self.pogoda = [
            [[1], [50, 150]],                   # 全谷物和杂豆：50-150
            [[2], [50, 100]],                   # 薯类：50-100克
            [[1, 2], [250, 400]],               # 谷薯类：250~400克
            [[3, 7], [25, 35]],                 # 大豆及坚果类：25-35克
            [[4], [300, 500]],                  # 蔬菜类：300-500克
            [[6], [200, 350]],                  # 水果类：200-350克
            [[8, 9], [40, 75]],                 # 畜禽肉：40-75克
            [[10], [300, 300]],                 # 奶及奶制品：300克
            [[11], [40, 50]],                   # 蛋类：40~50克
            [[12], [40, 75]],                   # 水产品：40-75克
            [[19], [25, 30]],                   # 油：25-30克
            [[20], [0, 6]],                     # 盐：<6克
        ]
        datas = getData()  # 获取数据
        # 根据食物成分表获取食材的能量及三种能量营养素数据
        data = datas.file1
        adata = np.array(data)
        cols = list(data.columns)
        scId = list(data['Sid'])
        scCid = list(data['Scid'])
        scN = list(data['食物名称'])
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
        dataset = datas.file2
        columns = list(dataset.columns)
        data = np.array(dataset)
        for ind, item in enumerate(data):
            if float(self.age) <= float(item[0]):
                # 能量
                Energy = data[ind - 1][columns.index('能量' + '-' + self.sex + '-' + self.strength)]
                self.dayEnergy = [0.95*Energy, 1.05*Energy]
                # 碳水化化物
                Cho = str(data[ind - 1][columns.index('总碳水化合物百分比')]).split('-')
                self.dayCho = [int(int(x) * Energy * 0.01 / 4) for x in Cho]
                # 蛋白蛋
                Pro = data[ind - 1][columns.index('蛋白质RNI' + '-' + self.sex)] / 2
                # print(Pro)
                # exit()
                self.dayPro = [0.95*Pro, 1.05*Pro]
                # 脂肪
                Fat = str(data[ind - 1][columns.index('总脂肪百分比')]).split('-')
                self.dayFat = [int(int(x) * Energy * 0.01 / 9) for x in Fat]
                break
        self.dayEpfc = [self.dayEnergy, self.dayPro, self.dayFat, self.dayCho]
        # print(self.dayEpfc)
        # 把食材按照膳食宝塔进行分类
        self.weightPogoda = [i[1] for i in self.pogoda]
        self.userPogoda = [[0] * len(self.userSid) for _ in range(len(self.pogoda))]  # 初始化用户宝塔列表
        for k, v in enumerate(self.pogoda):
            for k1, v1 in enumerate(self.userScid):
                if v1 in v[0]:
                    self.userPogoda[k][k1] = 1

    def numb2(self, st):
        aa = str(st).split('.')[0] + '.' + str(st).split('.')[1][:2]
        return aa

    def train(self):
        g = GA(self.pop_size, self.x_num, self.x_min, self.x_max)  # 实例化遗传算法
        # 初始化当前用户的能量、蛋白蛋、脂肪、碳水化化物和膳食宝塔信息
        g.userSN = self.userSN  # 用户当日所有的食材名称
        g.sc1g = np.array(self.userSepfc) * 0.01  # 获取1g食材所含的能量、蛋白蛋、脂肪、碳水化化物的重量
        # print(self.userSepfc)
        g.dayEpfc = self.dayEpfc  # 获取指定用户每天需要的能量、蛋白蛋、脂肪、碳水化化物的量
        g.weightPogoda = self.weightPogoda  # 膳食宝塔中各层级的重量标准
        g.userPogoda = self.userPogoda  # 膳食宝塔中各层级的食材信息
        pop = g.init()  # 初始化种群
        old_pop = np.random.randint(self.x_min, self.x_max, (self.pop_size, self.x_num))
        # print('初始化：', pop)
        best_pop = np.zeros((10, g.x_num))  # 保存全局最优的前10个解
        last_pop = np.zeros((10, g.x_num))  # 保存最后一轮最优的前10个解
        for i in range(self.N_GENERATIONS):
            pop = np.where(pop < self.x_min, self.x_min, pop)  # 处理重量小于0的值
            pop = np.where(pop > self.x_max, self.x_max, pop)  # 处理重量小于0的值
            pop = torch.as_tensor(pop, dtype=torch.float64)  # np转换成as_tensor
            result, popfun = g.cal(pop)
            # 根据梯度更新种群中所有个体的自变量
            new_pop = g.gradient_pop(result, pop)
            new_pop = torch.as_tensor(new_pop)
            me1 = torch.mean(popfun, dim=1)
            sor1 = torch.sort(me1)  # 返回排序索引
            top10v1 = sor1[0][:10]
            top10k1 = sor1[1][:10]
            last_pop = new_pop[top10k1].detach().numpy()  # 本次迭代最优前10
            con_pop = np.concatenate((best_pop, last_pop), axis=0)
            best_pop = g.sort_pop(con_pop)[:10, :]
        return last_pop[0], best_pop[0]


class GA(object):
    def __init__(self, pop_size, x_num, x_min, x_max):
        super(GA, self).__init__()
        # 初始化遗传算法参数
        self.pop_size = pop_size     # 一个种群中个体的个数
        self.x_num = x_num   # 自变量个数
        self.x_min = x_min          # 自变量最小值
        self.x_max = x_max       # 自变量最大值
        # 传入当前用户的能量、蛋白蛋、脂肪、碳水化化物和膳食宝塔信息
        self.userSN = None  # 用户当日所有的食材名称
        self.sc1g = None  # 获取1g食材所含的能量、蛋白蛋、脂肪、碳水化化物的重量
        self.dayEpfc = None  # 获取指定用户每天需要的能量、蛋白蛋、脂肪、碳水化化物的量
        self.weightPogoda = None  # 膳食宝塔中各层级的重量标准
        self.userPogoda = None  # 膳食宝塔中各层级的食材信息

    # 初始化种群
    def init(self):
        pop = np.random.randint(self.x_min, self.x_max, (self.pop_size, self.x_num))  # 随机生成种群 pop为食材重量
        # print(pop)
        return pop

    # 计算群体中个体的适应度函数
    def cal(self, pop):
        pop = torch.as_tensor(pop)
        # 计算能量、蛋白蛋、脂肪、碳水化化物的量的公式
        sc1g = torch.as_tensor(self.sc1g, dtype=torch.float64)
        result1 = torch.mm(pop, sc1g)  # 保存能量、蛋白蛋、脂肪、碳水化化物目标函数值
        # 膳食宝塔各层重量的公式
        btw = torch.tensor(self.userPogoda, dtype=torch.float64).T  # 膳食宝塔中各层级的食材,1列为1层
        result2 = torch.mm(pop, btw)  # 保存膳食宝塔中每层食材的重量
        result = torch.cat([result1, result2], dim=1)  # 所有配餐指标的值
        # 均值
        jz1 = torch.mean(torch.as_tensor(self.dayEpfc), dim=1)
        jz2 = torch.mean(torch.as_tensor(self.weightPogoda, dtype=torch.float64), dim=1)
        jz = torch.cat([jz1, jz2])
        popfun = (result - jz).pow(2).sqrt()  # 所有适应度函数值 函数值越小效果越好
        return result, popfun

    # 群中所有个体的梯度值
    def gradient_pop(self, result, sc_weight):
        zb_user = result  # 获取种群中所有个体的真实指标值  与指标维度对应
        new_weight = sc_weight.detach().numpy()
        sc_pop = torch.zeros((self.pop_size, self.x_num))   # 存取种群中所有个体需要更新的梯度 与食材维度对应
        # 所有指标标准   对食材重量x进行上下界约束
        bz = self.dayEpfc + self.weightPogoda  # 标准指标值
        bz_up = torch.as_tensor([i[1] for i in bz])
        bz_down = torch.as_tensor([i[0] for i in bz])
        a = torch.zeros_like(zb_user) - 1
        b = torch.zeros_like(zb_user) + 1
        c = torch.zeros_like(zb_user)
        # torch.where()函数的作用是按照一定的规则合并两个as_tensor类型。
        # torch.where(condition，a，b)其中
        # 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出
        result1 = torch.where(zb_user - bz_up > 0, a, c)   # 如果重量超过上届，则进行减去一个单位
        result2 = torch.where(bz_down - zb_user > 0, b, c)  # 如果重量低于下届，则进行加上一个单位
        zb_table = result1 + result2  # -1表示过多，需要减小;+1表示不足，需要增大.
        # 根据营养素调整重量
        aa = zb_table.split(1, 1)[:4]  # 能量、蛋白蛋、脂肪、碳水化化物指标情况
        gd_pop = sc_pop + aa[0] + aa[1] + aa[2] + aa[3]
        sc1g = torch.as_tensor(self.sc1g.T)
        for k, v in enumerate(sc_weight):
            wei = torch.mul(v, sc1g)
            # max_ind = torch.max(wei, dim=1)  # 计算能量、蛋白蛋、脂肪、碳水化化物含量最多的食材
            sor_ind = torch.sort(wei, descending=True, dim=1)  # 计算能量、蛋白蛋、脂肪、碳水化化物含量最多的食材
            select1 = [i[np.random.randint(0, 3)] for i in sor_ind[1][:, :3]]  # 从前三个含量多的食材挑选一种进行操作
            for k1, v1 in enumerate(select1):
                new_weight[k][v1] += zb_table[k][k1]
        # 根据膳食宝塔调整重量
        bt = torch.as_tensor(self.userPogoda, dtype=torch.float64).T  # 获取用户膳食宝塔中各层级的食材信息
        # print(bt)
        for k, v in enumerate(zb_table[:, 4:]):
            val = torch.sum(bt * v, 1)
            # print(v)
            # print(val)
            # exit()
            new_weight[k] += val.detach().numpy()
        return new_weight

    # 排序函数: 根据适应度函数值对pop种群中的个体进行排序，适应度值越低越靠前
    def sort_pop(self, pop):
        result, popfun = self.cal(pop)  # 计算群体中个体的适应度函数
        me = torch.mean(popfun, dim=1)  # 行
        sor = torch.sort(me)  # 返回排序索引
        s_pop = pop[sor[1]]
        return s_pop
