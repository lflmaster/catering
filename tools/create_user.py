import numpy as np
import pandas as pd
import csv
import random


# 批量生成用户信息，用于实验测试
# 需要生成100个用户的基本信息，每个用户10条上一次配餐记录


class UserInfo:
    def __init__(self):
        self.id = None  # 设置用户id
        self.age = None  # 设置用户年龄
        self.sex = None  # 设置用户性别
        self.labor = None  # 设置用户劳动强度
        self.symptoms = []  # 设置用户健康症状
        self.favourite = []  # 设置用户食材喜好
        dir_path1 = '../source/'
        file1 = pd.read_csv(dir_path1 + '食物成分数据.csv')
        self.sid = list(file1['Sid'])
        self.scid = list(file1['Scid'])
        self.sid_index = {v1: k1 for k1, v1 in enumerate(self.sid)}  # 食材id值 -> 下标索引值
        self.set_user(1000)  # 参数为用户个数
        self.output_csv(1)  # 参数为每个用户的配餐个数

    def set_user(self, user_num):
        random.seed(50)  # 随机种子
        self.id = [i+1 for i in range(user_num)]
        self.age = [random.randint(18, 45) for i in range(user_num)]
        self.sex = [random.choice(['男', '女']) for i in range(user_num)]
        self.labor = [random.choice(['轻', '中', '重']) for i in range(user_num)]
        # 男症状:symptoms_0 女症状:symptoms_1
        symptoms_0 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 79, 83, 99, 100, 110, 111, 112, 113, 114, 115, 116, 119]
        symptoms_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 79, 81, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
        for i in range(user_num):
            if self.sex[i] == '男':
                self.symptoms.append(random.sample(symptoms_0, random.randint(2, 3)))
            if self.sex[i] == '女':
                self.symptoms.append(random.sample(symptoms_1, random.randint(2, 3)))
        scid_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for i in range(user_num):
            a = []
            for j in random.sample(scid_range, random.randint(5, 8)):
                b = random.choice([i+1 for i, x in enumerate(self.scid) if x is j])
                a.append(b)
            self.favourite.append(a)

    # 膳食宝塔  参数：食材库, 分数表, 范围
    def once_recipe(self, sc_list, sc_obs_list, num_list):
        # 膳食宝塔食材类别，无油和盐
        pogoda_scid = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
        all_scid = self.scid  # 所有食材对应的食材类id
        # 根据用户传入的食材id获取对应的类id
        u_scid = [all_scid[self.sid_index[int(i)]] for i in sc_list]
        scid = np.unique(u_scid)  # 所有无重复的食材类
        scid_sc_obs = {}  # 存储每一类食材的分数
        for i in scid:
            dic = {}
            for k1, v1 in enumerate(u_scid):
                if i == v1:
                    sid = sc_list[k1]
                    sc = sc_obs_list[k1]
                    dic[sid] = sc
            sor_dic = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
            scid_sc_obs[i] = sor_dic
        # print('scid_sc_obs:',scid_sc_obs)
        sc = []  # 分数较高食材的id
        for k1, v1 in scid_sc_obs.items():
            if k1 in pogoda_scid:
                if len(v1) >= 1:
                    num = np.random.randint(int(num_list[0]), int(num_list[1]))
                    for k2, v2 in enumerate(v1.items()):
                        if v2[1] >= 0:
                            sc.append(v2[0])
                            if k2 == num - 1:
                                break
        return sc

    def output_csv(self, pc_num):
        with open('./用户信息.csv', 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['ID', 'Uid', 'age', 'sex', 'labor', 'symptoms', 'favourite', 'last_recipe'])
            ID = 0
            for i in range(len(self.id)):
                for j in range(pc_num):
                    uid = self.id[i]  # 设置用户id
                    age = self.age[i]  # 设置用户年龄
                    sex = self.sex[i]  # 设置用户性别
                    labor = self.labor[i]  # 设置用户劳动强度
                    symptoms = ':'.join(map(str, self.symptoms[i]))  # 设置用户健康症状
                    favourite = ':'.join(map(str, self.favourite[i]))  # 设置用户食材喜好
                    res = ':'.join(map(str, self.once_recipe(self.sid, [random.randint(5, 20) for i in self.sid], [1, 3])))
                    print(ID, uid, age, sex, labor, symptoms, favourite, res)
                    writer.writerow([ID, uid, age, sex, labor, symptoms, favourite, res])
                    ID += 1


if __name__ == '__main__':

    UserInfo()
