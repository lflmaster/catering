import sys
sys.path.append("./code")
import pandas as pd
from choose_shicai import *  # 算法类
from calculate_weight import *
import csv

class getData:
    def __init__(self):

        # 用户数据 Uid,age,sex,labor,symptoms,favourite,last_recipe
        dir_path1 = './source/'
        self.file1 = pd.read_csv(dir_path1 + '用户信息.csv')
        # self.Uid = list(file1['Uid'])
        # self.age = list(file1['age'])
        # self.sex = list(file1['sex'])
        # self.labor = list(file1['labor'])
        # self.symptoms = list(file1['symptoms'])
        # self.favourite = list(file1['favourite'])
        # self.last_recipe = list(file1['last_recipe'])


class recipe:
    def __init__(self):
        self.repic_sid = []  # 配餐食材id列表
        self.repic_name = []  # 配餐食材name列表
        self.best_weight = []  # 最优的配餐食材重量
        self.last_weight = []  # 最后一轮配餐食材重量
        self.all_user()

    def once_recipe(self, age, sex, labor, symptoms, favourite, last_recipe):
        # 选择食材
        user = simple_choose(symptoms, favourite, last_recipe)
        u_sck = user.choose()  # 用户食材库
        # print('粗选食材库：', u_sck, len(u_sck))
        sc1 = well_choose(symptoms, favourite, last_recipe, u_sck)
        self.repic_sid, self.repic_name = sc1.train()
        # print(self.repic_sid, len(self.repic_sid))
        # 计算重量
        js = dayShicai(age, sex, labor, self.repic_sid)
        self.last_weight, self.best_weight = js.train()

    def all_user(self):
        data = getData().file1
        with open('./data/配餐结果.csv', 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['ID', 'Uid', 'age', 'sex', 'labor', 'repic_sid', 'best_weight', 'last_weight'])
            for i in range(len(data)):
                uid = data.iloc[i, :].Uid
                age = data.iloc[i, :].age
                sex = data.iloc[i, :].sex
                labor = data.iloc[i, :].labor
                symptoms = list(map(int, data.iloc[i, :].symptoms.split(':')))
                favourite = list(map(int, data.iloc[i, :].favourite.split(':')))
                last_recipe = list(map(int, data.iloc[i, :].last_recipe.split(':')))
                # print(symptoms, favourite, last_recipe)
                self.once_recipe(age, sex, labor, symptoms, favourite, last_recipe)

                repic_sid = ':'.join(map(str, self.repic_sid))  # 配餐食材id列表
                repic_name = ':'.join(map(str, self.repic_name))  # 配餐食材name列表
                best_weight = ':'.join(map(str, self.best_weight))  # 最优的配餐食材重量
                last_weight = ':'.join(map(str, self.last_weight))  # 最后一轮配餐食材重量
                writer.writerow([i, uid, age, sex, labor, repic_sid, best_weight, last_weight])
                print(i, '用户', uid, '配餐结果：', repic_sid)


if __name__ == '__main__':

    recipe()

