import pandas as pd
import numpy as np


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


class resoult:
    def __init__(self):
        data = getData()
        self.file2 = data.file2
        self.file3 = data.file3
        self.file4 = data.file4
        self.file5 = data.file5

    def one_info(self, user_id_list):
        all_Uid = self.file3['Uid']
        info_id = [k for k, v in enumerate(all_Uid) if v in user_id_list]
        user_info = self.user_info(user_id_list)
        for k, i in enumerate(info_id):
            print('*'*10, '用户' + str(i) + '配餐结果：', '*'*10)
            print(user_info[k])
            repic_sid = list(map(int, self.file3.iloc[i, :].repic_sid.split(':')))
            wight = list(map(float, self.file3.iloc[i, :].best_weight.split(':')))
            bm = self.sc_name(repic_sid)
            for j in range(len(repic_sid)):
                one_res = str(repic_sid[j]) + ':' + str(bm[j]) + ":" + str(wight[j]).split('.')[0] + "g"
                print(one_res)

    def sc_name(self, sid_list):
        all_Uid = self.file2['Sid']
        all_bm = self.file2['别名']
        info_id = []
        for i in sid_list:
            index = [k for k, v in enumerate(all_Uid) if v == i][0]
            info_id.append(index)
        bm_list = [all_bm[i].split('、')[0] for i in info_id]
        return bm_list

    def zz_name(self, zid_list):
        all_zid = self.file5['Zid']
        all_name = self.file5['症状名称']
        info_id = [k for k, v in enumerate(all_zid) if v in zid_list]
        info = [all_name[i] for i in info_id]
        return info

    def user_info(self, user_id):
        all_Uid = self.file4['Uid']
        info_id = [k for k, v in enumerate(all_Uid) if v in user_id]
        info = []
        for i in info_id:
            age = self.file4.iloc[i, :].age
            sex = self.file4.iloc[i, :].sex
            labor = self.file4.iloc[i, :].labor
            symptoms = list(map(int, self.file4.iloc[i, :].symptoms.split(':')))
            favourite = list(map(int, self.file4.iloc[i, :].favourite.split(':')))
            last_recipe = list(map(int, self.file4.iloc[i, :].last_recipe.split(':')))
            info.append([age, sex, labor, self.zz_name(symptoms),self.sc_name(favourite), self.sc_name(last_recipe)])
        return info


if __name__ == '__main__':
    res = resoult()
    res.one_info([286, 298, 564, 489, 618])
    # res.one_info([286, 298, 489, 618])
    # res.one_info([154, 385, 635, 899])
