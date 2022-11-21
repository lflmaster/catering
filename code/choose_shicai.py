import os
import numpy as np
import pandas as pd
import csv


class getData:
    def __init__(self):
        # 食材相关数据
        dir_path1 = './source/'
        file1 = pd.read_csv(dir_path1 + '食物成分数据.csv')
        self.d1_Sid = list(file1['Sid'])
        self.d1_Scid = list(file1['Scid'])
        self.d1_Sname = list(file1['食物名称'])
        self.d1_Sbm = list(file1['别名'])

        file2 = pd.read_csv(dir_path1 + '相克与宜搭.csv')
        self.d2_Sid = list(file2['Sid'])
        self.d2_XKSid = list(file2['XKSid'])
        self.d2_YDSid = list(file2['YDSid'])

        file3 = pd.read_csv(dir_path1 + '健康养生.csv')
        self.d3_Zid = list(file3['Zid'])
        self.d3_ZSid = list(file3['ZSid'])
        self.d3_YCZid = list(file3['YCZid'])
        self.d3_JCZid = list(file3['JCZid'])

    def sc_id_name(self, lis):
        nam = []
        for k, v in enumerate(lis):
            n = self.d1_Sbm[self.d1_Sid.index(v)].split('、')[0]
            # n = self.d1_Sname[self.d1_Sid.index(v)]
            nam.append(n)
        return nam


# 粗选食材
class simple_choose:
    def __init__(self, symptoms, favourite, last_recipe):
        self.data = getData()  # 获取数据
        # 食材信息
        self.sc_num = len(self.data.d1_Sid)  # 食材总数量数量
        self.sid_index = {v1: k1 for k1, v1 in enumerate(self.data.d1_Sid)}  # 食材id值 -> 下标索引值
        self.index_sid = {k2: v2 for k2, v2 in enumerate(self.data.d1_Sid)}  # 下标索引值 -> 食材id值
        # 保存食材分数
        self.sc_obs = [0. for _ in range(self.sc_num)]  # 初始化食材打分表：根据分数推荐食材
        # 用户信息
        self.symptoms = symptoms  # 设置用户健康症状  症状id值
        self.favourite = favourite  # 设置用户食材喜好  食材id值
        self.last_recipe = last_recipe  # 设置用户上一次配餐  食材id值

    # 粗选食材
    def choose(self):
        # 根据食材相克与宜搭和用户信息更新食材打分列表
        self.update_sc_obs()
        # 根据膳食宝塔和食材分数推荐一种食材组合，一种食材组合即为一种状态
        r_recip = self.sc_pogoda(self.data.d1_Sid, self.sc_obs, [3, 4])
        # 初步形成无重复的配餐列表
        sc_list = np.unique(self.last_recipe + r_recip + self.favourite)
        return sc_list

    # 更新粗选分数表
    def update_sc_obs(self):
        # 根据食材相克与宜搭给食材打分
        for i in self.last_recipe:
            if i in self.data.d2_Sid:
                i_yd = self.data.d2_YDSid[self.data.d2_Sid.index(i)]
                if not pd.isnull(i_yd):
                    i_yd = i_yd.split(':')
                    # print('i_yd',i_yd)
                    for j1 in i_yd:
                        j1_ = self.sid_index[int(j1)]
                        self.sc_obs[j1_] = self.sc_obs[j1_] + 20  # 宜搭食材+20分
                i_xk = self.data.d2_XKSid[self.data.d2_Sid.index(i)]
                if not pd.isnull(i_xk):
                    i_xk = i_xk.split(':')
                    # print('i_xk',i_xk)
                    for j2 in i_xk:
                        j2_ = self.sid_index[int(j2)]
                        self.sc_obs[j2_] = self.sc_obs[j2_] - 10000  # 相克食材-10000分
        # 根据用户健康状况给食材打分
        for i in self.symptoms:
            if i in self.data.d3_ZSid:
                i_YCZid = self.data.d3_YCZid[self.data.d3_Zid.index(i)]
                if not pd.isnull(i_YCZid):
                    i_YCZid = i_YCZid.split(':')
                    # print(i_YCZid)
                    for j1 in i_YCZid:
                        j1_ = self.sid_index[int(j1)]
                        self.sc_obs[j1_] = self.sc_obs[j1_] + 20  # 宜吃食材+20分
                i_JCZid = self.data.d3_JCZid[self.data.d3_Zid.index(i)]
                if not pd.isnull(i_JCZid):
                    i_JCZid = i_JCZid.split(':')
                    # print(i_JCZid)
                    for j2 in i_JCZid:
                        j2_ = self.sid_index[int(j2)]
                        self.sc_obs[j2_] = self.sc_obs[j2_] - 10000  # 忌吃食材-10000分
        # 根据用户爱好给食材打分
        for i in self.favourite:
            if i in self.data.d1_Sid:
                i_ = self.sid_index[int(i)]
                self.sc_obs[i_] = self.sc_obs[i_] + 25  # 用户喜好的食材+25分
        # 根据用户上一次配餐给食材打分
        for i in self.last_recipe:
            if i in self.data.d1_Sid:
                i_ = self.sid_index[int(i)]
                self.sc_obs[i_] = self.sc_obs[i_] + 20  # 用户上一次配餐的食材+20分

    # 膳食宝塔  参数：食材库, 分数表, 范围
    def sc_pogoda(self, sc_list, sc_obs_list, num_list):
        # 膳食宝塔食材类别，无油和盐
        pogoda_scid = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
        all_scid = self.data.d1_Scid  # 所有食材对应的食材类id
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


# 精选食材
class well_choose:
    def __init__(self, symptoms, favourite, last_recipe, sc_list):
        self.data = getData()  # 获取数据
        # 食材信息
        self.sc_list = sc_list  # 获取用户食材库
        # self.sc_num = len(self.data.d1_Sid)  # 食材总数量数量
        self.sid_index = {v1: k1 for k1, v1 in enumerate(self.data.d1_Sid)}  # 食材id值 -> 下标索引值
        # self.index_sid = {k2: v2 for k2, v2 in enumerate(self.data.d1_Sid)}  # 下标索引值 -> 食材id值

        # 用户信息
        self.symptoms = symptoms  # 设置用户健康症状  症状id值
        self.favourite = favourite  # 设置用户食材喜好  食材id值
        self.last_recipe = last_recipe  # 设置用户上一次配餐  食材id值
        # 膳食宝塔
        self.pogoda_scid = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
        # 配餐中每个类别食材个数
        self.recip_c_num = [np.random.randint(1, 3) for i in range(len(self.pogoda_scid))]
        self.u_c_num = [0 for i in range(len(self.pogoda_scid))]  # 用户配餐各类食材数量
        # self.pogoda_scid = [1,  4]

        self.epochs_data = None

        # 强化学习  参数
        self.lr = 0.01
        self.gamma = 0.9
        self.e_greed = 0.1
        self.epochs = 1
        self.max_times = 20  # 替换食材最大次数
        # 需要初始化
        self.state = []  # 存储配餐食材id
        self.action_dim = None
        self.state_dim = None
        # self.recip_num = []  # 配餐中每个类别食材个数
        self.action_space = None
        self.state_space = None
        self.model = None  # 算法模型
        self.th_times = None  # 记录替换食材次数

    def reset(self):
        # 记录替换食材次数
        self.th_times = 0
        # 初始化动作和状态维度数
        self.action_dim = len(self.sc_list)
        self.state_dim = np.sum(self.recip_c_num) + 1
        # 初始化动作和状态空间
        self.action_space = self.sc_list
        self.state_space = [i for i in range(self.state_dim)]
        # 初始化算法模型
        self.model = Sarsa(self.state_dim, self.action_dim, self.lr, self.gamma, self.e_greed)
        return len(self.state)

    def train(self):
        self.epochs_data = []
        # if not os.path.exists('./data'):  # 是否存在这个文件夹
        #     os.makedirs('./data')  # 如果没有这个文件夹，那就创建一个
        # with open('./data/精选数据.csv', 'w', encoding='utf-8') as f:
        #     csv_writer = csv.writer(f)
        # csv_writer.writerow(['epoch', 'total_steps', 'total_reward', 'a1'])
        for epoch in range(self.epochs):
            total_reward = 0
            total_steps = 0
            state = self.reset()
            action = self.model.sample(state)
            while True:
                # 只有在step中使用真实的动作，其他地方使用id号
                next_state, reward, done = self.step(self.action_space[action])
                next_action = self.model.sample(next_state)
                # 训练
                self.model.learn(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                total_reward += reward
                total_steps += 1
                # if epoch % 5 == 0:
                # print('Episode %03s: steps = %02s , reward = %.1f' % (epoch, total_steps, total_reward))
                obs1_1 = self.u3_obs(self.state)
                obs1_2 = self.xkyd_obs(self.state)
                obs1 = list(np.sum([obs1_1, obs1_2], axis=0))
                a1 = np.sum(obs1)
                # print('配餐食材id：', self.state,'配餐分数：',a1)
                # print('配餐食材结果：', self.data.sc_id_name(self.state))
                # csv_writer.writerow([epoch, total_steps, total_reward, a1, self.state])
                if done:
                    break
        # self.model.save()
        return self.state, self.data.sc_id_name(self.state)

    def test(self):
        total_reward = 0
        actions = []
        state = self.reset()
        while True:
            action = self.model.predict(state)
            next_state, reward, done = self.step(action)
            state = next_state
            total_reward += reward
            actions.append(action)
            if done:
                break
        return actions, total_reward

    def step(self, action):  # 状态：已选的配餐食材 动作为食材的id号
        # 计算当前状态分数
        obs1_1 = self.u3_obs(self.state)
        obs1_2 = self.xkyd_obs(self.state)
        obs1 = list(np.sum([obs1_1, obs1_2], axis=0))
        a1 = np.sum(obs1)
        state = self.state + [action]
        # 计算下一状态分数
        state1 = np.unique(state)
        obs2_1 = self.u3_obs(state1)
        obs2_2 = self.xkyd_obs(state1)
        obs2 = list(np.sum([obs2_1, obs2_2], axis=0))
        a2 = np.sum(obs2)
        reward = a2 - a1
        # 判断是否终止
        done = False
        ac_c = self.data.d1_Scid[self.sid_index[int(action)]]  # 根据用户传入的食材id获取对应的类别
        if ac_c in self.pogoda_scid:
            ac_ind = self.pogoda_scid.index(ac_c)
            if self.u_c_num[ac_ind] < self.recip_c_num[ac_ind] and action not in self.state:
                self.state.append(action)  # 执行动作
                self.u_c_num[ac_ind] += 1
                # print(self.u_c_num[ac_ind])
            if self.u_c_num == self.recip_c_num and action not in self.state:  # 替换食材
                sc_temp = [action]  # 与action相同的食材id
                sc_fs = []  # s加入c_temp中食材的配餐分数
                sc_list1 = []  # 与action不相同的食材id
                u_scid = [self.data.d1_Scid[self.sid_index[int(i)]] for i in self.state]
                for k, i in enumerate(u_scid):
                    if i == ac_c:
                        sc_temp.append(self.state[k])
                    else:
                        sc_list1.append(self.state[k])
                for k, i in enumerate(sc_temp):
                    pc_l = sc_list1 + [i]
                    p1_1 = self.u3_obs(pc_l)
                    p1_2 = self.xkyd_obs(pc_l)
                    p1 = np.sum(list(np.sum([p1_1, p1_2], axis=0)))
                    sc_fs.append(p1)
                min_ind = sc_fs.index(np.min(sc_fs))
                sc_temp.pop(min_ind)  # 删除分数最小的食材
                self.state = sc_list1 + sc_temp
                self.th_times += 1
                if self.th_times > self.max_times:
                    done = True
        return len(self.state), 0.1 * reward, done

    # 计算精选传入配餐食材列表的相克宜搭分数
    def xkyd_obs(self, sc_list):
        pc_obs = [0. for _ in range(len(sc_list))]  # 初始化配餐食材列表的分数
        # 根据食材相克与宜搭给食材打分
        for k1, i1 in enumerate(sc_list):
            if i1 in self.data.d2_Sid:
                i1_yd = self.data.d2_YDSid[self.data.d2_Sid.index(i1)]
                i1_xk = self.data.d2_XKSid[self.data.d2_Sid.index(i1)]
                if not pd.isnull(i1_yd):
                    i1_yd = str(i1_yd).split(':')
                    for j1 in sc_list[k1 + 1:]:
                        if j1 in i1_yd:
                            pc_obs[k1] = pc_obs[k1] + 20  # 宜搭食材+20分
                if not pd.isnull(i1_xk):
                    i1_xk = str(i1_xk).split(':')
                    for j1 in sc_list[k1 + 1:]:
                        if j1 in i1_xk:
                            pc_obs[k1] = pc_obs[k1] - 10000  # 相克食材-10000分
        return pc_obs

    # 计算精选传入配餐食材列表的用户健康状况、食材偏好、上一次配餐的分数
    def u3_obs(self, sc_list):
        pc_obs = [0. for _ in range(len(sc_list))]  # 初始化配餐食材列表的分数
        # 根据用户健康状况给食材打分
        for i in self.symptoms:
            if i in self.data.d3_ZSid:
                i_YCZid = self.data.d3_YCZid[self.data.d3_Zid.index(i)]
                if not pd.isnull(i_YCZid):
                    i_YCZid = i_YCZid.split(':')
                    for k1, j1 in enumerate(sc_list):
                        if j1 in i_YCZid:
                            pc_obs[k1] = pc_obs[k1] + 20  # 宜吃食材+20分
                i_JCZid = self.data.d3_JCZid[self.data.d3_Zid.index(i)]
                if not pd.isnull(i_JCZid):
                    i_JCZid = i_JCZid.split(':')
                    for k1, j1 in enumerate(sc_list):
                        if j1 in i_JCZid:
                            pc_obs[k1] = pc_obs[k1] - 10000  # 忌吃食材-10000分
        for k1, i1 in enumerate(sc_list):
            # 根据用户爱好给食材打分
            if i1 in self.favourite:
                pc_obs[k1] = pc_obs[k1] + 25  # 用户喜好的食材+25分
            # 根据用户上一次配餐给食材打分
            if i1 in self.last_recipe:
                pc_obs[k1] = pc_obs[k1] + 20  # 用户上一次配餐的食材+20分
        return pc_obs

    # 膳食宝塔  参数：食材库, 分数表, 范围
    def sc_pogoda(self, sc_list, sc_obs_list, num_list):
        # 膳食宝塔食材类别，无油和盐
        pogoda_scid = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
        all_scid = self.data.d1_Scid  # 所有食材对应的食材类id
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


# Sarsa算法
class Sarsa:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, e_greed=0.1):
        # 参数
        # state_dim：状态个数   action_dim：动作个数
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((state_dim, action_dim))
        # print('初始化Q表：\n', self.Q)

    def sample(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.action_dim)  # 随机选择一个动作
        else:
            action = self.predict(state)  # 预测一个动作
        return action

    def predict(self, state):
        """ 根据输入观察值，预测输出的动作值 """
        # 给出状态下标，挑选出该状态对应的最大动作Q值的下标
        all_actions = self.Q[state, :]
        # print('得到该状态的所有动作Q值：', all_actions)
        max_action = np.max(all_actions)
        # print('得到该状态的最大动作Q值：', max_action)
        # 防止最大的 Q 值有多个，找出所有最大的 Q，然后再随机选择
        # where函数返回一个 array， 每个元素为下标
        max_action_list = np.where(all_actions == max_action)[0]
        # print('得到该状态的最大动作Q值下标的列表：', max_action_list)
        action = np.random.choice(max_action_list)
        # print('从最大动作Q值列表中随机选取一个动作下标：', action)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        # print(state, action, reward, next_state, next_action, done)
        # 更新Q值表
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.Q[next_state, next_action]
            # print('reward:',reward,'target_q:', target_q)
        self.Q[state, action] += self.lr * (target_q - self.Q[state, action])

    def save(self):
        if not os.path.exists('./data'):  # 是否存在这个文件夹
            os.makedirs('./data')  # 如果没有这个文件夹，那就创建一个
        npy_file = './data/sarsa_q_table.npy'
        np.save(npy_file, self.Q)
        # print(npy_file + ' saved.')

    def load(self, npy_file='./data/sarsa_q_table.npy'):
        self.Q = np.load(npy_file)
        # print(npy_file + ' loaded.')
