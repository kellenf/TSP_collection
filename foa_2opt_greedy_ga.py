import random
import numpy as np
import math
import pandas as  pd


class FOA(object):
    def __init__(self, num_city, num_total, iteration, adp_norm, data, save_path, save_freq, **kwargs):
        self.num_city = num_city
        self.num_total = num_total
        self.dis_adp = [0 for _ in range(self.num_total)]
        self.iteration = iteration
        self.adp_norm = adp_norm
        self.save_path = save_path
        self.save_freq = save_freq
        self.location = data
        self.dis_mat = self.distance_p2p_mat()
        # self.generate_best()
        self.ga_choose_num = 10
        self.ga_ratio = 0.5
        self.cross_ratio = 0.9
        self.mutate_ratio = 0.9
        self.generate_fruits()
        print('get data,she shape is {}'.format(self.location.shape))

    def get_data(self, data):
        return np.loadtxt(data)[:, 1:]

    # 贪心初始化一条最优路径
    def generate_best(self,flag = True):
        if flag:
            initial = [0]
            rest = [x for x in range(1, self.num_city)]
            initial_path = 0
            while len(rest) != 0:
                start = initial[-1]
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if self.dis_mat[start][x] < tmp_min:
                        tmp_min = self.dis_mat[start][x]
                        tmp_choose = x
                initial_path += tmp_min
                initial.append(tmp_choose)
                rest.remove(tmp_choose)
            initial_path += self.dis_mat[tmp_choose][0]
            return initial, initial_path
        else:
            initial = [x for x in range(self.num_city)]
            initial_path = 0
            for index in range(self.num_city - 1):
                initial_path += self.dis_mat[index][index+1]
            initial_path += self.dis_mat[index][0]
            return initial, initial_path

    def generate_fruits(self):
        best, initial_path = self.generate_best(True)
        self.fruits = [np.array(best)]
        for _ in range(self.num_total - 1):
            tmp = best.copy()
            np.random.shuffle(tmp)
            self.fruits.append(np.array(tmp))

    # 对称矩阵，两个城市之间的距离 支持2d和3d
    def distance_p2p_mat(self):
        dis_mat = []
        for i in range(self.num_city):
            dis_mat_each = []
            for j in range(self.num_city):
                tmp = 0
                for k in range(len(self.location[i])):
                    tmp += pow(self.location[i][k] - self.location[j][k], 2)
                tmp = math.sqrt(tmp)
                dis_mat_each.append(tmp)
            dis_mat.append(dis_mat_each)
        return dis_mat

    # 目标函数计算,适应度计算，中间计算。适应度为1/总距离*10000
    def dis_adp_total(self, ):
        self.dis_adp = []
        for i in range(self.num_total):
            try:
                dis = self.dis_mat[self.fruits[i][self.num_city - 1]][self.fruits[i][0]]  # 回家
            except:
                import pdb
                pdb.set_trace()
            for j in range(self.num_city - 1):
                dis = self.dis_mat[self.fruits[i][j]][self.fruits[i][j + 1]] + dis
            dis_adp_each = self.adp_norm / dis
            self.dis_adp.append(dis_adp_each)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):
        x = np.array(x)
        y = np.array(y)
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        # start, end = np.random.choice(path_list,2)
        # order = list(random.sample(path_list,2))
        start = np.random.randint(len_-5)
        end = start +5
        # order.sort()
        # start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = int(np.where(y == sub)[0])
            if not (index >= start and index < end):
                x_conflict_index.append(index)
        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = int(np.where(x == sub)[0])
            if not (index >= start and index < end):
                y_confict_index.append(index)
        assert len(x_conflict_index) == len(y_confict_index)
        # 交叉
        tmp = x[start:end].copy()
        x[start:end] =  y[start:end]
        y[start:end] = tmp
        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return x, y

    def ga_cross(self, x, y):
        x = np.array(x)
        y = np.array(y)
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        # start, end = np.random.choice(path_list,2)
        order = list(random.sample(path_list,2))
        # start = np.random.randint(len_-3)
        # end = start +3
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = int(np.where(y == sub)[0])
            if not (index >= start and index < end):
                x_conflict_index.append(index)
        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = int(np.where(x == sub)[0])
            if not (index >= start and index < end):
                y_confict_index.append(index)
        assert len(x_conflict_index) == len(y_confict_index)
        # 交叉
        tmp = x[start:end].copy()
        x[start:end] =  y[start:end]
        y[start:end] = tmp
        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self,scores,choose_num):
        scores_order = np.argsort(scores)[::-1].copy()
        scores_order = scores_order[0:choose_num]
        genes_choose = []
        genes_score = []
        for sub in scores_order:
            genes_choose.append(self.fruits[sub])
            genes_score.append(scores[sub])
        return genes_score, genes_choose

    def ga_choose(self,genes_score,genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub*1.0/sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self,gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        # 获得优质父代
        scores = self.dis_adp
        genes_score, genes_choose = self.ga_parent(scores,self.ga_choose_num)
        ga_result = []
        # for sub in genes_choose[0:self.ga_choose_num]:
        #     ga_result.append(list(sub))
            # 轮盘赌方式对父代进行选择
        gene_x, gene_y = self.ga_choose(genes_score,genes_choose)
        # 交叉
        if np.random.rand() < self.cross_ratio:
            gene_x, gene_y = self.ga_cross(gene_x, gene_y)
        # 变异
        if np.random.rand() < self.mutate_ratio:
            gene_x_new = self.ga_mutate(gene_x)
        if np.random.rand() < self.mutate_ratio:
            gene_y_new = self.ga_mutate(gene_y)

        if not (gene_x in ga_result):
            ga_result.append(gene_x)
        if not (gene_y in ga_result):
            ga_result.append(gene_y)

        ga_result = ga_result[:self.num_total]
        ga_result = [np.array(x) for x in ga_result]
        return ga_result

    def get_best(self):
        self.dis_adp_total()
        sortindex = np.argsort(self.dis_adp)[::-1]
        best_adp = self.dis_adp[sortindex[0]]
        best_fruit = self.fruits[sortindex[0]]
        return  best_adp, best_fruit


    def opt_2(self, x):
        city_list = [i for i in range(self.num_city)]
        choice = np.random.choice(city_list, 2)
        choice.sort()
        res_list = []
        a, b = list(choice)
        part1 = list(x[:a])
        part2 = list(x[a:b])
        part3 = list(x[b:])

        # 1
        res = part1 + part2[::-1] + part3
        res_list.append(res)
        # 2
        list1, list2 = self.swap_part(part1, part3)
        res = list1 + part2 + list2
        res_list.append(res)
        # 3
        res = list1 + part2[::-1] + list2
        res_list.append(res)
        return res_list

    def fly(self):
        self.dis_adp_total()
        sortindex = np.argsort(self.dis_adp)[::-1]
        best_adp = self.dis_adp[sortindex[0]]
        best_fruit = self.fruits[sortindex[0]]
        fruits = [best_fruit]
        while len(fruits) < self.num_total*2:
            if np.random.rand() < self.ga_ratio:
                ga_result = self.ga()
                fruits += ga_result
            else:
                opt_2_result = self.opt_2(best_fruit)
                fruits += opt_2_result

        self.dis_adp_total()
        sortindex = np.argsort(self.dis_adp)[::-1]
        keep_index = sortindex[:self.num_total]

        self.fruits = np.array(fruits)[keep_index]
        self.fruits = list(fruits)

        return best_adp, best_fruit

    def run(self):
        print('==================Running the code==================')
        BEST_LIST = None
        best_adp = -math.inf

        with open(self.save_path, 'w') as f:
            for i in range(1, self.iteration + 1):
                max_adp, best_list = self.fly()
                # self.ga()
                if max_adp > best_adp:
                    best_adp = max_adp
                    BEST_LIST = best_list
            bestlen = self.adp_norm / best_adp
            return self.location[BEST_LIST], bestlen


import pandas as pd

# x = [t for t in range(10)]
# yy = x.copy()
# np.random.shuffle(yy)
#
# tx,ty = FOA.ga_cross(x,yy)
# print(tx,ty)


def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(int(x))
        tmp.append(tmpline)
    data = tmp
    return data


data = read_tsp('data/st70.tsp')
data = np.array(data)
data = data[:, 1:]
for _ in range(10):
    foa = FOA(num_city=data.shape[0], num_total=200, iteration=1000, adp_norm=1000, data=data.copy(),
              save_path='sdsdsdssd.txt', save_freq=10)
    res, result = foa.run()
    print(result)



# import os
# # print(os.getcwd())
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
# colors = ('#377eb8', '#ff7f00', '#4daf4a','#aaaaaa')
# def plot(i, data):
#
#     x = data[:,0]
#     y = data[:,1]
#     z = data[:,2]
#     ax.scatter(x, y, z,color=colors[i],marker = '+')
#     # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
#     # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
#     # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
#     ax.plot3D(data[:,0], data[:,1], data[:,2],color = colors[i])
# result = []
# for i in range(4):
#     data = pd.read_csv('path5_{}.csv'.format(i),header=None)
#     data = np.array(data)
#     foa = FOA(num_city=data.shape[0],num_total=20,iteration=100,adp_norm=1000,data=data,save_path='sdsdsdssd.txt',save_freq=10)
#     res,lenth = foa.run()
#
#     res = np.array(res).astype(np.int)
#     plot(i,res)
#
#     print(lenth)
#     # result.append(res)
# # result = np.concatenate(result,axis = 0)
# # print(result.shape)
# # plot(result)
# plt.show()
