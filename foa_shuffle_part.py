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
        self.generate_best()
        self.generate_fruits()
        print('get data,she shape is {}'.format(self.location.shape))

    def get_data(self, data):
        return np.loadtxt(data)[:, 1:]

        # 贪心初始化一条最优路径

    def generate_best(self, flag=True):
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
                initial_path += self.dis_mat[index][index + 1]
            initial_path += self.dis_mat[index][0]
            return initial, initial_path

    def generate_fruits(self):
        best, initial_path = self.generate_best(True)
        self.fruits = [best]
        for _ in range(self.num_total - 1):
            a = np.random.randint(0, self.num_city)
            b = np.random.randint(0, self.num_city)
            a = min(a, b)
            b = max(a, b)
            part1 = list(best[:a])
            part2 = list(best[a:b])
            part3 = list(best[b:])
            if len(part2)>=1:
                try:
                    np.random.shuffle(part2)
                    part2 = list(part2)
                except:
                    import pdb
                    pdb.set_trace()
            res = part1 + part2 + part3
            self.fruits.append(res)

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

    def swap_part(self,list1,list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

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

    def shuffle_part(self,data):
        x = np.random.randint(0,len(data))
        y = np.random.randint(0,len(data))
        start = min(x,y)
        end = max(y,x)
        # x = np.random.randint(0,len(data)-10)
        # y = x+10
        part1 = data[:start]
        part2 = data[start:end]
        part3 = data[end:]
        # np.random.shuffle(part2)
        part2 = part2[::-1]
        res_list = [part1+part2+part3]
        return res_list
    def fly(self):
        self.dis_adp_total()
        sortindex = np.argsort(self.dis_adp)[::-1]
        best_adp = self.dis_adp[sortindex[0]]
        best_fruit = self.fruits[sortindex[0]]
        self.fruits = [best_fruit]
        tmp = 1
        while tmp < self.num_total:
            opt_2_res = self.shuffle_part(best_fruit.copy())
            self.fruits += opt_2_res
            tmp += len(opt_2_res)
        self.fruits = self.fruits[:self.num_total]

        return best_adp,best_fruit

    def run(self):
        print('==================Running the code==================')
        BEST_LIST = None
        best_adp = -math.inf
        with open(self.save_path, 'w') as f:
            for i in range(1, self.iteration + 1):
                max_adp, best_list = self.fly()
                if max_adp > best_adp:
                    best_adp = max_adp
                    BEST_LIST = best_list
            bestlen = self.adp_norm/best_adp
            return self.location[BEST_LIST],bestlen
import pandas as pd
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data
data = read_tsp('data/st70.tsp')
# data = read_tsp('data/burma14.tsp')
data = np.array(data)
data = data[:,1:]
for _ in range(10):
    foa = FOA(num_city=data.shape[0],num_total=50,iteration=1000,adp_norm=1000,data=data.copy(),save_path='sdsdsdssd.txt',save_freq=10)
    res, length = foa.run()
    print(length)
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