import random
import numpy as np
import math
import pandas as  pd
import tqdm


class FOA(object):
    def __init__(self, num_city, num_total, iteration, data, save_path, save_freq, **kwargs):
        self.num_city = num_city
        self.num_total = num_total
        self.iteration = iteration
        self.save_path = save_path
        self.save_freq = save_freq
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, self.location)
        # self.fruits = self.greedy_init(self.dis_mat, num_total, num_city)
        self.fruits = self.random_init(num_total,num_city)

    # 贪心初始化种群
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # def greedy_init_2(self,dis_mat,num_total,num_city):
    #     start_index = 0
    #     result = [[]]*num_total
    #     rest = [x for x in range(0, num_city)]
    #     # 所有起始点都已经生成了
    #     if start_index >= num_city:
    #         start_index = np.random.randint(0,num_city)
    #         result[i] = result[start_index].copy()
    #         continue
    #     current = start_index
    #     rest.remove(current)
    #     # 找到一条最近邻路径
    #     result_one = [current]
    #     while len(rest) != 0:
    #         tmp_min = math.inf
    #         tmp_choose = -1
    #         for x in rest:
    #             if dis_mat[current][x] < tmp_min:
    #                 tmp_min = dis_mat[current][x]
    #                 tmp_choose = x
    #
    #         current = tmp_choose
    #         result_one.append(tmp_choose)
    #         rest.remove(tmp_choose)
    #     # result[i] = result_one
    #     result[0] = result_one
    #     start_index += 1
    #     return result
    # 随机初始化种群
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        assert a != b
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            assert a != b
            result += dis_mat[a][b]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    # 对序列做2opt操作
    def single_GA(self, best, x):
        # 1.翻转法，根据序列的差异长度选择反转长度
        tmp = np.array(best) - np.array(x)
        l = np.where(tmp != 0)
        l = len(l[0])
        # top = max(self.num_city - l,3)
        top = 5
        print(l)
        a = np.random.randint(top)
        t = x[a:a+l]
        return x[:a] + t[::-1]+x[a+l:]


    # def _3opt(self, x):
    #     city_list = [i for i in range(self.num_city)]
    #     choice = np.random.choice(city_list, 2)
    #     choice.sort()
    #     i, j = choice
    #     a = x[:i]
    #     b = x[i:j]
    #     c = x[j:]
    #     tmp_best = []
    #     tmp_best_adp = []
    #     # param1
    #     # tmp = a+b+c
    #     # tmp_adp = 1./self.compute_pathlen(tmp, self.dis_mat)
    #     # tmp_best = [a + b + c, ]
    #     # tmp_best_adp  = [tmp_adp]
    #     # param2
    #     tmp = a + b + c[::-1]
    #     tmp_adp = 1. / self.compute_pathlen(tmp, self.dis_mat)
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(tmp_adp)
    #     # param3
    #     tmp = a + c[::-1] + b[::-1]
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(1. / self.compute_pathlen(tmp, self.dis_mat))
    #     # param4
    #     tmp = a + b[::-1] + c
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(1. / self.compute_pathlen(tmp, self.dis_mat))
    #     # param5
    #     tmp = a + c + b[::-1]
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(1. / self.compute_pathlen(tmp, self.dis_mat))
    #     # param6
    #     tmp = a + c[::-1] + b
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(1. / self.compute_pathlen(tmp, self.dis_mat))
    #     # param7
    #     tmp = a + b[::-1] + c[::-1]
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(1. / self.compute_pathlen(tmp, self.dis_mat))
    #     # param8
    #     tmp = a + c + b
    #     tmp_best.append(tmp)
    #     tmp_best_adp.append(1. / self.compute_pathlen(tmp, self.dis_mat))
    #
    #     max_adp = max(tmp_best_adp)
    #     max_index = tmp_best_adp.index(max_adp)
    #     result = tmp_best[max_index]
    #     return result

    # 迭代一次，返回适应度最高的个体和他的序列。
    def fly(self, fruits, num_total):
        # 选择适应度最高的个体
        self.dis_adp = self.compute_adp(self.fruits)
        # 从高到底适应度排序
        sortindex = np.argsort(-self.dis_adp)
        best_adp = self.dis_adp[sortindex[0]]
        best_fruit = self.fruits[sortindex[0]]
        # 保留种群中适应度最好的个体
        self.fruits[0] = best_fruit
        for i in range(1, num_total):
            # 2opt生成一个新个体
            x = self.fruits[i]
            opt_2_res = self.single_GA(best_fruit,x)
            self.fruits[i] = opt_2_res
        return best_adp, best_fruit

    # 总共的迭代过程，返回最终最好的长度，以及一段路径
    def run(self):
        best_fruit = None
        best_adp = -math.inf
        for i in range(self.iteration):
            tmp_adp, tmp_list = self.fly(self.fruits, self.num_total)
            if tmp_adp > best_adp:
                print('wwww')
                best_adp = tmp_adp
                best_fruit = tmp_list
        bestlen = 1.0 / best_adp
        return self.location[best_fruit], bestlen


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
data = np.array(data)
data = data[:, 1:]
result_ = []
runtimes = 1
import time

start_time = time.time()
for _ in tqdm.trange(runtimes):
    foa = FOA(num_city=data.shape[0], num_total=50, iteration=1000, data=data.copy(), save_path='sdsdsdssd.txt',
              save_freq=10)
    res, length = foa.run()
    result_.append(length)
end_time = time.time()
result_ = np.array(result_)

print('best:{:.2f}\tworst:{:.2f}\tmean:{:.2f}\tstd:{:.2f}\taverage_time:{:.2f}'.format(result_.min(), result_.max(),
                                                                                       result_.mean(), result_.std(), (
                                                                                                   end_time - start_time) / runtimes))

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
