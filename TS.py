import random
import math
import numpy as np
import matplotlib.pyplot as plt


class TS(object):
    def __init__(self, num_city, data):
        self.taboo_size = 80
        self.iteration = 1000
        self.num_city = num_city
        self.location = data
        self.taboo = []
        self.path = self.random_init(num_city)
        self.best_path = self.path
        self.cur_path = self.path
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.best_length = self.compute_pathlen(self.path, self.dis_mat)

        # 显示初始化后的路径
        init_pathlen = 1. / self.compute_pathlen(self.path, self.dis_mat)
        init_best = self.location[self.path]
        init_best = np.vstack((init_best, init_best[0]))
        plt.subplot(2, 2, 2)
        plt.title('init best result')
        plt.plot(init_best[:, 0], init_best[:, 1])
        # 存储结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / init_pathlen]

    # 初始化一条随机路径
    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        random.shuffle(tmp)
        return tmp

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
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 产生随机解
    def ts_search(self, x):
        moves = []
        new_paths = []
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                tmp = x.copy()
                tmp[i], tmp[j] = tmp[j], tmp[i]
                new_paths.append(tmp)
                moves.append([i, j])
        return new_paths, moves

    # 禁忌搜索
    def ts(self):
        for cnt in range(self.iteration):
            new_paths, moves = self.ts_search(self.cur_path)
            new_lengths = self.compute_paths(new_paths)
            sort_index = np.argsort(new_lengths)
            min_l = new_lengths[sort_index[0]]
            min_path = new_paths[sort_index[0]]
            min_move = moves[sort_index[0]]

            # 更新当前的最优路径
            if min_l < self.best_length:
                self.best_length = min_l
                self.best_path = min_path
                self.cur_path = min_path
                # 更新禁忌表
                if min_move in self.taboo:
                    self.taboo.remove(min_move)

                self.taboo.append(min_move)
            else:
                # 找到不在禁忌表中的操作
                while min_move in self.taboo:
                    sort_index = sort_index[1:]
                    min_l = new_lengths[sort_index[0]]
                    min_path = new_paths[sort_index[0]]
                    min_move = moves[sort_index[0]]
                self.cur_path = min_path
                assert self.cur_path != self.best_path
                self.taboo.append(min_move)
            # 禁忌表超长了
            if len(self.taboo) > self.taboo_size:
                self.taboo = self.taboo[1:]
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_length)
            print(cnt, self.best_length)
        print(self.best_length)

    def run(self):
        self.ts()
        plt.subplot(2, 2, 4)
        plt.title('convergence curve')
        plt.plot(self.iter_x, self.iter_y)
        return self.location[self.best_path], self.best_length


# 读取数据
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
plt.suptitle('TS in st70.tsp')
data = data[:, 1:]
plt.subplot(2, 2, 1)
plt.title('raw data')
show_data = np.vstack([data, data[0]])
plt.plot(data[:, 0], data[:, 1])

func = TS(num_city=data.shape[0], data=data.copy())
best_path, best_length = func.run()
plt.subplot(2, 2, 3)
# 加上一行因为会回到起点
best_path = np.vstack([best_path, best_path[0]])
plt.plot(best_path[:, 0], best_path[:, 1])
plt.title('result')
plt.show()
