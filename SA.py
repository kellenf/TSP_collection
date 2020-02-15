import random
import math
import numpy as np
import matplotlib.pyplot as plt


class SA(object):
    def __init__(self, num_city, data):
        self.T0 = 2000
        self.Tend = 1e-3
        self.rate = 0.94
        self.size = 100
        self.num_city = num_city
        self.scores = []
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.2
        # fruits中存每一个个体是下标的list
        self.fire = self.random_init(num_city)
        self.fires = []
        self.dis_mat = self.compute_dis_mat(num_city, data)

        # 显示初始化后的路径
        init_pathlen = 1. / self.compute_pathlen(self.fire, self.dis_mat)
        init_best = self.location[self.fire]
        init_best = np.vstack((init_best, init_best[0]))
        plt.subplot(2, 2, 2)
        plt.title('init best result')
        plt.plot(init_best[:, 0], init_best[:, 1])
        # 存储存储每个温度下的最终路径，画出收敛图
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

    # 计算一个温度下产生的一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 产生一个新的解：随机交换两个元素的位置
    def get_new_fire(self, fire):

        t = [x for x in range(len(fire))]
        a, b = np.random.choice(t, 2)
        # 1.交换两个元素的位置，效果不好
        # fire[a], fire[b] = fire[b], fire[a]
        # return  fire

        # 2片段逆转，效果比1好
        x = min(a, b)
        y = max(a, b)
        a = fire[:x]
        b = fire[x:y][::-1]
        c = fire[y:]
        return a + b + c

    # 退火策略，根据温度变化有一定概率接受差的解
    def eval_fire(self, raw, get, temp):
        len1 = self.compute_pathlen(raw, self.dis_mat)
        len2 = self.compute_pathlen(get, self.dis_mat)
        dc = len2 - len1
        p = max(1e-4, np.exp(-dc / temp))
        if len2 < len1:
            return get
        elif np.random.rand() <= p:
            return get
        else:
            return raw

    # 模拟退火总流程
    def sa(self):
        count = 0
        # 记录最优解
        best_path = self.fire
        best_length = self.compute_pathlen(self.fire, self.dis_mat)

        while self.T0 > self.Tend:
            count += 1
            # 产生在这个温度下的一些随机解
            tmp_fires = []
            for _ in range(self.size):
                tmp_new = self.get_new_fire(self.fire.copy())
                self.fire = self.eval_fire(self.fire, tmp_new, self.T0)
                tmp_fires.append(tmp_new)
            # 记录当前温度下的最优解
            lengths = self.compute_paths(tmp_fires)
            min_length = min(lengths)
            min_index = lengths.index(min_length)
            min_path = tmp_fires[min_index]
            # 更新最优解
            if min_length < best_length:
                best_length = min_length
                best_path = min_path
            # 降低温度
            self.T0 *= self.rate
            # 记录路径收敛曲线
            self.iter_x.append(count)
            self.iter_y.append(best_length)
        return best_length, best_path

    def run(self):
        best_length, best_path = self.sa()
        plt.subplot(2, 2, 4)
        plt.title('convergence curve')
        plt.plot(self.iter_x, self.iter_y)
        return self.location[best_path], best_length


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
plt.suptitle('SA in st70.tsp')
data = data[:, 1:]
plt.subplot(2, 2, 1)
plt.title('raw data')
show_data = np.vstack([data, data[0]])
plt.plot(data[:, 0], data[:, 1])
Best, Best_path = math.inf, None

foa = SA(num_city=data.shape[0], data=data.copy())
path, path_len = foa.run()
if path_len < Best:
    Best = path_len
    Best_path = path
plt.subplot(2, 2, 3)
# 加上一行因为会回到起点
Best_path = np.vstack([Best_path, Best_path[0]])
plt.plot(Best_path[:, 0], Best_path[:, 1])
plt.title('result')
plt.show()
