import math
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np

pq = PriorityQueue()


class Node(object):
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path = path
        self.bound = bound

    def __cmp__(self, other):
        return cmp(self.bound, other.bound)
    def __lt__(self, other):  # operator <
        return self.bound < other.bound
    def __str__(self):
        return str(tuple([self.level, self.path, self.bound]))


## 动态规划法
class DP(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, data)

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

    # 计算路径长度, goback:是否计算回到起始点的路径
    def compute_pathlen(self, path, dis_mat, goback=True):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        if goback:
            result = dis_mat[a][b]
        else:
            result = 0.0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def run(self,src=0):
        optimal_tour = []
        n = len(self.dis_mat)
        if not n:
            raise ValueError("Invalid adj Matrix")
        u = Node()
        PQ = PriorityQueue()
        optimal_length = 0
        v = Node(level=0, path=[0])
        min_length = float('inf')  # infinity
        v.bound = self.bound(self.dis_mat, v)
        PQ.put(v)
        while not PQ.empty():
            print(PQ.qsize())
            v = PQ.get()
            if v.bound < min_length:
                u.level = v.level + 1
                for i in filter(lambda x: x not in v.path, range(1, n)):
                    u.path = v.path[:]
                    u.path.append(i)
                    if u.level == n - 2:
                        l = set(range(1, n)) - set(u.path)
                        u.path.append(list(l)[0])
                        # putting the first vertex at last
                        u.path.append(0)

                        _len = self.length(self.dis_mat, u)
                        if _len < min_length:
                            min_length = _len
                            optimal_length = _len
                            optimal_tour = u.path[:]

                    else:
                        u.bound = self.bound(self.dis_mat, u)
                        if u.bound < min_length:
                            PQ.put(u)
                    # make a new node at each iteration! python it is!!
                    u = Node(level=u.level)

        # shifting to proper source(start of path)
        optimal_tour_src = optimal_tour
        if src is not 1:
            optimal_tour_src = optimal_tour[:-1]
            y = optimal_tour_src.index(src)
            optimal_tour_src = optimal_tour_src[y:] + optimal_tour_src[:y]
            optimal_tour_src.append(optimal_tour_src[0])

        return optimal_tour_src, optimal_length

    def length(self, adj_mat, node):
        tour = node.path
        # returns the sum of two consecutive elements of tour in adj[i][j]
        return sum([adj_mat[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)])

    def bound(self, adj_mat, node):
        path = node.path
        _bound = 0

        n = len(adj_mat)
        determined, last = path[:-1], path[-1]
        # remain is index based
        remain = filter(lambda x: x not in path, range(n))

        # for the edges that are certain
        for i in range(len(path) - 1):
            _bound += adj_mat[path[i]][path[i + 1]]

        # for the last item
        _bound += min([adj_mat[last][i] for i in remain])

        p = [path[0]] + list(remain)
        # for the undetermined nodes
        for r in remain:
            _bound += min([adj_mat[r][i] for i in filter(lambda x: x != r, p)])
        return _bound

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


data = read_tsp('data/bayg29.tsp')
data = np.array(data)
data = data[:, 1:]

foa = DP(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy())
Best_path, Best = foa.run()
print('规划的路径长度:{}'.format(Best))
# 显示规划结果
# plt.scatter(Best_path[:, 0], Best_path[:, 1])
# Best_path = np.vstack([Best_path, Best_path[0]])
# plt.plot(Best_path[:, 0], Best_path[:, 1])
# plt.title('规划结果')
# plt.show()

