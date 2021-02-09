import random
import math
import numpy as np
import matplotlib.pyplot as plt


class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        # fruits中存每一个个体是下标的list
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.fruits = self.greedy_init(self.dis_mat,num_total,num_city)
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        init_best = self.location[init_best]

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

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
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
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

    def ga_mutate(self, gene):
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
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits

        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1./best_score)
            print(i,1./best_score)
        print(1./best_score)
        return self.location[BEST_LIST], 1. / best_score


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
data = data[:, 1:]
Best, Best_path = math.inf, None

model = GA(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy())
path, path_len = model.run()
if path_len < Best:
    Best = path_len
    Best_path = path
# 加上一行因为会回到起点
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(Best_path[:, 0], Best_path[:,1])
Best_path = np.vstack([Best_path, Best_path[0]])
axs[0].plot(Best_path[:, 0], Best_path[:, 1])
axs[0].set_title('规划结果')
iterations = range(model.iteration)
best_record = model.best_record
axs[1].plot(iterations, best_record)
axs[1].set_title('收敛曲线')
plt.show()


