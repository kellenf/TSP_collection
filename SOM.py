'''
notice: this code is referenced by other
https://github.com/DiegoVicen/som-tsp
'''
import random
import math
import numpy as np
import matplotlib.pyplot as plt


class SOM(object):
    def __init__(self, num_city, data):
        self.num_city = num_city
        self.location = data.copy()
        self.iteraton = 8000
        self.learning_rate = 0.8
        self.dis_mat = self.compute_dis_mat(num_city, self.location)
        self.best_path = []
        self.best_length = math.inf
        self.iter_x = []
        self.iter_y = []

    def normalize(self, points):
        """
        Return the normalized version of a given vector of points.
        For a given array of n-dimensions, normalize each dimension by removing the
        initial offset and normalizing the points in a proportional interval: [0,1]
        on y, maintining the original ratio on x.
        """
        ratio = (points[:, 0].max() - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()), 1
        ratio = np.array(ratio) / max(ratio)
        m = lambda c: (c - c.min()) / (c.max() - c.min())
        norm = m(points)
        # norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
        m = lambda p: ratio * p
        return m(norm)
        # return norm.apply(lambda p: ratio * p, axis=1)

    def generate_network(self, size):
        """
        Generate a neuron network of a given size.
        Return a vector of two dimensional points in the interval [0,1].
        """
        return np.random.rand(size, 2)

    def get_neighborhood(self, center, radix, domain):
        """Get the range gaussian of given radix around a center index."""

        # Impose an upper bound on the radix to prevent NaN and blocks
        if radix < 1:
            radix = 1

        # Compute the circular network distance to the center
        deltas = np.absolute(center - np.arange(domain))
        distances = np.minimum(deltas, domain - deltas)

        # Compute Gaussian distribution around the given center
        return np.exp(-(distances * distances) / (2 * (radix * radix)))

    def get_route(self, cities, network):
        """Return the route computed by a network."""
        f = lambda c: self.select_closest(network, c)
        dis = []
        for city in cities:
            dis.append(f(city))
        index = np.argsort(dis)
        return index

    def select_closest(self, candidates, origin):
        """Return the index of the closest candidate to a given point."""
        return self.euclidean_distance(candidates, origin).argmin()

    def euclidean_distance(self, a, b):
        """Return the array of distances of two numpy arrays of points."""
        return np.linalg.norm(a - b, axis=1)

    def route_distance(self, cities):
        """Return the cost of traversing a route of cities in a certain order."""
        points = cities[['x', 'y']]
        distances = self.euclidean_distance(points, np.roll(points, 1, axis=0))
        return np.sum(distances)

    # 随机初始化
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

    # 计算一条路径的长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def smo(self):
        citys = self.normalize(self.location)
        n = citys.shape[0] * 8
        network = self.generate_network(n)

        for i in range(self.iteraton):
            index = np.random.randint(self.num_city - 1)
            city = citys[index]
            winner_idx = self.select_closest(network, city)

            gaussian = self.get_neighborhood(winner_idx, n // 10, network.shape[0])

            network += gaussian[:, np.newaxis] * self.learning_rate * (city - network)

            self.learning_rate = self.learning_rate * 0.99997
            n = n * 0.9997
            if n < 1:
                break
            route = self.get_route(citys, network)
            route_l = self.compute_pathlen(route, self.dis_mat)
            if route_l < self.best_length:
                self.best_length = route_l
                self.best_path = route
            self.iter_x.append(i)
            self.iter_y.append(self.best_length)
            print(i, self.iteraton, self.best_length)
            # 画出初始化的路径
            if i == 0:
                plt.subplot(2, 2, 2)
                plt.title('convergence curve')
                show_data = self.location[self.best_path]
                show_data = np.vstack([show_data, show_data[0]])
                plt.plot(show_data[:, 0], show_data[:, 1])

        return self.best_length, self.best_path

    def run(self):
        self.best_length, self.best_path = self.smo()
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
plt.suptitle('PSO in st70.tsp')
data = data[:, 1:]
plt.subplot(2, 2, 1)
plt.title('raw data')
# 加上一行因为会回到起点
show_data = np.vstack([data, data[0]])
plt.plot(data[:, 0], data[:, 1])

model = SOM(num_city=data.shape[0], data=data.copy())
Best_path, Best_length = model.run()


Best_path = np.vstack([Best_path, Best_path[0]])
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(Best_path[:, 0], Best_path[:,1])
Best_path = np.vstack([Best_path, Best_path[0]])
axs[0].plot(Best_path[:, 0], Best_path[:, 1])
axs[0].set_title('规划结果')
iterations = model.iter_x
best_record = model.iter_y
axs[1].plot(iterations, best_record)
axs[1].set_title('收敛曲线')
plt.show()
