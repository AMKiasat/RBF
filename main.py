import math
import numpy as np
from sklearn.cluster import KMeans


def reading_files(filename):
    list1 = []
    list2 = []
    file = open(filename)
    for line in file:
        if line.__contains__('crim'):
            continue
        values = [value.strip('"') for value in line.strip().split(',')]
        list2.append(float(values.pop()) / 100000)
        list1.append(values)
    data = np.array(list1, dtype=float)
    label = np.array(list2, dtype=float)
    return data, label


def distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def gaussian_kernel(x, kernels, s):
    gaussian = []
    for i in kernels:
        gaussian.append(math.exp(-1 * (distance(x, i) ** 2)) / (2 * (s ** 2)))
    return gaussian


def max_distance(x):
    max = [0, 0, 0]
    for i in range(len(x) - 1):
        for j in range(len(x) - 1):
            dis = distance(x[i], x[j])
            if dis > max[0]:
                max = [dis, i, j]
                # print(max)
                # print(x[i], x[j])
    return max


def feed_forward(x, w, c, s):
    gaussian = []
    answer = []
    for i in range(len(x)):
        g = gaussian_kernel(x[i], c, s)
        gaussian.append(g)
        answer.append(np.dot(g, w))
    return np.array(gaussian), answer


def back_propagate(x, y, label, w, lr):
    error = label - y
    for i in range(len(x)):
        w += lr * error[i] * x[i]


def train_RBF(x, y, cluster_num=20, epoch=2000, learning_rate=0.01):
    kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(x)
    centers = kmeans.cluster_centers_
    sigma = max_distance(centers)[0] / math.sqrt(2 * len(centers))
    wi = np.random.rand(cluster_num)
    # print(wi)

    for i in range(epoch):
        gus, ans = feed_forward(x, wi, centers, sigma)
        # print(len(gus[0]))
        # print(gus[0])
        print(ans)
        back_propagate(gus, ans, y, wi, learning_rate)

        # print(max(gaussian))


if __name__ == '__main__':
    # data, label = reading_files('BostonHousing.csv')
    data, label = reading_files('housing.csv')
    # for i in range(len(data)):
    #     print(data[i], label[i])

    train_RBF(data, label)

