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
        list2.append(values.pop())
        list1.append(values)
    data = np.array(list1, dtype=float)
    label = np.array(list2)
    return data, label


def distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def gaussian_kernel(x, kernels, s):
    gaussian = []
    for i in kernels:
        gaussian.append(math.exp((-1 * distance(x, i) ** 2) / 2 * s ** 2))
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


def train_RBF(x, y, cluster=20, Sigma=4):
    kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(x)
    centers = kmeans.cluster_centers_
    # print(centers[0])
    # print(centers[17])
    sigma = max_distance(centers)[0] / math.sqrt(2 * len(centers))
    # print(sigma)
    print(np.max(gaussian_kernel(x[0], centers, sigma)))
    print(gaussian_kernel(x[0], centers, sigma))


if __name__ == '__main__':
    # data, label = reading_files('BostonHousing.csv')
    data, label = reading_files('housing.csv')
    # for i in range(len(data)):
    #     print(data[i], label[i])

    train_RBF(data, label)
