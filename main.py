import math
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def reading_files(filename):
    list1 = []
    list2 = []
    file = open(filename)
    for line in file:
        if line.__contains__('crim'):
            continue
        values = [value.strip('"') for value in line.strip().split(',')]
        list2.append(float(values.pop()) / 1000)
        list1.append(values)
    data = np.array(list1, dtype=float)
    label = np.array(list2, dtype=float)
    return data, label


def distance(point1, point2):
    distance = 0
    # print(point1, point2)
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def gaussian_kernel(x, kernels, s):
    gaussian = []
    for kernel in kernels:
        # print(x, kernel)
        gaussian.append(math.exp(-1 * (distance(x, kernel) ** 2) / (2 * (s ** 2))))
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
    g = gaussian_kernel(x, c, s)
    answer = np.dot(g, w)
    # print(answer)
    return g, answer


def back_propagate(g, y, label, w, lr):
    error = label - y
    w += lr * error * np.array(g)

    # print(w)


def train_RBF(x, y, cluster_num=20, epoch=50, learning_rate=0.001):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init="auto").fit(x)
    centers = kmeans.cluster_centers_
    sigma = max_distance(centers)[0] / math.sqrt(2 * cluster_num)
    # print(sigma)
    wi = np.random.rand(cluster_num)
    # print(wi)

    for i in range(epoch):
        for j in range(len(x)):
            gus, ans = feed_forward(x[j], wi, centers, sigma)
            # print(ans)
            back_propagate(gus, ans, y[j], wi, learning_rate)

        # print(max(gaussian))

    with open('wights.pkl', 'wb') as file:
        pickle.dump(wi, file)
    with open('centers.pkl', 'wb') as file:
        pickle.dump(centers, file)
    with open('sigma.pkl', 'wb') as file:
        pickle.dump(sigma, file)


def test_RBF(x):
    with open('wights.pkl', 'rb') as file:
        wights = pickle.load(file)
    with open('centers.pkl', 'rb') as file:
        centers = pickle.load(file)
    with open('sigma.pkl', 'rb') as file:
        sigma = pickle.load(file)

    answer = []
    for i in range(len(x)):
        gus, ans = feed_forward(x[i], wights, centers, sigma)
        answer.append(ans)
    return answer


if __name__ == '__main__':
    # data, label = reading_files('BostonHousing.csv')
    data, label = reading_files('housing.csv')
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=7)

    train_RBF(train_data, train_labels)

    predictions = test_RBF(test_data)

    correct = 0
    for i in range(len(test_labels)):
        if abs(predictions[i] - test_labels[i]) < 100:
            correct += 1
    print(correct/len(test_labels))


