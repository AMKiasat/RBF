import csv
import numpy as np

def reading_files(filename):
    list1 = []
    # list2 = []
    file = open(filename)
    for line in file:
        if line.__contains__('crim'):
            continue
        values = [float(value.strip('"')) for value in line.strip().split(',')]
        # list2.append(scaled.pop())
        list1.append(values)
        print(values)
    data = np.array(list1, dtype=int)
    # label = np.array(list2)
    return data


if __name__ == '__main__':
    reading_files('BostonHousing.csv')
