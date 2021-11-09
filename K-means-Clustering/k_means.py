# Name: Yunika Upadhayaya
# Student ID - 1001631183

import sys
import numpy as np
import math
import random


# read in from the file
def read_files(filename):
    input_file = open(filename, 'r')
    file_lines = input_file.readlines()
    file_data = []
    for each_line in file_lines:
        array = each_line.split()
        array = [float(count) for count in array]
        file_data.append([array, -1])
    return file_data


# initialization
def check_initialization(initialized_name, file_data, num_k):
    val_cluster = 0
    if initialized_name == "round_robin":
        for indices in range(len(file_data)):
            file_data[indices][1] = val_cluster
            val_cluster = (val_cluster + 1) % num_k
    elif initialized_name == "random":
        for indices in range(len(file_data)):
            file_data[indices][1] = random.randint(0, num_k - 1)
    else:
        print("Please enter valid initialization type.")
        exit(1)
    return val_cluster, file_data


def find_grp(values):
    val = set(map(lambda count: count[1], values))
    return [[y[0] for y in values if y[1] == count] for count in val]


def get_cluster_means(value_group):
    mean_cluster = []
    for indices in range(k_num):
        mean_cluster.append(np.mean(value_group[indices], axis=0))
    return mean_cluster


def find_squareRoot(value1, value2):
    value_sum = 0
    for indices in range(len(value1)):
        value_sum += (value1[indices] - value2[indices]) ** 2
    return math.sqrt(value_sum)


# classification phase
def classification(mean_cluster, file_data):
    while True:
        create_cluster = []
        for indices in range(len(file_data)):
            distance_small = 99999
            val_cluster = -1
            for count in range(len(mean_cluster)):
                if find_squareRoot(file_data[indices][0], mean_cluster[count]) < distance_small:
                    val_cluster = count
                    distance_small = find_squareRoot(file_data[indices][0], mean_cluster[count])
            create_cluster.append([file_data[indices][0], val_cluster])

        check = True
        for indices in range(len(file_data)):
            if file_data[indices][1] != create_cluster[indices][1]:
                check = False
                break
        if check:
            break

        file_data = create_cluster.copy()
        value_group = find_grp(file_data)

        for indices in range(k_num):
            mean_cluster[indices] = np.mean(value_group[indices], axis=0)
    return file_data


# print the output
def print_output(file_data):
    for index in range(len(file_data)):
        if len(file_data[index][0]) == 1:
            print("{:10.4f} --> cluster {:d}".format(file_data[index][0][0], file_data[index][1] + 1))
        else:
            print("({:10.4f}, {:10.4f}) --> cluster {:d}".format(
                file_data[index][0][0], file_data[index][0][1], file_data[index][1] + 1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Improper format!\n Try: k_means.py <data_file> <K> <initialization>")
        exit(0)

    file_name = sys.argv[1]
    k_num = int(sys.argv[2])
    initialize_name = sys.argv[3]

    all_data = read_files(file_name)
    num_cluster, all_data = check_initialization(initialize_name, all_data, k_num)
    value_groups = find_grp(all_data)
    mean_cluster_values = get_cluster_means(value_groups)
    all_data = classification(mean_cluster_values, all_data)
    print_output(all_data)

