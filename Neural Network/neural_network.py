# Name: Yunika Upadhayaya
# Student ID: 1001631183

import numpy as np
import sys


# calculate target hot vector
def vector_target(dim, g, data_b_num):
    target_hot_vector = np.zeros((dim, g), dtype=int)
    for e, temp in enumerate(target_hot_vector):
        d = data_b_num[e]
        target_hot_vector[e][d - 1] = 1
    return target_hot_vector


# activation function
def sigmoid_function(sum_weight):
    res = 1 / (1 + np.exp(-sum_weight))
    return res


# function to normalize data
def normalize_function(data_a1):
    maximum_abs = np.amax(data_a1)
    data_a1 = np.divide(data_a1, maximum_abs)
    return data_a1


# function to generate random weight
def random_weight_generator(layers_num, unit_per_layer_num, dimension, g):
    dict_weight = {}
    dict_bias = {}
    index_key = 2
    hidden_layer_num = layers_num - 2
    if hidden_layer_num == 1:
        weight1 = np.random.uniform(-0.045, 0.045, (unit_per_layer_num, dimension))
        weight2 = np.random.uniform(-0.045, 0.045, (g, unit_per_layer_num))
        bias = np.random.uniform(-0.045, 0.045, unit_per_layer_num)
        dict_bias[index_key] = bias
        dict_weight[index_key] = weight1
        index_key = index_key + 1
        bias1 = np.random.uniform(-0.045, 0.045, g)
        dict_bias[index_key] = bias1
        dict_weight[index_key] = weight2
    elif hidden_layer_num == 0:
        weight1 = np.random.uniform(-0.045, 0.045, (g, dimension))
        dict_weight[index_key] = weight1
        bias1 = np.random.uniform(-0.045, 0.045, g)
        dict_bias[index_key] = bias1
    elif hidden_layer_num >= 2:
        weight1 = np.random.uniform(-0.045, 0.045, (unit_per_layer_num, dimension))
        bias = np.random.uniform(-0.045, 0.045, unit_per_layer_num)
        dict_bias[index_key] = bias
        dict_weight[index_key] = weight1
        index_key += 1
        for e in range(0, hidden_layer_num - 1):
            weight_x = np.random.uniform(-0.045, 0.045, (unit_per_layer_num, unit_per_layer_num))
            bias_x = np.random.uniform(-0.045, 0.045, unit_per_layer_num)
            dict_weight[index_key] = weight_x
            dict_bias[index_key] = bias_x
            index_key += 1

        weight2 = np.random.uniform(-0.045, 0.045, (g, unit_per_layer_num))
        bias1 = np.random.uniform(-0.045, 0.045, g)
        dict_weight[index_key] = weight2
        dict_bias[index_key] = bias1
    return dict_weight, dict_bias


# function to calculate the sum of the calculated weight
def sum_weight_calculate(dict_bias, dict_weight, dict_b, i, dict_a):
    dict_a[i] = []
    dict_b[i] = []
    length = len(dict_bias[i])

    for e in range(0, length):
        sum_weight = dict_bias[i][e]
        previous_layer = len(dict_b[i - 1])
        for unique_classes in range(0, previous_layer):
            sum_weight = sum_weight + dict_b[i - 1][unique_classes] * dict_weight[i][e][unique_classes]
        d = sigmoid_function(sum_weight)
        dict_b[i].append(d)
        dict_a[i].append(sum_weight)
    return dict_a, dict_b


# calculation of neural network
def neural_network(training_file, testing_file, layers_num, unit_per_layer_num, round_num):
    training_data = np.loadtxt(training_file, dtype=str)
    testing_data = np.loadtxt(testing_file, dtype=str)

    data_a = training_data[:, :-1]
    data_b = training_data[:, -1:]
    data_a1 = (np.asmatrix(data_a)).astype(float)

    test_data_a = testing_data[:, :-1]
    test_data_b = testing_data[:, -1:]
    test_data_a1 = (np.asmatrix(test_data_a)).astype(float)

    # training phase
    dim, dimension = data_a1.shape

    data_b = data_b.flatten()

    unique_classes = np.unique(data_b)
    unique_class_list = list(unique_classes)
    unique_class_num = []
    for e, temp in enumerate(unique_classes):
        unique_class_num.append(e + 1)

    g = len(unique_classes)
    data_b_num = []

    for temp in data_b:
        index = unique_class_list.index(temp)
        target = unique_class_num[index]
        data_b_num.append(target)

    target_hot_vector = vector_target(dim, g, data_b_num)
    normal_data_a = normalize_function(data_a1)

    dict_weight, dict_bias = random_weight_generator(layers_num, unit_per_layer_num, dimension, g)

    dict_a = {}
    dict_b = {}
    dict_delta = {}
    et = 1

    print("Training phase started...\n")
    for h in range(0, round_num):
        print("Starting round", h, "...\n")
        for new, temp in enumerate(normal_data_a):
            new_line = np.array(normal_data_a[new])
            dict_b[1] = new_line[0]
            for i in range(2, layers_num + 1):
                dict_a, dict_b = sum_weight_calculate(dict_bias, dict_weight, dict_b, i, dict_a)
            dict_delta[layers_num] = []
            for j in range(0, len(dict_b[layers_num])):
                val = (dict_b[layers_num][j] - target_hot_vector[new][j]) * dict_b[layers_num][j] * (
                        1 - dict_b[layers_num][j])
                dict_delta[layers_num].append(val)
            for l in range(layers_num - 1, 1, -1):
                dict_delta[l] = []
                for m in range(0, len(dict_b[l])):
                    val1 = 0
                    for k in range(0, len(dict_delta[l + 1])):
                        val1 = val1 + dict_delta[l + 1][k] * dict_weight[l + 1][k][m]
                    val1 = val1 * dict_b[l][m] * (1 - dict_b[l][m])
                    dict_delta[l].append(val1)
            for n in range(2, layers_num + 1):
                for o in range(0, len(dict_b[n])):
                    dict_bias[n][o] = dict_bias[n][o] - et * dict_delta[n][o]
                    for p in range(0, len(dict_b[n - 1])):
                        dict_weight[n][o][p] = dict_weight[n][o][p] - et * dict_delta[n][o] * dict_b[n - 1][p]
        et = et * 0.98
    print("\n\nDone with training phase!!!\n\n")

    # testing phase
    x_test, y_test = test_data_a1.shape
    test_data_b = test_data_b.flatten()

    test_unique = np.unique(test_data_b)
    test_unique_list = list(unique_classes)
    test_unique_num = []
    for e, temp in enumerate(test_unique):
        test_unique_num.append(e + 1)

    test_data_b_num = []
    for temp in test_data_b:
        index = test_unique_list.index(temp)
        target = test_unique_num[index]
        test_data_b_num.append(target)

    normal_test_data_a = normalize_function(test_data_a1)

    test_dict_b = {}
    test_dict_a = {}
    vector_accuracy = []

    for new in range(0, x_test):
        new_line = np.array(normal_test_data_a[new])
        test_dict_b[1] = new_line[0]
        for i in range(2, layers_num + 1):
            test_dict_a, test_dict_b = sum_weight_calculate(dict_bias, dict_weight, test_dict_b, i, test_dict_a)

        predicted = np.argmax(test_dict_b[layers_num])
        id_object = new + 1
        class_true = test_data_b[new]
        if test_data_b_num[new] != test_unique_num[predicted]:
            acc = 0.0
        else:
            acc = 1.0
        vector_accuracy.append(acc)

        predicted_class = unique_class_list[predicted]

        print("ID=%5d, predicted=%10s, true=%10s, acc=%4.2f" % (id_object, predicted_class, class_true, acc))

    class_accuracy = (sum(vector_accuracy) / len(vector_accuracy))
    print("Classification accuracy=%6.4f" % class_accuracy)


if len(sys.argv) != 6:
    print(
        "Improper format!\n Try: neural_network.py <training_file> <testing_file> <layers_num> <unit_per_layer_num> <rounds>")
    exit()
neural_network(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
