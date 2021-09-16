# Name: Yunika Upadhayaya
# Student Id: 1001631183

import math
import statistics
import sys
from classes import Classifier
from classes import Attribute
from classes import Object

all_classes = []
class_objects = []


def std_dev(lines, avg):
    num = len(lines)
    avg = sum(lines) / num
    deviations = [(data - avg) ** 2 for data in lines]
    variance = sum(deviations) / (num - 1)
    if variance < 0.0001:
        variance = 0.0001
    stddev = math.sqrt(variance)
    return stddev


def read_file(file_name):
    try:
        with open(file_name) as file:
            lines = []
            for each_line in file:
                lines.append(each_line.rstrip())
    except FileNotFoundError:
        output = "The file " + file_name + " does not exist. Try again."
        print(output)
        exit()
    return lines


def gaussian(value, stddev, mean):
    x = 1 / (stddev * math.sqrt(2 * math.pi))
    y = math.exp(-1 * (((value - mean) ** 2) / (2 * stddev ** 2)))
    product = x * y
    return product


def find_accuracy(obj_class, probability_class):
    max_values = []

    for a, probability in enumerate(probability_class):
        if probability == max(probability_class):
            max_values.append(a + 1)

    if len(max_values) == 1:
        if max_values[0] != obj_class:
            return 0
        else:
            return 1
    else:
        if not (obj_class in max_values):
            return 0
        else:
            return 1 / len(max_values)


def naive_bayes(training_file, testing_file):
    # training phase
    class_level = []
    train_data = read_file(training_file)

    for each_line in train_data:
        string = each_line.split()
        temp_float = []
        for item in string:
            temp_float.append(float(item))

        if not (temp_float[-1] in class_level):
            class_level.append(temp_float[-1])

    length = len(class_level)
    for each_item in range(0, length):
        all_classes.append(Classifier(each_item + 1))
        length_data = len(train_data[0].split()) - 1
        for item in range(0, length_data):
            all_classes[each_item].attr.append(Attribute(item + 1))

    for each_item in train_data:
        string = each_item.split()
        temp_float = []
        for item in string:
            temp_float.append(float(item))
        num_class = int(temp_float[-1])

        for a in all_classes:
            if num_class == a.class_id:
                for token, i in enumerate(temp_float[:-1]):
                    a.attr[token].nums.append(i)

    for value in all_classes:
        len1 = len(value.attr[0].nums)
        len2 = len(train_data)
        value.probability = len1 / len2

    for a in all_classes:
        for b in a.attr:
            if not (len(b.nums) == 0):
                b.mean_value = statistics.mean(b.nums)
            b.stdDeviation_value = std_dev(b.nums, b.mean_value)

    for a in all_classes:
        for b in a.attr:
            print(
                f"Class {a.class_id:d}, attribute {b.id_attribute:d}, mean = {b.mean_value:.2f}, std = {b.stdDeviation_value:.2f}")

    # classification phase
    print()
    testing_data = read_file(testing_file)
    for token, k in enumerate(testing_data):
        string = k.split()
        temp_float = []
        for item in string:
            temp_float.append(float(item))
        obj_temp = Object(token + 1, temp_float[-1])

        length = len(all_classes)
        for item in range(0, length):
            given_xc = 1

            for index, i in enumerate(temp_float[:-1]):
                token_attr = all_classes[item].attr[index]
                given_xc *= gaussian(i, token_attr.stdDeviation_value, token_attr.mean_value)

            obj_temp.p_xc.append(given_xc)

        for x in range(0, length):
            obj_temp.px += (obj_temp.p_xc[x] * all_classes[x].probability)

        probability_class = []

        for a in range(0, length):
            probability_class.append((obj_temp.p_xc[a] * all_classes[a].probability) / obj_temp.px)

        obj_temp.probability = max(probability_class)
        obj_temp.prob_class = probability_class.index(max(probability_class)) + 1
        obj_temp.acc = find_accuracy(obj_temp.o_class, probability_class)
        print("ID= %5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (obj_temp.Id, obj_temp.prob_class, obj_temp.probability, obj_temp.o_class, obj_temp.acc))

        class_objects.append(obj_temp)

    sum_acc = 0
    for item in class_objects:
        sum_acc += item.acc

    print()
    length = len(class_objects)
    print("Classification accuracy = %6.4f" % (sum_acc/length))


if len(sys.argv) != 3:
    print("Improper format!\n Try: naive_bayes.py <training_file> <test_file>")
    exit()

naive_bayes(sys.argv[1], sys.argv[2])



