# Name - Yunika Upadhayaya
# ID - 1001631183

import sys
import numpy as np
import math
import random
from collections import Counter


class make_tree(object):

    def __init__(self, attribute_fine, threshold_fine, node_id):
        self.attribute_fine = attribute_fine
        self.threshold_fine = threshold_fine
        self.child_on_left = None
        self.child_on_right = None
        self.calculate_gain = 0
        self.calculate_distribution = None
        self.node_id = node_id


def decision_tree_level(data_eg, attribute, defaults, given_mode = "randomized", node_id = 1):
    if data_eg.shape[0] < pruning_threshold:
        all_tree = make_tree(-1, -1, node_id = node_id)
        all_tree.calculate_distribution = calculate_distribution(data_eg)
        return all_tree
    elif len(np.unique(data_eg[:, -1])) == 1:
        all_tree = make_tree(-1, -1, node_id = node_id)
        all_tree.calculate_distribution = [0 if a != data_eg[0, -1] else 1 for a in range(len(classes_num))]
        return all_tree
    else:
        attribute_fine, threshold_fine, highest_gain = calculate_attributes(data_eg, attribute, given_mode)
        all_tree = make_tree(attribute_fine, threshold_fine, node_id)
        left_value_examples = data_eg[data_eg[:, attribute_fine] < threshold_fine]
        right_value_examples = data_eg[data_eg[:, attribute_fine] >= threshold_fine]
        all_tree.calculate_gain = highest_gain
        all_tree.child_on_left = decision_tree_level(left_value_examples, attribute, calculate_distribution(data_eg), given_mode = given_mode, node_id = 2*node_id)
        all_tree.child_on_right = decision_tree_level(right_value_examples, attribute, calculate_distribution(data_eg), given_mode = given_mode, node_id = 2 * node_id + 1)
    return all_tree


def calculate_info_gain(data_eg, attribute, calc_threshold):
    data_left = data_eg[data_eg[:, attribute] < calc_threshold]
    data_right = data_eg[data_eg[:, attribute] >= calc_threshold]
    attribute_tar = list(data_eg[:, -1])

    dictionary = Counter(attribute_tar)
    attribute_tar = np.asarray(attribute_tar)
    attribute_tar = np.unique(attribute_tar)
    base_entropy = 0
    for a in range(len(attribute_tar)):
        if dictionary[attribute_tar[a]] > 0:
            base_entropy = base_entropy - ((dictionary[attribute_tar[a]]) / len(data_eg)) * math.log(
                (dictionary[attribute_tar[a]] / len(data_eg)), 2)

    left_data_dict = Counter(data_left[:, -1])
    left_entropy_value = 0
    left_value_target = np.asarray(data_left)
    left_value_target = np.unique(left_value_target)
    for a in range(len(left_value_target)):
        if left_data_dict[left_value_target[a]] > 0:
            left_entropy_value = left_entropy_value - (
                        (left_data_dict[left_value_target[a]]) / len(data_left)) * math.log(
                ((left_data_dict[left_value_target[a]]) / len(data_left)), 2)

    right_data_dict = Counter(data_right[:, -1])
    right_value_target = np.asarray(data_right)
    right_value_target = np.unique(right_value_target)

    right_entropy_value = 0
    for a in range(len(right_value_target)):
        if right_data_dict[right_value_target[a]] > 0:
            right_entropy_value = right_entropy_value - (
                        (right_data_dict[right_value_target[a]]) / len(data_right)) * math.log(
                ((right_data_dict[right_value_target[a]]) / len(data_right)), 2)

    calculate_gain = base_entropy - (len(data_left) / len(data_eg)) * left_entropy_value - (
                len(data_right) / len(data_eg)) * right_entropy_value
    return calculate_gain


def calculate_distribution(data_eg):
    if len(data_eg) == 0:
        return [0]
    data_from_class = data_eg[:, -1]
    data_from_class = list(data_from_class)
    probability = [data_from_class.count(classes_num[a]) for a in range(len(classes_num))]
    return np.asarray(probability) / len(data_from_class)


def get_data(name_file):
    file_data = np.loadtxt(name_file, dtype=np.float32)
    return file_data


def calculate_attributes(data_eg, data_attributes, given_mode = "randomized"):
    highest_gain = attribute_fine = threshold_fine = -1
    if given_mode == 'optimized':
        for a in data_attributes:
            val_attr = data_eg[:, a]
            lowest = min(val_attr)
            highest = max(val_attr)
            for b in range(1, 51):
                calc_threshold = lowest + b * (highest - lowest) / 51
                calculate_gain = calculate_info_gain(data_eg, a, calc_threshold)
                if calculate_gain > highest_gain:
                    highest_gain = calculate_gain
                    attribute_fine = a
                    threshold_fine = calc_threshold
        return attribute_fine, threshold_fine, highest_gain
    else:
        a = random.randint(0, data_eg.shape[1] - 2)
        val_attr = data_eg[:, a]
        lowest = min(val_attr)
        highest = max(val_attr)
        for b in range(1, 51):
            calc_threshold = lowest + b * (highest - lowest) / 51
            calculate_gain = calculate_info_gain(data_eg, a, calc_threshold)
            if calculate_gain > highest_gain:
                highest_gain = calculate_gain
                attribute_fine = a
                threshold_fine = calc_threshold
    return attribute_fine, threshold_fine, highest_gain


def decision_tree_highest_level(data_eg, given_mode = "randomized"):
    attribute = [a for a in range(data_eg.shape[1] - 1)]
    value_default = calculate_distribution(data_eg)
    return decision_tree_level(data_eg, attribute, value_default, given_mode=given_mode)


def get_values_class(value_test, value_trees, value_tree_id):
    if value_trees.child_on_left is None and value_trees.child_on_right is None:
        dictionary = list(value_trees.calculate_distribution)
        value_class_predicted = dictionary.index(max(dictionary)) + min(classes_num)
        return value_class_predicted
    else:
        if value_test[value_trees.attribute_fine] < value_trees.threshold_fine or value_trees.child_on_right is None:
            return get_values_class(value_test, value_trees.child_on_left, value_tree_id)
        else:
            return get_values_class(value_test, value_trees.child_on_right, value_tree_id)


def print_forest( q = [], value_tree_id = 1 ):
    if q:
        value_trees = q.pop(0)
        print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (
            value_tree_id, value_trees.node_id, value_trees.attribute_fine, value_trees.threshold_fine,
            value_trees.calculate_gain))
        if value_trees.child_on_left is not None: q.append( value_trees.child_on_left )
        if value_trees.child_on_right is not None: q.append( value_trees.child_on_right )

        print_forest( q, value_tree_id = value_tree_id)


if len(sys.argv) != 5:  # main function start here
    print("Improper format!\n Try: decision_tree.py <training_file> <test_file> <option> <pruning_thr>")
    sys.exit(0)
training_data = get_data(sys.argv[1])
classes_num = np.unique(training_data[:, -1])
testing_data = get_data(sys.argv[2])
given_mode = sys.argv[3]
pruning_threshold = int(sys.argv[4])
# make_forest = [decision_tree_highest_level(training_data)]

make_forest = []
if given_mode == 'forest3':
    for a in range(
            3):
        make_forest.append(decision_tree_highest_level(training_data, given_mode=given_mode))
        print_forest(q=[make_forest[-1]], value_tree_id = a+1)
elif given_mode == 'forest15':
    for a in range(15):
        make_forest.append(decision_tree_highest_level(training_data, given_mode = given_mode ))
        print_forest(q=[make_forest[-1]], value_tree_id = a+1)
elif given_mode == "optimized" or given_mode == "randomized":
    make_forest.append(decision_tree_highest_level(training_data, given_mode=given_mode))
    print_forest(q=[make_forest[-1]], value_tree_id= 1)
else:
    print("Please provide the valid arguments as in the question.\n")
    sys.exit(0)
data_final = []
for b in range(len(make_forest)):
    class_accuracy = []
    value_classes_predicted = [get_values_class(a, make_forest[b], b + 1) for a in testing_data[:, :-1]]
    for a in range(len(testing_data)):
        if type(value_classes_predicted[a]) == np.float32 or type(value_classes_predicted[a]) == np.float64:
            if value_classes_predicted[a] == testing_data[a, -1]:
                class_accuracy.append(1)
            else:
                class_accuracy.append(0)
        else:
            if testing_data[a, -1] in value_classes_predicted[a]:
                class_accuracy.append(1 / len(value_classes_predicted[a]))
            else:
                class_accuracy.append(0)
        print("ID=%5d" % (a + 1) + ", predicted=%3d" % (value_classes_predicted[a]) + ", true=%3d" % (
            testing_data[a, -1]) + ", accuracy=%4.2f" % (class_accuracy[-1]))

    class_accuracy = np.asarray(class_accuracy)
    data_final.append(np.sum(class_accuracy) / len(class_accuracy))
    print('classification accuracy=%6.4f' % (np.sum(class_accuracy) / len(class_accuracy)))
