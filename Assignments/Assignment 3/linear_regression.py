# Name: Yunika Upadhayaya
# Student ID: 1001631183

import sys
from numpy import genfromtxt
import numpy as np


def get_x(a, degree):
    return_value = [1]
    for num in a:
        for count in range(1, degree + 1):
            first_float = float(num)
            second_float = float(count)
            return_value.append(first_float ** second_float)
    return_value = np.array(return_value)
    return return_value


def training_phase(training_file, degree, lambda_num):
    training_data = genfromtxt(training_file)
    a = []
    b = []
    for num in training_data:
        a.append(get_x(num[:-1], degree))
        b.append(num[-1])
    a = np.array(a)
    b = np.array([b]).T
    y = np.linalg.pinv(lambda_num * np.identity(len(a[0])) + a.T @ a) @ a.T @ b
    return y


def test_phase(test_file, degree, y):
    test_data = genfromtxt(test_file)
    a = []
    b = []

    for count_ind, i in enumerate(test_data):
        a.append(get_x(i[:-1], degree))
        b.append(i[-1])
        target = y. T @ a[count_ind]
        error_sqaured = (b[count_ind] - target) ** 2
        print("ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f" % (count_ind + 1, target, b[count_ind], error_sqaured))


def linear_regression(training_file, test_file, degree, lambda_num):
    if degree <= 0 or degree >= 11:
        print("Invalid Degree! Degree can only between 1 and 10.")
    if lambda_num < 0:
        print("Invalid lambda! Lambda cannot be less than 0.")
    y = training_phase(training_file, degree, lambda_num)
    for count_ind, i in enumerate(y):
        print("w%d=%.4f" % (count_ind, i))
    test_phase(test_file, degree, y)


if len(sys.argv) != 5:
    print("Improper format!\n Try: linear_regression.py <training_file> <test_file> <degree> <lambda>")
linear_regression(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
