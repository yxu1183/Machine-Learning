# Name: Yunika Upadhayaya
# ID: 1001631183
# Assignment 7 - Value Iteration Algorithm

import sys
import numpy as np
import csv

moves = {"Left": "<",
         "Right": ">",
         "Up": "^",
         "Down": "v", }


def read_data(file_name):
    data_file = []
    with open(file_name) as file:
        read = csv.reader(file)
        for each_row in read:
            data_file.append(each_row)
    return data_file


def value_iteration():
    U = len(all_data), len(all_data[0])
    policy_sym = np.chararray(U, unicode=True)
    u_prime = np.zeros(U)
    for count in range(k_value):
        Y = u_prime.copy()
        for x in range(U[0]):
            for y in range(U[1]):
                if all_data[x][y] == 'X':
                    u_prime[x][y] = 0
                    policy_sym[x][y] = 'x'
                elif all_data[x][y] != '.':
                    u_prime[x][y] = float(all_data[x][y])
                    policy_sym[x][y] = 'o'
                else:
                    value_max = 0
                    for course_action in moves:
                        check_val = max_val((x, y), Y, U, course_action)
                        value_max = max(check_val, value_max)
                        if check_val == value_max:
                            policy_sym[x][y] = moves[course_action]
                    u_prime[x][y] = function((x, y)) + gamma_value * value_max
    return Y, policy_sym


def function(check_state):
    check_val = all_data[check_state[0]][check_state[1]]
    if check_val == 'X':
        return 0
    elif check_val == '.':
        return non_terminal_value
    else:
        return float(check_val)


def check_valid(a):
    if 0 <= a[0] < len(all_data):
        if 0 <= a[1] < len(all_data[1]):
            if all_data[a[0]][a[1]] == 'X':
                return False
            return True
    return False


def check_trans(s_prime, a, b):
    value = 0

    if b == "Left":
        move_state = (a[0], a[1] - 1)
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.8

        move_state = (a[0] - 1, a[1])
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

        move_state = (a[0] + 1, a[1])
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

    elif b == "Right":
        move_state = (a[0], a[1] + 1)
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.8

        move_state = (a[0] - 1, a[1])
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

        move_state = (a[0] + 1, a[1])
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

    elif b == "Up":
        move_state = (a[0] - 1, a[1])
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.8

        move_state = (a[0], a[1] - 1)
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

        move_state = (a[0], a[1] + 1)
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

    elif b == "Down":
        move_state = (a[0] + 1, a[1])
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.8

        move_state = (a[0], a[1] - 1)
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

        move_state = (a[0], a[1] + 1)
        if not check_valid(move_state):
            move_state = a
        if move_state == s_prime:
            value += 0.1

    return value


def max_val(check_state, Y, U, course_action):
    check_val = 0
    for x_value in range(U[0]):
        for y_value in range(U[1]):
            check_val += (check_trans((x_value, y_value), check_state, course_action) * Y[x_value][y_value])
    return check_val


def print_fun(utilities_val, policy_val):
    print("\nutilities:")
    for x in range(len(utilities_val)):
        for y in range(len(utilities_val[0])):
            print("{:6.3f}".format(utilities_val[x][y]), end=' ')
        print()

    print("\npolicy:")
    for x in range(len(policy_val)):
        for y in range(len(policy_val[0])):
            print("{:6s}".format(policy_val[x][y]), end=' ')
        print()


# main function stars here
if len(sys.argv) != 5:
    print("Improper format!\n Try: value_iteration.py <environmental_file> <non_terminal_reward> <gamma> <K>")
    exit(0)
else:
    all_data = read_data(sys.argv[1])
    non_terminal_value = float(sys.argv[2])
    gamma_value = float(sys.argv[3])
    k_value = int(sys.argv[4])
    utilities, policy = value_iteration()
    print_fun(utilities, policy)
