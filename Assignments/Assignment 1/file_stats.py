# Name: Yunika Upadhayaya
# ID: 1001631183
# CSE 4309 - Assignment 1 (Task 9)

import math


def file_stats(pathname):
    with open(pathname) as file:
        lines = []
        for each_line in file:
            lines.append(float(each_line.rstrip()))
    num = len(lines)
    avg = sum(lines) / num
    deviations = [(data - avg) ** 2 for data in lines]
    variance = sum(deviations)/(num-1)
    stddev = math.sqrt(variance)
    return avg, stddev

#for testing:

average, std = file_stats("numbers1.txt")
print(average, std)
