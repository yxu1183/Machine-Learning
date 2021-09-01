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
    # print(num)
    # print(lines)
    # print(avg)
    # print(stddev)


average, std = file_stats("numbers1.txt")
print(average, std)
