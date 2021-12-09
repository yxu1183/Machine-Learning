# Name: Yunika Upadhayaya
# ID: 1001631183
# Bayesian Estimation - Honors Project

import sys
import numpy as np


def read_file(fileName):
    with open(fileName, 'r') as file:
        data_file = file.read()
    return data_file


class Bayes_Estimation:
    def __init__(self, m_prob, m_given):
        self.m_probability = m_prob
        self.given_m = m_given

    def prob_a(self):
        return sum(self.m_probability * self.given_m)

    def prob_b(self):
        return 1 - self.prob_a()

    def prob_c(self, given):
        if given == 'a':
            return self.prob_a()
        elif given == 'b':
            return self.prob_b()
        else:
            return None

    def prob_a_gm(self, initial_m):
        return self.prob_m(initial_m)

    def prob_b_gm(self, initial_m):
        return 1 - self.prob_a_gm(initial_m)

    def prob_c_gm(self, given, initial_m):
        if given == 'a':
            return self.prob_a_gm(initial_m)
        elif given == 'b':
            return self.prob_b_gm(initial_m)
        else:
            return None

    def prob_m(self, initial_m):
        return self.given_m[self.given_m == initial_m][0]

    def prob_p_m(self, initial_m):
        return self.m_probability[self.given_m == initial_m][0]

    def p_m_prob(self, given):
        final_val = self.prob_c(given)
        for initial_m in self.given_m:
            val_a = self.prob_c_gm(given, initial_m) * self.prob_p_m(initial_m) / final_val
            self.m_probability[self.given_m == initial_m] = val_a

    def print_post_prob(self):
        for initial_m in self.given_m:
            print("p(m = %.1f | data) = %.4f" % (initial_m, self.prob_p_m(initial_m)))
        print("p(c = 'a' | data) = %.4f" % (self.prob_a()))


if __name__ == "__main__":
    if len(sys.argv) != 2:  # main function start here
        print("Improper format!\n Try: bayes_estimation.py <text_file>")
        sys.exit(0)
    else:
        file_name = sys.argv[1]
        file_data = read_file(file_name)
        m_probability = np.array([0.9, 0.04, 0.03, 0.02, 0.01])
        given_m = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        calculate_bias = Bayes_Estimation(m_probability, given_m)
        for data_i in file_data:
            calculate_bias.p_m_prob(data_i)
        calculate_bias.print_post_prob()
