import math
import numpy as np


class Gaussian:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def generate_points(self, count):
        return np.random.normal(self.mean, self.std, count)

    def prob(self, data_point):
        x = (data_point - self.mean) / abs(self.std)
        prob = 1 / (math.sqrt(2 * math.pi) * abs(self.std)) * math.exp(-x * x / 2)
        return prob

    def predict(self, data):
        return [self.prob(data_point) for data_point in data]

    def __str__(self):
        return f'Gaussian ~ N({self.mean}, {self.std})'
