import math


class ExpectationMaximisation:

    def __init__(self, first_gaussian, second_gaussian, data_points, mix=0.5):
        self.data = data_points
        self.one = first_gaussian
        self.two = second_gaussian
        self.mix = mix
        self.log_likelihood = 0.0

    def fit(self, n=1):
        for i in range(0, n):
            weights = self.__estimation_step()
            self.__maximisation_step(weights)

    def __estimation_step(self):

        for data_point in self.data:
            prob_from_first = self.one.prob(data_point) * self.mix
            prob_from_second = self.two.prob(data_point) * (1 - self.mix)

            # Normalise the probabilities
            total = prob_from_first + prob_from_second
            prob_from_first = prob_from_first / total
            prob_from_second = prob_from_second / total

            self.log_likelihood = self.log_likelihood + math.log(total)
            yield (prob_from_first, prob_from_second)

    def __maximisation_step(self, weights):

        (first_weights, second_weights) = zip(*weights)
        sum_first_weights = sum(first_weights)
        sum_second_weights = sum(second_weights)

        # Update means
        self.one.mean = sum(
            weight * data_point for (weight, data_point) in zip(first_weights, self.data)) / sum_first_weights
        self.two.mean = sum(
            weight * data_point for (weight, data_point) in zip(second_weights, self.data)) / sum_second_weights

        # Update standard deviations
        self.one.std = math.sqrt(sum(weight * ((data_point - self.one.mean) ** 2)
                                     for (weight, data_point) in zip(first_weights, self.data)) / sum_first_weights)

        self.two.std = math.sqrt(sum(weight * ((data_point - self.two.mean) ** 2)
                                     for (weight, data_point) in zip(second_weights, self.data)) / sum_second_weights)
        # Update mix
        self.mix = sum_first_weights / len(self.data)

    def prob(self, data_point):
        return self.mix * self.one.prob(data_point) + (1 - self.mix) * self.two.prob(data_point)

    def predict(self, data):
        return [self.prob(data_point) for data_point in data]
