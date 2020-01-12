import matplotlib.pyplot as plt
import numpy as np

from expectation_maximisation import ExpectationMaximisation
from gaussian import Gaussian

if __name__ == '__main__':
    seed = 113377
    np.random.seed(seed)

    first_dist = Gaussian(1, 2)
    second_dist = Gaussian(8, 2)

    print(first_dist)
    print(second_dist)

    # Full dataset
    data = np.append(first_dist.generate_points(1000), second_dist.generate_points(1000))
    x_pos = np.linspace(min(data), max(data), 2000)
    combined_gaussian = Gaussian(np.mean(data), np.std(data))
    plt.hist(data, bins=20, density=True)

    plt.plot(x_pos, combined_gaussian.predict(x_pos), label="Gaussian")
    plt.legend()
    plt.show()

    print('Running EM...')
    em = ExpectationMaximisation(first_dist, second_dist, data)
    em.fit(20)
    print('Done!')

    plt.hist(data, bins=20, density=True)
    plt.plot(x_pos, em.predict(x_pos), label='EM fit')
    plt.legend()
    plt.show()

# Reference: https://www.kaggle.com/charel/learn-by-example-expectation-maximization
