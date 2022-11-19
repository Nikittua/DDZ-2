import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import random
from statsmodels.distributions.empirical_distribution import ECDF

from scipy.stats import poisson








mu = 21.5
count = 1000
r = poisson.rvs(mu, size=count)
# plt.hist(r, bins=np.linspace(0, 35, 36), alpha=0.5, label='counting process', ec='black', align='left')
# plt.plot(poisson.pmf(np.linspace(0, 35, 36),mu)*count)
# plt.legend()
# plt.show()

print(r)

# Эмпирическая функция распределения
cdf = ECDF(r)
plt.plot(cdf.x, cdf.y, label="statmodels")
plt.legend()
plt.show()

# x = np.sort(r)
# y = np.arange(len(x))/float(len(x))
# plt.plot(x, y)
# plt.show()