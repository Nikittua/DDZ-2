import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import random
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import poisson
from scipy.stats import triang
import scipy.stats as st
import math
import sys

# theta = 21.5
# sample_size = 100
# get_sample = lambda n: np.random.poisson(theta, n)
# x = np.arange(0, sample_size, 1)
# y = poisson.pmf(x, theta)
# plt.plot(x, y, label='population')
# sample = get_sample(sample_size)
# plt.hist(sample, density=True, label='sample')
# # plt.show()

# log_likelihood = lambda rate: sum([np.log(poisson.pmf(v, rate)) for v in sample])

# rates = np.arange(0, sample_size, 1)
# estimates = [log_likelihood(r) for r in rates]
# plt.xlabel('parameter')
# plt.plot(rates, estimates)
# plt.show()
# print('parameter value: ', rates[estimates.index(max(estimates))])

mu = 21.5
count = 100
r = poisson.rvs(mu, size=count)


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api
from scipy import stats
from scipy.optimize import minimize 

# generate an independent variable 
x = np.linspace(-10, 30, 100)
# generate a normally distributed residual
e = np.random.normal(10, 5, 100)
# generate ground truth
y = 10 + 4*x + e
df = pd.DataFrame({'x':x, 'y':y})
df.head()

sns.regplot(x='x', y='y', data = df)
# plt.show() 

features = api.add_constant(df.x)
model = api.OLS(y, features).fit()
model.summary() 
res = model.resid
standard_dev = np.std(res)

def MLE_Norm(parameters):
   const, beta, std_dev = parameters
   pred = const + beta*x
   LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
   neg_LL = -1*LL
   return neg_LL 
 # minimize arguments: function, intial_guess_of_parameters, method
mle_model = minimize(MLE_Norm, np.array([2,2,2]), method='L-BFGS-B')
print(mle_model) 