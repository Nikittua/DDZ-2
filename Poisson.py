import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import random
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import poisson
import scipy.stats as st
import math
import sys
# plt.hist(r, bins=np.linspace(0, 35, 36), alpha=0.5, label='counting process', ec='black', align='left')
# plt.plot(poisson.pmf(np.linspace(0, 35, 36),mu)*count)
# plt.legend()
# plt.show()
# x = np.sort(r)
# y = np.arange(len(x))/float(len(x))
# plt.plot(x, y)
# plt.show()


#2.1
mu = 21.5
count = 100
r = poisson.rvs(mu, size=count)
#print(r)

# 2.2
cdf = ECDF(r)
plt.plot(cdf.x, cdf.y, label="statmodels")
plt.legend()
plt.show()



#3.1 Метод моментов

#general formula for the nth sample moment   
def sample_moment(sample, n):       
  summed = np.sum([el**n for el in sample])       
  length = len(sample)       
  return 1/length * summed    
#function to estimate parameters k and theta   
def estimate_pars(sample):       
  m1 = sample_moment(sample, 1)       
  m2 = sample_moment(sample, 2)       
  k = m1**2/(m2 - m1**2)       
  theta = m2/m1 - m1       
  return k, theta

k_hat, theta_hat = estimate_pars(r) 
#print(k_hat, theta_hat)

#3.1 Метод макс правдоподобия

# theta_A = r / np.sum(r)
# def mle(R, theta):
#   return (math.factorial(np.sum(R)) // (np.prod([math.factorial(r) for r in R]))) * \
#          np.prod([theta[i]**R[i] for i in range(len(R))])

# ml_A = mle(r ,theta_A)
# print(ml_A)