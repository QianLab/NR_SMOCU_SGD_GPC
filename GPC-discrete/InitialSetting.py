# -*- coding: utf-8 -*-
#%%
"""
Created on Fri Sep 18 00:03:35 2020

@author: Aggie


Build a discrete space with Gaussian process with continuous parameter
GPC6
"""

import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
import pandas as pd
import string

#generate data space
df = pd.read_csv('wine.data.csv', header = None)
temparray = choice(np.array([1, 2]), 2, replace = False)
temparray = [1, 2]
print(temparray)
df2 = df[df[0].isin(temparray)]
#df2 = df[~df[6].isin(['?'])]
xall2 = df2.iloc[:, 1:14].to_numpy().astype(float)
xall = (xall2 - xall2.mean(axis = 0))/xall2.std(axis = 0)
yall = (df2.iloc[:, 0].to_numpy() == temparray[0]).astype(int)


xspace, xtest, yspace, ytest = train_test_split(xall, yall, test_size = 0.5)


f_num = xspace.shape[1]
c_num = 2
kernel = GPy.kern.RBF(f_num, variance = 1, lengthscale = 0.4)
lik = GPy.likelihoods.Bernoulli()
parameters_ =  [f_num, c_num, kernel, lik]

xinterval = [np.array(0), np.array(4)]
linterval = [0.1, 4]#length scale
vinterval = [50, 200]# sigma^2 scale
px = 1/(xinterval[1] - xinterval[0])**f_num
px_log = -f_num*np.log(xinterval[1] - xinterval[0])


smocu_x_num = len(xspace)
error_x_num = len(xspace)
information_sum_num = 10000
mc_search_num = len(xspace)


discrete_label = True
optimize_label = False
is_real_data = True



Perror = 0

def SetGlobalNumber(x_num = 1000): #############################################################################
    print('yes')
    return
    


#def GroundTruthProbability(x):########################################################################
# x can be single point and xspace


def BayesianError(x_num):
    # x = x.reshape(-1, self.f_num)
    #     py_1 = self.gpc.predict(x)[0]
    #     py_0 = 1 - py_1
    #     pymat = np.concatenate((py_0, py_1), axis = 1) #pymat size x_num*cnum

    return Perror

def ModelDraw(model, name):
    #this is for f_num == 2
    pass
    print(model.gpc.kern)
    #model.gpc.plot()
    #plt.savefig('c_'+name)

if __name__ == "__main__":
    xspace = np.random.uniform(xinterval[0], xinterval[1], (1, f_num))
    print(GroundTruthProbability(xspace))