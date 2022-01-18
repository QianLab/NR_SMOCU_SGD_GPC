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

f_num = 2 #feature number          #only change here for high-dimension
c_num = 2
kernel = GPy.kern.RBF(f_num, variance = 1, lengthscale = 0.4)
lik = GPy.likelihoods.Bernoulli()
parameters_ =  [f_num, c_num, kernel, lik]

xinterval = [np.array(0), np.array(4)]
linterval = [0.1, 4]#length scale
vinterval = [50, 200]# sigma^2 scale
px = 1/(xinterval[1] - xinterval[0])**f_num
px_log = -f_num*np.log(xinterval[1] - xinterval[0])

discrete_label = False
optimize_label = True

Perror = 0.2


def SetGlobalNumber(x_num = 1000): #############################################################################
    return
    


def GroundTruthProbability(x):########################################################################
# x can be single point and xspace

    x = x.reshape(-1, f_num)

    if len(x) == 1:
        x1 = int(x[0, 0])
        x2 = int(x[0, 1])
        py_1 = np.array([[(x1+x2)%2*(1-Perror)+(1-(x1+x2)%2)*Perror
        ]])
    else:
        x1 = x[:, 0:1].astype(int)
        x2 = x[:, 1:2].astype(int)
        py_1 = (x1+x2)%2*(1-Perror)+(1-(x1+x2)%2)*Perror#py_1 should be size xnum*1


    #py_1 = lik.gp_link.transf(mgpr.predict(x)[0])
    # gpm.predict(x)
    
    # j = np.where(np.all(np.isclose(xspace, x), axis = 1))[0][0]
    # py_1 = pspace[j]
    py_0 = 1 - py_1
    #py = np.array([py_0, py_1]) #pymat size x_num*cnum
    py = np.concatenate((py_0, py_1), axis = 1)
    return py

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
    model.gpc.plot()
    plt.savefig('c_'+name)

if __name__ == "__main__":
    xspace = np.random.uniform(xinterval[0], xinterval[1], (1, f_num))
    print(GroundTruthProbability(xspace))