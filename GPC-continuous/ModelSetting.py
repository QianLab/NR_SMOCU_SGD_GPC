# -*- coding: utf-8 -*-
#%%
"""
Created on Fri Sep 18 00:03:35 2020

@author: Aggie


continuous code
"""

import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm
from InitialSetting import *
from scipy import special
from scipy import integrate
import math
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from svgd import *
import torch

A = np.polynomial.hermite.hermgauss(8)


def GroundTruthFunction(x):
    # x is a single point f_num
    py = GroundTruthProbability(x)
    py = py.reshape(-1)
    y = choice(range(c_num), p = py)
    return y

def XspaceGenerate_(x_num):
    # xspace = np.linspace(xinterval[0], xinterval[1], x_num)
    # xspace = xspace.reshape(-1, f_num) 
    # xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, f_num))
    if discrete_label:
        if x_num == len(xspace):
            return xspace
        sampleidx = choice(range(xspace.shape[0]), x_num, replace = False)#############################
        return xspace[sampleidx]
    xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, f_num))
    return xspace



def XspaceGenerateApprox_(x_num, x):###############How about high dimension
    
    d = kernel.lengthscale
    #xspace = np.random.multivariate_normal(mean = x, scale = d, x_num)
    if f_num > 1:
        cov = np.eye(f_num)*d
        mean = x.reshape(-1)
        xspace = np.random.multivariate_normal(mean = mean, cov = cov, size = x_num)
    else:
        xspace = np.random.normal(x, d, (x_num, 1))##this is only for single dimension
    # idxarray = np.all(xspace >= xinterval[0], axis=1) & np.all(xspace <= xinterval[1], axis = 1)
    # xspace = xspace[idxarray].reshape(-1, f_num)
    xspace = norm.rvs(size = (x_num, f_num), loc = x, scale = d)
    xspace, _ = Xtruncated(xinterval[0], xinterval[1], xspace)
    wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)
    wspace_log = np.sum(wspace_log_array, axis=1)+np.log(x_num/len(xspace))
    
    
    return xspace, wspace_log, px_log

def InitialDataGenerator(f, initial_num = 10): 
    X_ = XspaceGenerate_(initial_num)
    #X_ = np.random.uniform(xinterval[0], xinterval[1], (initial_num, f_num))
    Y_ = np.zeros((initial_num, 1))
    for i in range(initial_num):
        Y_[i] = f(X_[i:i+1])
    
    Xindex = None
    return X_, Y_, Xindex

try:
    BayesianError
except:
    def BayesianError(x_num):
        
        xspace = XspaceGenerate_(x_num)

        pymat = GroundTruthProbability(xspace)
        bayesian_error = np.amin(1-pymat, axis =1)
        return bayesian_error.mean()


def Xtruncated(xlower, xupper, xspace):
    idxarray = np.all(xspace >= xlower, axis=1) & np.all(xspace <= xupper, axis = 1)
    return xspace[idxarray].reshape(-1, f_num), idxarray



class ModelSet():
    def __init__(self,  X, Y, parameters = parameters_, hypernum = 10):
        #__slots__ = ['hypernum', 'f_num', 'X', 'Y', 'multi_hyper', 'hyperset']
        #self.parameterset = parameter# random generate parameterset
        
        self.f_num = f_num
        self.hypernum = hypernum
        self.X = X
        self.Y = Y
        self.multi_hyper = True
        #self.is_real_data = False

        #generate prior set
        varianceset = np.random.uniform(vinterval[0], vinterval[1], [hypernum, 1])####################
        lengthset = np.random.uniform(linterval[0], linterval[1], [hypernum, 1])################
        hyperset0 = np.concatenate((varianceset, lengthset), axis = 1)

        #posterior set
        self.hyperset = self.HyperParticle(hyperset0, n_iter = 1000)

        self.ModelSetGen(self.hyperset)


    def ModelSetGen(self, hyperset):
        self.modelset = []
        for m in range(self.hypernum):
            variance = hyperset[m, 0]
            lengthscale = hyperset[m, 1]
            parameters = parameters_
            parameters[2]=GPy.kern.RBF(f_num, variance=variance, lengthscale=lengthscale)
            self.modelset.append(Model(self.X, self.Y, parameters=parameters, optimize=False))

    def Update(self, x, y):
        x = x.reshape(-1, self.f_num)
        self.X = np.concatenate((self.X, x))
        self.Y = np.concatenate((self.Y, [[y]]), axis = 0)
        self.hyperset = self.HyperParticle(self.hyperset, n_iter = 100)
        self.ModelSetGen(self.hyperset)

    def dloglikelihood(self, theta_array):
        grad_array = np.zeros(theta_array.shape)
        for i, theta in enumerate(theta_array):
            var = theta[0]
            length = theta[1]
            if var < vinterval[0] or var > vinterval[1] or length < linterval[0] or length > linterval[1]:
                if var < vinterval[0]:
                    grad_array[i, 0] = 10
                if var > vinterval[1]:
                    grad_array[i, 0] = -10
                if length < linterval[0]:
                    grad_array[i, 1] = 10
                if length > linterval[1]:
                    grad_array[i, 1] = -10
            else:
                kernel = GPy.kern.RBF(f_num, variance = var, lengthscale = length)
                m = GPy.core.GP(X=self.X,
                                Y=self.Y,
                                kernel=kernel,
                                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                                likelihood=lik)
                grad_array[i]=-m.objective_function_gradients()[1]

        return grad_array

    def HyperParticle(self, initial_array, n_iter = 1000):
        #array size: particle_num * parameter_num
        updated_array = SVGD().update(initial_array, self.dloglikelihood, n_iter=n_iter, stepsize=0.05)
        return updated_array



    def ObcClassifierError(self, x_num):
        xspace = self.XspaceGenerate(x_num)
        p = 1.0/self.hypernum
        pyTheta = np.zeros((x_num, c_num))
        for model in self.modelset:
            pyTheta += model.predict_proba(xspace)*p
        yhat = np.argmax(pyTheta, axis = 1)
        py = GroundTruthProbability(xspace)
        classifier_error = 1-py[np.arange(x_num), yhat]
        return classifier_error.mean()

    def XspaceGenerate(self, x_num):
        xspace = XspaceGenerate_(x_num)
        return xspace


        


class Model():

    def __init__(self,  X, Y, parameters = parameters_, optimize = False):#####################################################
        #__slots__ = ['parameters', 'f_num', 'gpc', 'xinterval', 'optimize', 'lik', 'c_num']
        # X size is x_num*fnum
        # Y size is x_num
        self.parameters = parameters
        self.f_num = f_num
        self.c_num = parameters[1]
        kernel = parameters[2]
        #lik = parameters[3]
        #self.gpc = GaussianProcessClassifier(kernel = kernel, optimizer=None).fit(X, Y)
        
        m = GPy.core.GP(X=X,
                        Y=Y, 
                        kernel=kernel, 
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood=lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts = 40, max_iters=2000, verbose=False)
        if m.kern.lengthscale.item()>4:
            kernel = GPy.kern.RBF(f_num, variance = m.kern.variance.item(), lengthscale = 4)
            m = GPy.core.GP(X = X,
                        Y = Y,
                        kernel = kernel,
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood = lik)
        self.gpc = m
        #self.c_num = c_num
        self.xinterval = xinterval
        self.optimize = optimize
        self.lik = lik
    
    def predict_proba(self, x):
        x = x.reshape(-1, self.f_num)
        M = len(x)//1000
        if M <= 1:
            py_1 = self.gpc.predict(x)[0]
            py_0 = 1 - py_1
            pymat = np.concatenate((py_0, py_1), axis = 1) #pymat size x_num*cnum
            return pymat
        else:
            pymat1 = np.zeros((len(x), 2))
            for m in range(M):
                idx = range(m*1000, m*1000+1000)
                pymat1[idx, 1:2] = self.gpc.predict(x[idx, :])[0]
                pymat1[idx, 0] = 1-pymat1[idx, 1]
            idx = range(m*1000+1000, len(x))
            pymat1[idx, 1:2] = self.gpc.predict(x[idx, :])[0]
            pymat1[idx, 0] = 1-pymat1[idx, 1]

            return pymat1

    def _noiseless_predict_torch(self, xt):
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)
        X_ = torch.tensor(self.gpc.X)
        K = self.K
        
        mu_t = K(xt, X_)@woodbury_vector
        sigma_tt = K(xt, xt) - K(xt, X_)@woodbury_inv@K(X_, xt)

        return mu_t, sigma_tt
        
    def K(self, xt, xs):

        kern = self.gpc.kern
        assert (kern.name == 'rbf') #the function is only coded for rbf kernel
            
        l1 = kern.lengthscale.item()
        l2 = kern.variance.item()
        Kts = l2*torch.exp(-torch.cdist(xt, xs)**2/2/l1**2)
        return Kts
    
    def predict_proba_torch(self, xt):
        xt = xt.reshape(-1, self.f_num)
        assert(type(xt)==torch.Tensor)
        
        Phi = lambda x : 0.5*(torch.erf(x/math.sqrt(2))+1) 
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)

        ft_hat = mu_t/torch.sqrt(sigma_tt+1)
        py_1 = Phi(ft_hat)
        py_0 = 1 - py_1
        pymat = torch.cat((py_0, py_1), axis = 1)
        return pymat

    def _calculate_mean_and_variance(self, xt, xs):
        xt = xt.reshape(-1, self.f_num)
        xs = xs.reshape(-1, self.f_num)
        muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False) 
        mu = muvar[0].reshape(-1)
        mu_s = mu[0:-1]
        mu_t = mu[-1]

        var = muvar[1].reshape(-1)
        sigma_ss = var[0:-1]
        X_ = self.gpc.X
        sigma_st =  self.gpc.kern.K(xs, xt) - self.gpc.kern.K(xs,X_)@self.gpc.posterior.woodbury_inv@self.gpc.kern.K(X_, xt)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt = var[-1]
        sigma_tt_hat = sigma_tt - sigma_st**2/sigma_ss
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_mean_and_variance_torch(self, x1, x2):
        xt = x1.reshape(-1, self.f_num)
        xs = torch.tensor(x2.reshape(-1, self.f_num))

        X_ = torch.tensor(self.gpc.X)
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)

        muvar = self.gpc.predict_noiseless(x2, full_cov=False) 
        mu_s = torch.tensor(muvar[0])
        sigma_ss = torch.tensor(muvar[1])

        K = self.K
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)
        sigma_st = K(xs, xt) - K(xs, X_)@woodbury_inv@K(X_, xt)
        sigma_tt_hat = sigma_tt - sigma_st**2/sigma_ss

        # muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False) 
        # mu = muvar[0].reshape(-1)
        # mu_s = mu[0:-1]
        # mu_t = mu[-1]

        # var = muvar[1].reshape(-1)
        # sigma_ss = var[0:-1]
        # X_ = self.gpc.X
        # sigma_st =  self.gpc.kern.K(xs, xt) - self.gpc.kern.K(xs,X_)@self.gpc.posterior.woodbury_inv@self.gpc.kern.K(X_, xt)
        # sigma_st = sigma_st.reshape(-1)
        # sigma_tt = var[-1]
        # sigma_tt_hat = sigma_tt - sigma_st**2/sigma_ss
        mu_s = mu_s.reshape(-1)
        mu_t = mu_t.reshape(-1)
        sigma_ss = sigma_ss.reshape(-1)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt_hat = sigma_tt_hat.reshape(-1)
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat
    
    def _calculate_posterior_predictive_from_joint_distribution(self, xt, xs, pt1s1, version = 'numpy'):
        if version == 'pytorch':
            pt = self.predict_proba_torch(xt)
        else:
            pt = self.predict_proba(xt)
        
        assert(pt.shape == (1, 2))
        pt0, pt1 = pt[0, 0], pt[0, 1]
        ps = self.predict_proba(xs)
        ps0, ps1 = ps[:, 0], ps[:, 1]
        if version == 'pytorch':
            ps1 = torch.tensor(ps1)
        pt0s1 = ps1 - pt1s1
        ps1_t1 = pt1s1/pt1
        ps0_t1 = 1-ps1_t1
        ps1_t0 = pt0s1/pt0
        ps0_t0 = 1-ps1_t0

        if version == 'pytorch':
            column_stack = torch.column_stack
        else:
            column_stack = np.column_stack
        ps_t0 = column_stack((ps0_t0, ps1_t0))
        ps_t1 = column_stack((ps0_t1, ps1_t1))

        return ps_t0, ps_t1

    
    def OneStepPredict(self, xt, xs, version = 'numpy'):
        #xt is xstar, xt is tensor for version = 'pytorch'
        #xs is an array of size x_num*f_num

        if version == 'pytorch':
            calculate_mean_variance = self._calculate_mean_and_variance_torch
            erf = torch.erf
            sqrt = torch.sqrt
            zeros = torch.zeros
        else:
            calculate_mean_variance = self._calculate_mean_and_variance
            erf = special.erf
            sqrt = np.sqrt
            zeros = np.zeros

        mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat = calculate_mean_variance(xt, xs)
        sigma_s = np.sqrt(sigma_ss)
        Phi = lambda x : 0.5*(erf(x/math.sqrt(2))+1) 
        
        def func4(f0):
            #This function use hermite Gaussian quadrature, 
            # return: a x_num array of value with index corresponding to xs. 
            # term3 = 1/(sigma1*math.sqrt(2*math.pi))*math.exp(-0.5*(x3)**2) is normalized as Gaussian function
            
            fs = f0*math.sqrt(2)*sigma_s+mu_s
            mu_t_hat = mu_t + sigma_st/sigma_ss*(fs-mu_s)
            ft_hat = mu_t_hat/sqrt(sigma_tt_hat+1)
            term1 = Phi(ft_hat)
            term2 = Phi(fs)
            return term1*term2/math.sqrt(math.pi)#math.sqrt(math.pi) is the constant for normalized Gaussian

        # joint distribution
        pt1s1 = zeros(len(xs))
        for i, f0 in enumerate(A[0]):
            pt1s1 += func4(f0)*A[1][i]

        ps_t0, ps_t1 = self._calculate_posterior_predictive_from_joint_distribution(xt, xs, pt1s1, version = version)
        return ps_t0, ps_t1


    def DataApprox(self, x):
        anum = 3
        X = self.gpc.X
        Y = self.gpc.Y
        kernel = self.parameters[2]
        d = kernel.lengthscale
        l = anum*d
        xlower = np.maximum(self.xinterval[0], x-l)
        xupper = np.minimum(self.xinterval[1], x+l)
        # idxarray = np.all(X >= xlower, axis=1) & np.all(X <= xupper, axis=1)
        
        X, idxarray = Xtruncated(xlower, xupper, X)

        #X = X[idxarray].reshape(-1, self.f_num)
        Y = Y[idxarray].reshape(-1, 1)

        return X, Y


    def UpdateNew(self, x, y):##############################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x), axis = 0)
        Y = np.concatenate((self.gpc.Y, [[y]]), axis = 0)
        # parameters = self.parameters
        # parameters[2] = self.gpc.kern
        # model2 = Model(X, Y, parameters, optimize = False)
        model2 = self.ModelTrain(X, Y)
        return model2

    def ModelTrain(self, X, Y):
        parameters = self.parameters
        parameters[2] = self.gpc.kern
        model2 = Model(X, Y, parameters, optimize = False)
        return model2
    
    
    def Update(self, x, y, optimize = False):########################################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x))
        Y = np.concatenate((self.gpc.Y, [[y]]), axis = 0)

        #lik = self.parameters[3]
        m = GPy.core.GP(X = X,
                        Y = Y,
                        kernel = self.gpc.kern,
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood = lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts = 40, max_iters=200, verbose = False)
        # else:
        #     pass
        if m.kern.lengthscale.item()>4:
            kernel = GPy.kern.RBF(f_num, variance = m.kern.variance.item(), lengthscale = 4)
            m = GPy.core.GP(X = X,
                        Y = Y,
                        kernel = kernel,
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood = lik)
        self.gpc = m

    def XspaceGenerate(self, x_num):
        xspace = XspaceGenerate_(x_num)
        return xspace

    def ObcClassifierError(self, x_num):
        xspace = self.XspaceGenerate(x_num)
        pyTheta = self.predict_proba(xspace)
        yhat = np.argmax(pyTheta, axis = 1)
        py  = GroundTruthProbability(xspace)
        classifier_error = 1-py[np.arange(x_num), yhat]
        # classifier_error = 0
        # # pyTheta = self.predict_proba(xspace)
        # # yhat = np.argmax(pyTheta, axis = 0)
        # # yhat = yhat.astype(int)
        # # pyhat_r = 
        # pyTheta = self.predict_proba(xspace)
        # yhat = np.argmax(pyTheta, axis = 1)

        # for i, _ in enumerate(xspace):
        #     x = xspace[i:i+1]# all the inputs should take 2d array 
        #     pyTheta = self.predict_proba(x)
        #     yhat = np.argmax(pyTheta)######################
        #     yhat = yhat.astype(int)
        #     py  = GroundTruthProbability(x)
        #     classifier_error += (1 - py[yhat])/x_num
        return classifier_error.mean()

    
    # def ModelDraw(self, name):
    #     print(self.gpc.kern)
    #     if f_num >2:
    #         self.gpc.plot(visible_dims = [0])
    #         self.gpc.plot_f(visible_dims = [0])
    #     else:
    #         self.gpc.plot()
    #         plt.savefig('c_'+name)
            
    #         if f_num == 1:
    #             xspace = self.XspaceGenerate(1000)
    #             pspace = np.zeros(len(xspace))
    #             for i, x in enumerate(xspace):
    #                 pspace[i] = GroundTruthProbability(x)[1]
    #             plt.plot(xspace, pspace, 'ro')
    #         #self.gpc.plot_f()
        
        
        #plt.plot(xspace, fspace, 'bo')
            #plt.savefig('f_'+name)

    def XspaceGenerateApprox(self, x_num, x):
        # xspace = XspaceGenerateApprox_(x_num, x)
        # return xspace


        d = self.gpc.kern.lengthscale.item()
        #xspace = np.random.multivariate_normal(mean = x, scale = d, x_num)
        if self.f_num > 1:
            cov = np.eye(self.f_num)*d
            mean = x.reshape(-1)
            xspace = np.random.multivariate_normal(mean = mean, cov = cov, size = x_num)
        else:
            xspace = np.random.normal(x, d, (x_num, 1))##this is only for single dimension
        # idxarray = np.all(xspace >= xinterval[0], axis=1) & np.all(xspace <= xinterval[1], axis = 1)
        # xspace = xspace[idxarray].reshape(-1, f_num)
        xspace = norm.rvs(size = (x_num, f_num), loc = x, scale = d)
        xspace, _ = Xtruncated(xinterval[0], xinterval[1], xspace)
        wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)
        wspace_log = np.sum(wspace_log_array, axis=1)+np.log(x_num/len(xspace))
        
        
        # if f_num>1:
        #     wspace = multivariate_normal.pdf(xspace, mean = mean, cov = cov )
        # else:
        #     wspace = norm.pdf(xspace, loc=x, scale=d)
        #xspace = np.random.uniform(max(xinterval[0], x-3*d), min(xinterval[1], x+3*d), (x_num, f_num))

        return xspace, wspace_log, px_log

# %%
