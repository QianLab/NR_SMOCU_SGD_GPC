# -*- coding: utf-8 -*-
#%%
"""
Created on Fri Sep 18 00:03:35 2020

@author: Aggie


Build a discrete space with Gaussian process with continuous parameter
GPC5
"""

import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm
from InitialSetting import *
from scipy.special import erf
from scipy import integrate
import math
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from svgd import *
A = np.polynomial.hermite.hermgauss(8)

def ReturnXspace():
    return len(xspace)

def SetGlobalNumber(x_num = 1000): #############################################################################
    global xspace, xtest, yspace, ytest
    xspace, xtest, yspace, ytest = train_test_split(xall, yall, test_size = len(xtest))
    #xspace = xspace+1
    print('yes')
    return


def GroundTruthFunction(x):
    # x is a single point f_num
    j = np.where(np.all(np.isclose(xspace, x), axis = 1))[0][0]
    y = yspace[j]
    return y

def XspaceGenerate_(x_num):
    # xspace = np.linspace(xinterval[0], xinterval[1], x_num)
    # xspace = xspace.reshape(-1, f_num) 
    # xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, f_num))
    if discrete_label:
        global xspace
        if len(xspace) <= x_num:
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
    num1 = initial_num//2
    num2 = initial_num - num1
    Index1 = choice(np.where(yspace == 0)[0], num1, replace = False)
    Index2 = choice(np.where(yspace == 1)[0], num2, replace = False)
    Index = np.concatenate((Index1, Index2))
    X_ = xspace[Index]
    Y_ = yspace[Index].reshape(-1, 1)
    #Index = Index.tolist()
    # X_ = XspaceGenerate_(initial_num)
    # #X_ = np.random.uniform(xinterval[0], xinterval[1], (initial_num, f_num))
    # Y_ = np.zeros((initial_num, 1)).astype(int)
    # Xindex = []
    # for i in range(initial_num):
    #     Y_[i] = f(X_[i:i+1])
    #     Xindex.append(np.where(np.all(np.isclose(xspace, X_[i:i+1]), axis = 1))[0][0])
    return X_, Y_, Index


def BayesianError(x_num):
    return 0


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
        self.gpc = m
        #self.c_num = c_num
        self.xinterval = xinterval
        self.optimize = optimize
        self.lik = lik
        self.is_real_data = is_real_data
        self.dataidx = []
    
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

    def OneStepPredict0(self, sx, tx):
        sx = sx.reshape(-1, self.f_num)
        tx = tx.reshape(-1, self.f_num)
        mucov = self.gpc.predict_noiseless(np.concatenate((sx, tx)), full_cov=True)
        mu = mucov[0].reshape(-1)
        cov = mucov[1]
        sigma11 = cov[0, 0]
        sigma12 = cov[0, 1]
        sigma22 = cov[1, 1]
        mu1 = mu[0]
        mu2 = mu[1]
        def func3(f1):
            mu2hat = mu2 + sigma12/sigma11*(f1-mu1)
            sigma22hat = sigma22 - sigma12**2/sigma11
            x1 = mu2hat/math.sqrt(sigma22hat+1)
            term1 = 0.5*(math.erf(x1/math.sqrt(2))+1) #norm.cdf(mu1hat/(sigma11hat+1))
            term2 = 0.5*(math.erf(f1/math.sqrt(2))+1)
            sigma1 = math.sqrt(sigma11)
            x3 = (f1-mu1)/sigma1
            term3 = 1/(sigma1*math.sqrt(2*math.pi))*math.exp(-0.5*(x3)**2)
            return term1*term2*term3

        ps1t1 = integrate.quad(func3, -np.inf, np.inf, epsabs = 1e-5)[0]
        ps = self.predict_proba(sx)
        ps0, ps1 = ps[0, 0], ps[0, 1]
        pt = self.predict_proba(tx)
        pt0, pt1 = pt[0, 0], pt[0, 1]
        ps0t1 = pt1 - ps1t1
        pt1_s1 = ps1t1/ps1
        pt0_s1 = 1-pt1_s1
        pt1_s0 = ps0t1/ps0
        pt0_s0 = 1-pt1_s0
        pt_s = np.array([[pt0_s0, pt1_s0],[pt0_s1, pt1_s1]])
        return pt_s

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
    
    def _calculate_posterior_predictive_from_joint_distribution(self, xt, xs, pt1s1):
        pt = self.predict_proba(xt)
        assert(pt.shape == (1, 2))
        pt0, pt1 = pt[0, 0], pt[0, 1]
        ps = self.predict_proba(xs)
        ps0, ps1 = ps[:, 0], ps[:, 1]
        pt0s1 = ps1 - pt1s1
        ps1_t1 = pt1s1/pt1
        ps0_t1 = 1-ps1_t1
        ps1_t0 = pt0s1/pt0
        ps0_t0 = 1-ps1_t0

        ps_t0 = np.column_stack((ps0_t0, ps1_t0))
        ps_t1 = np.column_stack((ps0_t1, ps1_t1))

        return ps_t0, ps_t1

    
    def OneStepPredict(self, xt, xs, version='numpy'):
        #xt is xstar
        #xs is an array of size x_num*f_num
        mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat = self._calculate_mean_and_variance(xt, xs)
        sigma_s = np.sqrt(sigma_ss)
        
        def func4(f0):
            #This function use hermite Gaussian quadrature, 
            # return: a x_num array of value with index corresponding to xs. 
            # term3 = 1/(sigma1*math.sqrt(2*math.pi))*math.exp(-0.5*(x3)**2) is normalized as Gaussian function
            Phi = lambda x : 0.5*(erf(x/math.sqrt(2))+1) 
            fs = f0*math.sqrt(2)*sigma_s+mu_s
            mu_t_hat = mu_t + sigma_st/sigma_ss*(fs-mu_s)
            ft_hat = mu_t_hat/np.sqrt(sigma_tt_hat+1)
            term1 = Phi(ft_hat)
            term2 = Phi(fs)
            return term1*term2/math.sqrt(math.pi)#math.sqrt(math.pi) is the constant for normalized Gaussian

        # joint distribution
        pt1s1 = np.zeros(len(xs))
        for i, f0 in enumerate(A[0]):
            pt1s1 += func4(f0)*A[1][i]

        ps_t0, ps_t1 = self._calculate_posterior_predictive_from_joint_distribution(xt, xs, pt1s1)
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
        self.gpc = m

    def XspaceGenerate(self, x_num):
        xspace = XspaceGenerate_(x_num)
        return xspace

    def ObcClassifierError(self, x_num):
        #xspace = self.XspaceGenerate(x_num)
        global xtest
        pyTheta = self.predict_proba(xtest)
        yhat = np.argmax(pyTheta, axis = 1)
        #np.mean(yhat != ytest)
    
        #py  = GroundTruthProbability(xspace)
        classifier_error = np.mean(yhat != ytest)
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
