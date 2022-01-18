import numpy as np
import copy
from scipy.stats import entropy
from scipy import special
from numpy.random import choice
from scipy.special import softmax
import GPy
import torch
from sklearn.neighbors import KDTree
from scipy.stats import norm

# xspace = np.random.uniform(-4, 4, (x_num, 1))

def SetGlobal(k_, softtype_, x_num_, approx_label_):
    global  k, softtype, x_num, approx_label
    x_num = x_num_
    k = k_
    softtype = softtype_
    approx_label = approx_label_

def SMOCU(x, model, xspace):
    # global xspace 
    # # if approx_label:
    # #     xspace = model.XspaceGenerateApprox(x_num, x)                xspace, wspace = model.XspaceGenerateApprox(x_num, x)
    # # else:
    # #     xspace = model.XspaceGenerate(x_num)
    #xspace = model.XspaceGenerate(x_num)
    pymat = model.predict_proba(xspace)
    if softtype == 0: ##it is original MOCU
        obc_correct = np.amax(pymat, axis = 1)
        smocu = np.mean(obc_correct)

    if softtype == 1:  ##soft MOCU with softmax
#            obc_correct = (pymat*np.exp(pymat*k) + (1-pymat)*np.exp(k-pymat*k))/(np.exp(pymat*k)+np.exp(k-pymat*k))
        obc_correct = np.sum(softmax(pymat*k, axis = 1)*pymat, axis=1)
        ############ softmax()
        smocu = np.mean(obc_correct)

    elif softtype == 2: # soft MOCU with logsumexp
#            pzmat_array = np.array([pymat, 1-pymat])
        obc_correct = special.logsumexp(k*pymat, axis = 1)/k##################################################################################
        smocu = np.mean(obc_correct)   #MOCU equals to bayesian max minus OBC max, bayesian max can be cancelled
    #print("x = %s" %x_num)
    return smocu

def D_SMOCU(x, model):
    smocu2 = 0
    py_x = model.predict_proba(x)
    xspace = model.XspaceGenerate(x_num)
    smocu1 = SMOCU(x, model, xspace)
    for i in range(model.c_num):
        p = py_x.flat[i]
        #print(p)
        y = i
        # if approx_label:
        #     model2 = model.UpdateApprox(x, y) # update model with data  within only the 3d region#######################
        # else:
        #     model2 = model.UpdateNew(x, y)
        model2 = model.UpdateNew(x, y)
        smocu2 += p*SMOCU(x, model2, xspace)
    return smocu2 - smocu1
    
def U_SMOCU_K(x, model, k, softtype, x_num, approx_label, version):  ########### delete xinterval           ################### model.xinterval model.generatexspace
    SetGlobal(k, softtype, x_num, approx_label)
    if approx_label is True:
        return D_SMOCU_Approx(x, model, version)
    elif approx_label == 4:
        return D_SMOCU_Approx4(x, model)
    return D_SMOCU(x, model)#this equals to E_x' logsumexp(p(y'|x', x, y))

def U_SMOCU(softtype = 0, k = 1, x_num = 1000, approx_label = False):
    return lambda x, model, version='numpy': U_SMOCU_K(x, model, k, softtype, x_num, approx_label, version)

def D_SMOCU_Approx(x, model, version):
    if version == 'pytorch':
        assert (
            type(x) == torch.Tensor and x.requires_grad
            ), "for pytorch, x should be a tensor with required grad"
    x = x.reshape(-1, model.f_num)
    smocu2 = 0
    
    

    if version == 'pytorch':
        
        logsumexp = torch.logsumexp
        mean = torch.mean
        amax = torch.amax
        sum = torch.sum
        softmax = torch.softmax
        exp = torch.exp

        py_x = model.predict_proba_torch(x)
        xspace_mean = x.detach().numpy()
        
    else:
        
        logsumexp = special.logsumexp
        mean = np.mean
        amax = np.amax
        sum = np.sum
        exp = np.exp

        py_x = model.predict_proba(x)
        xspace_mean = x

    if hasattr(model, 'is_real_data') and model.is_real_data:
        xspace = model.XspaceGenerate(x_num)
        PoverW = np.ones(x_num)
        assert(version == 'numpy') ## in discrete, version is numpy
    else:
        xspace, wspace_log, px_log = model.XspaceGenerateApprox(x_num, xspace_mean)
        wspace_log = wspace_log.reshape(-1)
        PoverW = np.exp(px_log - wspace_log)
    pymat1 = model.predict_proba(xspace)

    if version == 'pytorch':
        PoverW = torch.tensor(PoverW)
        pymat1 = torch.tensor(pymat1)

    def SMOCUApprox(pymat):
        if softtype == 0: ##it is original MOCU
            obc_correct = amax(pymat, axis = 1)

        if softtype == 1:  ##soft MOCU with softmax
            obc_correct = sum(softmax(pymat*k, axis = 1)*pymat, axis=1)

        elif softtype == 2: # soft MOCU with logsumexp
            obc_correct = logsumexp(k*pymat, axis = 1)/k
        smocu = mean(obc_correct*PoverW)
        return smocu

    
    

    smocu1 = SMOCUApprox(pymat1)

    pymat20, pymat21 = model.OneStepPredict(x, xspace, version)


    
    smocu2 = (SMOCUApprox(pymat20)*py_x[0, 0]+
            SMOCUApprox(pymat21)*py_x[0, 1])

    return smocu2-smocu1

def D_SMOCU_Approx2(x, model):
    x = x.reshape(-1, model.f_num)
    smocu2 = 0
    #px = 1/(model.xinterval[1] - model.xinterval[0])
    
    py_x = model.predict_proba(x)
    xspace, wspace_log, px_log = model.XspaceGenerateApprox(x_num, x)
    wspace_log = wspace_log.reshape(-1)
    PoverW = np.exp(px_log - wspace_log)

    def SMOCUApprox(pymat):
        if softtype == 0: ##it is original MOCU
            obc_correct = np.amax(pymat, axis = 1)

        if softtype == 1:  ##soft MOCU with softmax
            obc_correct = np.sum(softmax(pymat*k, axis = 1)*pymat, axis=1)

        elif softtype == 2: # soft MOCU with logsumexp
            obc_correct = logsumexp(k*pymat, axis = 1)/k
        smocu = np.mean(obc_correct*PoverW)
        return smocu
    pymat1 = model.predict_proba(xspace)
    smocu1 = SMOCUApprox(pymat1)

    for i in range(model.c_num):
        p = py_x.flat[i]
        #print(p)
        y = i
        # if approx_label:
        #     model2 = model.UpdateApprox
        # else:
        
        #Y_set = np.concatenate((Y_data, [[y]]), axis=0)
        
        model2 = model.UpdateNew(x, y)
        pymat = model2.predict_proba(xspace)

        smocu2 += SMOCUApprox(pymat)*p

    return smocu2-smocu1


def D_SMOCU_Approx4(x, model):
    # this use ADF approximation
    x = x.reshape(-1, model.f_num)
    smocu2 = 0    

    py_x = model.predict_proba(x)
    xspace, wspace_log, px_log = model.XspaceGenerateApprox(x_num, x)
    wspace_log = wspace_log.reshape(-1)
    PoverW = np.exp(px_log - wspace_log)

    def SMOCUApprox(pymat):
        if softtype == 0: ##it is original MOCU
            obc_correct = np.amax(pymat, axis = 1)

        if softtype == 1:  ##soft MOCU with softmax
            obc_correct = np.sum(softmax(pymat*k, axis = 1)*pymat, axis=1)

        elif softtype == 2: # soft MOCU with logsumexp
            obc_correct = logsumexp(k*pymat, axis = 1)/k
        smocu = np.mean(obc_correct*PoverW)
        return smocu
    pymat1 = model.predict_proba(xspace)
    smocu1 = SMOCUApprox(pymat1)

    mucov = model.gpc.predict_noiseless(x, full_cov=False)
    mu_ = mucov[0].reshape(-1).item()
    sigma2_ = mucov[1].item()

    #kstar = 
    
    #start = time()
    mucovstar = model.gpc.predict_noiseless(xspace, full_cov=False)   #leave for latter 
    mu_space = mucovstar[0] # 1000*1
    Sigmastar = mucovstar[1]
    #kstar = Sigmastar[0:1, 1:]
    #noiseless_variance = model.gpc.kern.K(x) - model.gpc.kern.K(x, X_) @ model.gpc.posterior.woodbury_inv @ model.gpc.kern.K(X_, x)
    X_ = model.gpc.X
    kstar = model.gpc.kern.K(xspace, x) - model.gpc.kern.K(xspace, X_)@ model.gpc.posterior.woodbury_inv @ model.gpc.kern.K(X_, x)  #size 1*1000
    #print(time() - start)

    # start = time()
    # mucovstar = model.gpc.predict_noiseless(xspace, full_cov=False)  #leave for latter 
    # print(time() - start)

    for i in range(2):

        

        p = py_x.flat[i]


        # y = i
        
        # model2 = model.UpdateNew(x, y)
        # pymat = model2.predict_proba(xspace)

        y = 2*i - 1

        z = y*mu_/np.sqrt(1+sigma2_)
        mu_hat = mu_+y*sigma2_*norm.pdf(z)/(norm.cdf(z)*np.sqrt(1+sigma2_))
        sigma2_hat = sigma2_ - (sigma2_**2)*norm.pdf(z)/((1+sigma2_)*norm.cdf(z))*(z+norm.pdf(z)/norm.cdf(z))
        sigma2_tilde = 1/(1/sigma2_hat - 1/sigma2_)
        mu_tilde = sigma2_tilde*(mu_hat/sigma2_hat - mu_/sigma2_)

        KSinv = 1/(sigma2_+sigma2_tilde)

        #mu_x = mustar[0]
        #mu_space = mustar
        mu_space_hat = mu_space + kstar * KSinv * (mu_tilde - mu_)
        sigma2_space_hat = Sigmastar - KSinv* kstar * kstar
        pymat2 = np.ones(( len(xspace), 2))
        pymat2[:, 1:2] = norm.cdf(mu_space_hat/np.sqrt(1+ sigma2_space_hat))
        pymat2[:, 0] = 1 - pymat2[:, 1]




        

        smocu2 += SMOCUApprox(pymat2)*p

    return smocu2-smocu1




def U_BALD(x, model):
    py_x = model.predict_proba(x)
    obj1 = entropy(py_x, base=2, axis = 1)

    C = np.sqrt(np.pi *np.log(2)/2)
    x2 = x.reshape(-1, model.f_num)
    f_predict = model.gpc.predict_noiseless(x2)
    mu = f_predict[0].item()
    var = f_predict[1].item()
    obj2 = C/np.sqrt(var+C**2)*np.exp(-(mu**2)/(2*(var+C**2)))
    return obj1 - obj2

def U_MES(x, model):
    py_x = model.predict_proba(x)
    obj = entropy(py_x, axis = 1)
    return obj

def U_RANDOM(x, model):
    obj = 0
    return obj

def U_TrueInformation_s(x, model, sumnum):
    lik = model.lik
    py_x = model.predict_proba(x)
    

    obj1 = entropy(py_x, base = 2, axis = 1)
    x = x.reshape(-1, model.f_num)
    f_predict = model.gpc.predict_noiseless(x)
    mu = f_predict[0].item()
    var = f_predict[1].item()

    fspace = np.random.normal(mu, var, (sumnum, 1))
    p = lik.gp_link.transf(fspace)
    py_0 = 1-p
    py = np.concatenate((py_0, p), axis = 1)
    temp = entropy(py, base=2, axis = 1)
    obj3 = temp.mean()

    return obj1 - obj3

def U_TrueInformation(information_sum_num = 10000):
    return lambda x, model: U_TrueInformation_s(x, model, sumnum = information_sum_num)