'''
Optimize method for continuous search space
include:
1. Exhausive search
2. Non-gradient method
3. Gradient ascent
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import torch

def MCSelector(func, model, mc_search_num = 1000):
    xspace = model.XspaceGenerate(mc_search_num)

    utilitymat = np.zeros(mc_search_num)+float('-Inf')

    if hasattr(model, 'multi_hyper') and model.multi_hyper:
            for i, x in enumerate(xspace):
                if hasattr(model, 'is_real_data') and model.is_real_data:
                    if i in model.dataidx:
                        continue
                x = xspace[i:i+1]
                for m in model.modelset:
                    utilitymat[i]+= func(x, m)
    else:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i+1]# all the inputs should take 2d array 
            # if version == 'pytorch':
            #     x = torch.tensor(x, requires_grad=True)
            utilitymat[i] = func(x, model)
    
    max_value = np.max(utilitymat, axis = None)
    max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))

    if hasattr(model, 'is_real_data') and model.is_real_data:
        model.dataidx = np.append(model.dataidx, max_index)

    # plt.figure()
    # plt.plot(xspace, utilitymat, 'ro')
    # plt.show()
    
    x = xspace[max_index]

    # plt.figure()
    # plt.plot(xspace, utilitymat)
    # plt.show()

    return x, max_value

def RandomSampling(model):
    x = model.XspaceGenerate(1)
    max_value = 0
    return x, max_value

def SGD(func, model, mc_search_num = 1000, learning_rate = 0.001):
    #for mm in range(100):
    random_num = round(0.7*mc_search_num)
    #x11, value11 = MCSelector(func, model, mc_search_num)
    x1, value1 = MCSelector(func, model, random_num)
    #x0 = model.XspaceGenerate(1).reshape(-1)
    x0 = torch.tensor(x1, requires_grad= True)
    optimizer = torch.optim.SGD([x0], lr=learning_rate)
    

    # for _ in range(round(0.3*mc_search_num)):
    #     loss = -func(x0, model, version='pytorch')

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print("loss: {}".format(loss))
    
    # x0 = torch.tensor(x1, requires_grad= True)
    # optimizer = torch.optim.Adam([x0], lr=learning_rate)
    

    for _ in range(round(0.3*mc_search_num)):
        loss = -func(x0, model, version='pytorch')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # print("loss: {}".format(loss))

    return x0.detach().numpy(), -loss

    # func2 = lambda x: -1.0*func(x, model)
    # bounds = np.array([model.xinterval[0], model.xinterval[1]])*np.ones((model.f_num, 2))
    # res = minimize(func2, x0, method='TNC', options={'disp':False}, bounds = bounds)
    # xstar = res.x
    # max_value = -res.fun
    # return xstar, max_value
    # max_value = float('-Inf')
    # for mm in range(50):
    #     x0 = model.XspaceGenerate(1).item()
    #     func2 = lambda x: -1.0*func(x, model)
    #     bounds = [(model.xinterval[0], model.xinterval[1])]
    #     res = minimize(func2, x0, method='TNC', 
    #                     options={ 'disp':False}, bounds = bounds)
    #     xstar22 = res.x
    #     max_value22 = -res.fun
    #     print(res)
    #     if max_value22.item() > max_value:
    #         max_value = max_value22.item()
    #         xstar = xstar22



    # # x0 = model.XspaceGenerate(1).item()
    # # func2 = lambda x: -1.0*func(x, model)
    # # bounds = [(-4, 4)]
    # # res = minimize(func2, x0, method='trust-constr', 
    # #                 options={#'xatol':1e-8, 
    # #                 'disp':True}, bounds = bounds)
    # # x = res.x
    # # max_value = -res.fun
    # return xstar, max_value