#!/usr/bin/env python3
# -*- coding: utf-8 -*-  ## This file is out of date, we have update Origianl MOCU SMOCU, and ADF methods
#%%
"""
GPC with MOCU
"""
#import sys
#sys.path.append('D:\\Google Drive\\3-Research\\2019-MOCU-active-learning\\utils')
import numpy as np
#import matplotlib.pyplot as plt
#from numpy.random import choice
from time import time
#import json
#import copy
#import pickle 
#import MethodsGpcSinglePoint as Methods
#import MethodsFlipError as Methods
#from numpy.random import default_rng, SeedSequence
from Initialization import  SetGlobalNumber, GroundTruthProbability, GroundTruthFunction, InitialDataGenerator, BayesianError, Model, discrete_label, ModelDraw, optimize_label, ModelSet
from sklearn.linear_model import LogisticRegression

from UtilityFunctions import *
from OptimizeMethod import *
from matplotlib import pyplot as plt
#import psutil
import tracemalloc




def SingleIteration(k, T, rglist, methodlist, smocu_x_num, error_x_num, information_sum_num, 
            mc_search_num = 1000, initial_num = 10, localprint = False, localdraw = False,
            optim_method = 'MC'):
    # smocu_x_num, the num to calculate SMOCU
    # error_x_num the num to calculate error 
    # information_sum_num the num to calculte True information
    # initial num, the num to intially train GP
    # mc search num, the num for monte carlo optimization

    optimize_label = True
    
    np.random.seed(rglist[k])
    all_data_txt = True

    SetGlobalNumber(mc_search_num)#generate a function as complex as search number

    f = lambda x: GroundTruthFunction(x)

    X_, Y_ = InitialDataGenerator(f, initial_num)

    #############
    # model = Model(X_, Y_)#############
    # from time import time
    # i = (2, 10)
    # x_num = 10
    # smocu_x_num = 1000
    # runnum = 1
    # softtype = 2
    # k = 10
    # SetGlobal(k_ = 10, softtype_ = 2, x_num_ = x_num, approx_label_ = True)
    # func1 = U_SMOCU(softtype = 2, k = 10, x_num = 1000, approx_label = False)
    # func2 = D_SMOCU_Approx
    # func3 = D_SMOCU_Approx2
    # array1 = np.zeros(runnum)
    # array2 = np.zeros(runnum)
    # array3 = np.zeros(runnum)
    # x = model.XspaceGenerate(1)
    # #x = np.array([[3]])
    # t1 = 0
    # t2 = 0
    # t3 = 0
    # for m in range(runnum):
    #     start_time = time()
    #     array1[m] = func1(x, model)
    #     t1+= (time() - start_time)
    #     start_time = time()
    #     array2[m] = func2(x, model)
    #     t2+=( time() - start_time)
    #     start_time = time()
    #     array3[m] = func3(x, model)
    #     t3+=( time() - start_time)
    # print(array1)
    # print(array2)
    # print(array3)
    # print(t1)
    # print(t2)
    # print(t3)
    # func4 = U_SMOCU(softtype = 2, k = 10, x_num = 1000000, approx_label = False)
    # aa1 = func4(x, model)
    # print(np.square(array1 - aa1).mean())
    # print(np.square(array2 - aa1).mean())
    # print(np.square(array3 - aa1).mean())

    

    print(k)
    bayesian_error = BayesianError(error_x_num)


    if optim_method == 'MC':
        Selector = MCSelector
    elif optim_method == 'nonGD':
        Selector = DerivativeFree
    elif optim_method == 'GD':
        Selector = GradientDescent
    
    #Active Learning
    for i in methodlist:
        model = Model(X_, Y_, optimize = optimize_label)
        # model = ModelSet(X_, Y_, hypernum = 20)
        
        if isinstance(i, int):
            if i == 0:
                str_label = 'random'
            elif i == 1:
                str_label = 'MES'
            elif i == 2:
                str_label = 'BALD'
            elif i == 3:
                str_label = 'MOCU'
            elif i == 4:
                str_label = 'True_Information'# Calculate information with Monte Carlo 10000
            elif i == 5:
                str_label = 'Weighted_MOCU2'# weghted MOCU with 1-ck
            elif i == 6:
                str_label = 'ThompsonSampling'
            elif i <0:
                str_label = 'Soft_MOCU_'+str(-i)
        elif isinstance(i, tuple):
            if i[0] == 0:
                str_label = 'MOCU'+str(i[2])
            elif i[0] == 2:
                str_label = 'Soft_MOCU'+str(i[1])+str(i[2])
            elif i[0] == 3:
                str_label = 'Weighted_MOCU_b'+str(i[1])

        error_txt = open('error'+str_label+'.txt', 'a')
        error_txt.write(str(k)+'\t')

        if all_data_txt:
            data_txt = open(str_label+'data.txt', 'a')
            acq_txt = open(str_label+'acq.txt', 'a')
            MOCU_txt = open(str_label+'MOCU.txt', 'a')

            data_txt.write(str(k)+'\t')
            acq_txt.write(str(k)+'\t')
            MOCU_txt.write(str(k)+'\t')
        
        for t in range(T):
            
            start_time = time()
            if isinstance(i, int):
                if i == 0:
                    # xidx = np.random.randint(len(problem.xspace))
                    # xstar = problem.xspace[xidx]
                    xstar, max_value = RandomSampling(model)
                elif i == 1:
                    xstar, max_value = Selector(U_MES, model = model, mc_search_num = mc_search_num)
                elif i == 2:
                    xstar, max_value = Selector(U_BALD, model = model, mc_search_num = mc_search_num)
                elif i == 3:
                    xstar, max_value = MCSelector(U_SMOCU(softtype = 0, k = 0, x_num = smocu_x_num), model = model, mc_search_num = mc_search_num)
                elif i == 4:
                    xstar, max_value = Selector(U_TrueInformation(information_sum_num = information_sum_num), model = model, mc_search_num = mc_search_num)
            elif isinstance(i, tuple):
                xstar, max_value = Selector(U_SMOCU(softtype = i[0], k = i[1], x_num = smocu_x_num, approx_label = i[2]), model = model, mc_search_num = mc_search_num)##################
                #xstar2, max_value = MCSelector(U_SMOCU(softtype = i[0], k = i[1], x_num = smocu_x_num, approx_label= True), model = model, mc_search_num = mc_search_num)##################
            if localprint:
                print("--- %s is %s seconds ---" % (str_label, time() - start_time))
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                
            ystar = f(xstar)

            if localprint and t%1 == 0:
                ModelDraw(model, str_label+str(t)+'.jpg')
            model.Update(xstar, ystar, optimize = True)
            
            errortemp = model.ObcClassifierError(x_num = error_x_num) - bayesian_error#################this one contribute most 
            error_txt.write(str(errortemp)+'\t')
            if all_data_txt:
                data_txt.write(str([xstar, ystar])+'\t')
                acq_txt.write(str(max_value)+'\t')
                # MOCU_txt.write(str(MOCU)+'\t')
        error_txt.write('\n')
        error_txt.close()
        if all_data_txt:
            data_txt.write('\n')
            data_txt.close()
            acq_txt.write('\n')
            MOCU_txt.write('\n')
            acq_txt.close()
            MOCU_txt.close()
        
# %%
