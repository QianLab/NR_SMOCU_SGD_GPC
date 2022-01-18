#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
GPC with MOCU
"""
#import sys
#sys.path.append('D:\\Google Drive\\3-Research\\2019-MOCU-active-learning\\utils')
import numpy as np
from time import time
from ModelSetting import  *

from UtilityFunctions import *
from OptimizeMethod import *
from matplotlib import pyplot as plt
#import psutil
import copy




def SingleIteration(k, T, rglist, methodlist, smocu_x_num, error_x_num, information_sum_num, 
            mc_search_num = 1000, initial_num = 10, localprint = False, localdraw = False,
            optim_method = 'MC', all_data_txt = False):
    # smocu_x_num, the num to calculate SMOCU
    # error_x_num the num to calculate error 
    # information_sum_num the num to calculte True information
    # initial num, the num to intially train GP
    # mc search num, the num for monte carlo optimization
    
    np.random.seed(rglist[k])

    SetGlobalNumber(mc_search_num)#generate a function as complex as search number

    f = lambda x: GroundTruthFunction(x)

    X_, Y_, Xindex = InitialDataGenerator(f, initial_num)

    

    print(k)
    bayesian_error = BayesianError(error_x_num)


    # if optim_method == 'MC':
    #     Selector = MCSelector
    # elif optim_method == 'nonGD':
    #     Selector = DerivativeFree
    # elif optim_method == 'GD':
    #     Selector = GradientDescent
    
    model0 = Model(X_, Y_, optimize = True)
    model0.dataidx = Xindex

    #Active Learning
    for i in methodlist:
        model = copy.copy(model0)
        # model = ModelSet(X_, Y_, hypernum = 20)
        
        if isinstance(i, int):
            if i == 0:
                str_label = 'random'
            elif i == 1:
                str_label = 'MES'
            elif i == 2:
                str_label = 'BALD'
            elif i == 3:
                str_label = 'MOCU' # OR-MOCU
            elif i == 4:
                str_label = 'True_Information'# Calculate information with Monte Carlo 10000
            elif i == 5:
                str_label = 'SMOCU_sgd' #NR-SMOCU-SGD
            elif i == 6:
                str_label = 'ThompsonSampling'
            elif i <0:
                str_label = 'Soft_MOCU_'+str(-i)
        elif isinstance(i, tuple):

            # str(i[2]) == True     NR-MOCU-RO
            # str(i[2]) == False    OR-MOCU
            # str(i[2]) == 4        ADF-MOCU
            if i[0] == 0:
                str_label = 'MOCU'+str(i[2]) 
            
            # str(i[2]) == True     NR-SMOCU-RO
            # str(i[2]) == False    OR-SMOCU
            # str(i[2]) == 4        ADF-SMOCU
            elif i[0] == 2:
                str_label = 'Soft_MOCU'+str(i[1])+str(i[2])
            elif i[0] == 3:
                str_label = 'Weighted_MOCU_b'+str(i[1])

        error_txt = open('error'+str_label+'.txt', 'a')
        error_txt.write(str(k)+'\t')

        errortemp = model.ObcClassifierError(x_num = error_x_num) - bayesian_error
        error_txt.write(str(errortemp)+'\t')

        time_txt = open('time'+str_label+'.txt', 'a')
        time_txt.write(str(k)+'\t')

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
                    #xstar, max_value = RandomSampling(model)
                    xstar, max_value = MCSelector(U_RANDOM, model = model, mc_search_num = mc_search_num)
                elif i == 1:
                    xstar, max_value = MCSelector(U_MES, model = model, mc_search_num = mc_search_num)
                elif i == 2:
                    xstar, max_value = MCSelector(U_BALD, model = model, mc_search_num = mc_search_num)
                elif i == 3:   
                    xstar, max_value = MCSelector(U_SMOCU(softtype = 0, k = 0, x_num = smocu_x_num, approx_label=False), model = model, mc_search_num = mc_search_num)
                elif i == 4:
                    xstar, max_value = MCSelector(U_TrueInformation(information_sum_num = information_sum_num), model = model, mc_search_num = mc_search_num)
                elif i == 5:    #SMOCU with approximation with continuous search
                    xstar, max_value = SGD(U_SMOCU(softtype = 2, k = 20, x_num = smocu_x_num, approx_label = True), model = model, mc_search_num = mc_search_num)
            elif isinstance(i, tuple):
                xstar, max_value = MCSelector(U_SMOCU(softtype = i[0], k = i[1], x_num = smocu_x_num, approx_label = i[2]), model = model, mc_search_num = mc_search_num)##################
             #   xstar2, max_value2 = MCSelector(U_SMOCU(softtype = 2, k = 20, x_num = smocu_x_num, approx_label= True), model = model, mc_search_num = mc_search_num)##################
            time_txt.write(str(time() - start_time)+'\t')
            if localprint:
                print("--- %s is %s seconds ---" % (str_label, time() - start_time))
                #current, peak = tracemalloc.get_traced_memory()
                #print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            ystar = f(xstar)

            if localprint and t%20 == 0:
                ModelDraw(model, str_label+str(t)+'.jpg')
            model.Update(xstar, ystar)
            
            errortemp = model.ObcClassifierError(x_num = error_x_num) - bayesian_error#################this one contribute most 
            error_txt.write(str(errortemp)+'\t')
            if all_data_txt:
                data_txt.write(str([xstar, ystar])+'\t')
                acq_txt.write(str(max_value)+'\t')
                # MOCU_txt.write(str(MOCU)+'\t')
        error_txt.write('\n')
        error_txt.close()
        time_txt.write('\n')
        time_txt.close()
        if all_data_txt:
            data_txt.write('\n')
            data_txt.close()
            acq_txt.write('\n')
            MOCU_txt.write('\n')
            acq_txt.close()
            MOCU_txt.close()
        
        del model

# %%
