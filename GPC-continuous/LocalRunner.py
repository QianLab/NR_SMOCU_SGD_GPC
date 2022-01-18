#%%
import numpy as np
import multiprocessing
import tracemalloc
tracemalloc.start()

from joblib import Parallel, delayed
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
ModelOptimize = 1
#1. Initially estimate Hyperparameter and doesn't change again
#2. Optimize the hyperparameter in each iteration
#3. SVGD to estimate hyperparameter

if ModelOptimize == 1:
    import ProblemSetting as PS
elif ModelOptimize == 2:
    import ProblemSetting2 as PS

#from tqdm import tqdm
num_cores = 1
runnum = 1
T = 50
methodlist = [0, 1, 2, (0, 0, False), (0, 0, True), (0, 0, 4), (2, 20, False), (2, 20, True), 5]# True is taking approximation
smocu_x_num = 1000
error_x_num = 100000
information_sum_num = 10000
mc_search_num = 1000
initial_num = 50


inputs = list(range(runnum))
#sq = np.random.SeedSequence(1985674231143)
sq = np.random.SeedSequence(1985674231143)
rglist = sq.generate_state(runnum)
for k in inputs:
    PS.SingleIteration(k, T, rglist, methodlist, smocu_x_num, error_x_num, information_sum_num, mc_search_num, initial_num, localprint = True, optim_method = 'nonGD')
    
    
    #methodlist meaning:
#0:
#    str_label = 'random'

#1:
#    str_label = 'MES'

#2:
#    str_label = 'BALD'

#5:
#    str_label = 'NR-SMOCU-SGD'

#(0, 0, Flase):
#    str_label = 'OR-MOCU'

#(0, 0, True):
#    str_label = 'NR-MOCU-RO'

#(0, 0, 4):
#    str_label = 'ADF-MOCU'

#(2, 20, False):
#    str_label = 'OR-SMOCU'

#(2, 20, True):
#    str_label = 'NR-SMOCU-RO'

#(2, 20, 4):
#    str_label = 'ADF-SMOCU'
