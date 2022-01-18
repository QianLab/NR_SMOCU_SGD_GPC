#%%
import numpy as np
import multiprocessing
import ProblemSetting as PS
from joblib import Parallel, delayed


#with open('RandomGenerator.txt', 'r') as f:
#    datalist = json.load(f)

#from tqdm import tqdm
num_cores = multiprocessing.cpu_count()
runnum = 300
T = 60
methodlist = [0, 1, 2, (0, 0, 4), (2, 20, True), (2, 20, False), (0, 0, True), (0, 0, False), 5]
smocu_x_num = 1000
error_x_num = 100000
information_sum_num = 10000
mc_search_num = 1000
initial_num = 100

inputs = list(range(runnum))
sq = np.random.SeedSequence(589135781)
rglist = sq.generate_state(runnum)
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores)(delayed(PS.SingleIteration)(k, T, rglist, methodlist, smocu_x_num, error_x_num, information_sum_num, mc_search_num, initial_num, optim_method = 'nonGD') for k in inputs)


#if i == 0:
#    str_label = 'random'
#elif i == 1:
#    str_label = 'MES'
#elif i == 2:
#    str_label = 'BALD'
# (2, 20, True) NR-SMOCU-20
# (2, 20, False) OR-SMOU-20
# (0, 0, 4) ADF-MOCU-RO
# (2, 20, 4) ADF-SMOCU-RO
# (0, 0, False) OR-MOCU-RO
# (0, 0, True) NR-MOCU-RO
# 5, NR-SMOCU-SGD