# Efficient Active Learning for Gaussian Process Classification by Error Reduction
This is the source code for the paper [Efficient Active Learning for Gaussian Process Classification by Error Reduction](https://proceedings.neurips.cc/paper/2021/file/50d2e70cdf7dd05be85e1b8df3f8ced4-Paper.pdf) published in Neurips2021. 

As we speeded up the integral calculation with Gaussian qudrature, now the running speed of NR-SMOCU-SGD and NR-(S)MOCU-RO is even faster than the results shown in the paper. 



# Running
The code includes two directories corresponding to two scenarios of active learning (AL): query synthesis scenario (with continous search space) and pool-based scenario (with discrete search space). In each directory, just run `LocalRunner.py` to compare the performance of different active learning algorithms. 

For pool-based AL problem, the recommended algorithms are `NR-MOCU-RO` and `NR-SMOCU-RO`, while for query synthesis AL problem, the recommended algorithm is `NR-SMOCU-SGD`.

For any questions and issues, please contact guangzhao27@gmail.com
