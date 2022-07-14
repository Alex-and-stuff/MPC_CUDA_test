# MPC program implemented in CUDA

# Introduction
Here I will try to implement the MPC simulation program 
from Matlab to CUDA C++. Since the original code shows 
promissing results but cannot run in real time, we wish
to convert it into CUDA C++ to increase the effiency of 
large matrix calculations.

# Thought process
1. Setup parameters and variables
2. Initialize the initial conditions (initial state and control)
3. Build track and obstacles (from perception)
    * BuildTrack()
    * BuildObstacles()
4. K prediction rollouts (CUDA-intensive part)
    1. Generate perturbation matrix and add to control (V = U + E)
    2. Generate the states for all rollouts at each predicted time horizon
    3. Calculate the state cost of each state in the state matrix
    4. Do sum reduction once to get cost of each rollout (first element of each row)
    5. Find the min of the costs (for weight calculation, can use thrust::min_element)
    6. Calculate w_tilde and eta (using sum reduction twice)
    7. Generate control sequence using the calculated weight (multiplication + sum reduction)
5. update the predicted state and control
