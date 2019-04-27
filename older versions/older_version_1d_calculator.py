# coding: utf-8
#! /usr/bin/env python
# FrequencyJumpLibrary

import numpy as np
from scipy import stats
import math as math

def KM (y, delta_t=1, Moments = [1,2,4,6,8], bandwidth = 1.5, Lowerbound = False, Upperbound = False, Kernel = 'Epanechnikov'):   #Kernel-based Regression
    Moments = [0] + Moments
    length=len(Moments)
    n = 5000
    Mn = int(n * bandwidth / 10)                                #Minor n       
    res = np.zeros([n + Mn, length])

    # Epanechnikov kernel:   3/4(1 - x²), x=-1 to x=1
    # #Uniform kernel:   1/2, , x=-1 to x=1
    Kernel = (3 * (1 - (np.linspace(-1 * bandwidth, 1 * bandwidth, Mn) / bandwidth) ** 2)) / (4 * bandwidth)             # Kernel1 = ones([Mn]) / (2 * bandwidth)
    
    yDist = y[1:] - y[:-1] 

    if (Lowerbound == False):
        Min = min(y)
    else:
        Min = Lowerbound
    if (Upperbound == False):
        Max = max(y)
    else:
        Max = Upperbound
    space = np.linspace(Min, Max, n + Mn)
    b = ((((y[:-1]-Min) / (abs(Max - Min))) * (n))).astype(int)
    trueb = np.unique(b[(b>=0)*(b<n)])
    for i in trueb:
        r = yDist[b==i]
        for l in range(length):
            res[i:i + Mn, l] += Kernel * (sum(r ** Moments[l]))

    res[:, 0][res[:, 0]==0]=1.
            
    for l in range(length-1):
        res[:, l+1] = np.divide(res[:, l+1],(res[:, 0] * math.factorial(Moments[l+1]) * (delta_t)))
        
    return res, space
