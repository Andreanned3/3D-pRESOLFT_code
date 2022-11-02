# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:16:59 2019

@author: andreas.boden
"""

import numpy as np
import matplotlib.pyplot as plt

side = 10

x, y = np.meshgrid(np.linspace(-5, 5, side), np.linspace(-5, 5, side))

sigma = 1.67

gauss = np.exp(-(x**2+y**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
const = np.ones([side, side])

bg = 1000
signal = 5000
counts = np.random.poisson(signal*gauss) + np.random.poisson(bg*const)

bases = np.array([const.reshape([1,side**2]).squeeze(), gauss.reshape([1,side**2]).squeeze()])
pinv_mat = np.linalg.pinv(bases)

coeffs = np.matmul(counts.reshape([1,side**2]).squeeze(), pinv_mat)
print('Coefficients = ', coeffs)

#plt.plot(gauss)
#plt.bar(x, counts)