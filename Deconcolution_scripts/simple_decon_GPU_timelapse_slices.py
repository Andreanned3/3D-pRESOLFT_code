# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:28:48 2020

@author: andreas.boden
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:55:48 2019

@author: andreas.boden
"""
import os

import DataIO_tools
import patterns_creator
import cusignal
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import copy
import json

cusignal.precompile_kernels()
"""Define H and H_transpose functions"""
def H(guess, psf):
    return cusignal.fftconvolve(guess, psf, mode='same') + 1e-12

def Ht(data, psf):
    return cusignal.fftconvolve(data, psf, mode='same') + 1e-12

"""Input pixel/ size (assuming isotropic pixels)"""
px_size_nm = 40

""" Make 2D PSF """
x_halfside_nm = 250
y_halfside_nm = 500

x_halfside_px = int(np.ceil(x_halfside_nm/px_size_nm))
y_halfside_px = int(np.ceil(y_halfside_nm/px_size_nm))

y, x = np.meshgrid(np.linspace(-px_size_nm*y_halfside_px, px_size_nm*y_halfside_px, 2*y_halfside_px+1),
                   np.linspace(-px_size_nm*x_halfside_px, px_size_nm*x_halfside_px, 2*x_halfside_px+1), indexing='ij')


"""Set PSF size parameters"""
large_g_fwhm = [450, 160]
large_g_sigma = np.divide(large_g_fwhm, 2.355)
small_g_fwhm = [120, 85]
small_g_sigma = np.divide(small_g_fwhm, 2.355)

large_g = np.exp(-(y**2/(2*large_g_sigma[0]**2) + x**2/(2*large_g_sigma[1]**2)))
small_g = np.exp(-(y**2/(2*small_g_sigma[0]**2) + x**2/(2*small_g_sigma[1]**2)))


"""Make effective psf as sum of small and large gaussian according to ratio"""
ratio = 0.18
psf = ratio*large_g + (1-ratio)*small_g


""" Single frame decon """
data_path = r'D:\Andreas\Dropbox (Biophysics)\3DpRESOLFT\NatureBiotechnology\Revision\Movies\Movie_1_source\C8_slice12_bg_subtracted.tif'

data = DataIO_tools.load_data(data_path)
if len(data.shape) == 3: #if only one slice or one timepoint
    data = np.asarray([data])
    
    
"""Optional background subtraction"""
bg = 0
data = data - bg
# data = data - data.min() + 1e-12
data = data.clip(0) + 1e-12
data = np.pad(data, ((0,0),(0,0),(15,15),(15,15)))
#plt.figure()
#plt.imshow(data)

Ht_norm = Ht(np.ones_like(data[0,0]), psf)

iterations = 50

"""Initialize arrays for output data"""
guess_array = []
dfg_array = []
err_array =[]
norm_punish_array = []

frame_shape = [data.shape[2], data.shape[3]]    

"""Initialize intermediate arrays"""
guess = np.ones(frame_shape)
dfg = np.zeros(frame_shape)
err = np.zeros(frame_shape)
normalized_punishment = np.zeros(frame_shape)

last_saved = 0

"""Make cupy arrays"""
data = cp.array(data)
psf = cp.array(psf)
guess = cp.array(guess)
normalized_punishment = cp.array(normalized_punishment)


"""Iterate over slices and timepoints and deconvolve 2D images"""
for t in range(data.shape[0]):
    print('t = ', t)
    for s in range(data.shape[1]):
        print('s = ', s)
        guess = np.ones(frame_shape)
        dfg = np.zeros(frame_shape)
        err = np.zeros(frame_shape)
        normalized_punishment = np.zeros(frame_shape)
    
        last_saved = 0
        """Make cp arrays"""
        data = cp.array(data)
        psf = cp.array(psf)
        guess = cp.array(guess)
        normalized_punishment = cp.array(normalized_punishment)
        for i in range(1, iterations+1):
            dfg = H(guess, psf)
            err = cp.divide(data[t][s], dfg)
            normalized_punishment = cp.divide(Ht(err, psf), Ht_norm)
            guess = cp.multiply(guess, normalized_punishment)
            
            if i > last_saved:
                last_saved = i
                # print('Saving... i = ', i)
                guess_array.append(cp.asnumpy(guess))
                # dfg_array.append(dfg)
                # err_array.append(err)
                # norm_punish_array.append(normalized_punishment)
        

        guess_array.append(cp.asnumpy(guess))


"""Save data NOTE: Saved data might need to be reshaped"""
save_guess = copy.copy(np.asarray(guess_array))
splitpath = os.path.splitext(data_path)

savepath = splitpath[0] + '_Deconvolved_0_' + '.tif'
i = 0
while os.path.isfile(savepath):
    i += 1
    splitpath = os.path.splitext(savepath)
    savepath = splitpath[0][0:-2] + str(i) + '_' + splitpath[1]

DataIO_tools.save_data(save_guess, savepath)

"""Save deconvolution paramaters as json file"""

paramdict = {'Voxel size (nm)': px_size_nm,
             'Large Gaussian FWHM (nm)': large_g_fwhm,
             'Small Gaussian FWHM (nm)': small_g_fwhm,
             'Ratio': ratio,
             'Background removed': bg}

json_path = os.path.splitext(savepath)[0] + '_parameters.json'

with open(json_path, 'w') as f:
        json.dump(paramdict, f)


