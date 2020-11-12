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

"""Input pixel/voxel size (assuming isotropic pixels/voxels)"""
px_size_nm = 50

""" Make 2D PSF """
#psf_size_nm = px_size_nm*21
#psf_size_px = psf_size_nm//px_size_nm
#
#fwhm_c_large = 160/px_size_nm
#fwhm_r_large = 160/px_size_nm
#large_g = patterns_creator.Anisotropic_GaussIm([psf_size_px, psf_size_px], fwhm_c_large, fwhm_r_large)
#
#s_c_small = 75/px_size_nm
#s_r_small = 75/px_size_nm
#small_g = patterns_creator.Anisotropic_GaussIm([psf_size_px, psf_size_px], s_c_small, s_r_small)
#
#ratio = 0.3
#psf = ratio*large_g + (1-ratio)*small_g

""" --- """

""" Make 3D PSF """

x_halfside_nm = 500
y_halfside_nm = 500
z_halfside_nm = 1000

x_halfside_px = int(np.ceil(x_halfside_nm/px_size_nm))
y_halfside_px = int(np.ceil(y_halfside_nm/px_size_nm))
z_halfside_px = int(np.ceil(z_halfside_nm/px_size_nm))

z, y, x = np.meshgrid(np.linspace(-px_size_nm*z_halfside_px, px_size_nm*z_halfside_px, 2*z_halfside_px+1),
                      np.linspace(-px_size_nm*y_halfside_px, px_size_nm*y_halfside_px, 2*y_halfside_px+1),
                      np.linspace(-px_size_nm*x_halfside_px, px_size_nm*x_halfside_px, 2*x_halfside_px+1), indexing='ij')

large_g_fwhm = [400, 165, 165]
large_g_sigma = np.divide(large_g_fwhm, 2.355)
small_g_fwhm = [150, 120, 120]
small_g_sigma = np.divide(small_g_fwhm, 2.355)

large_g = np.exp(-(z**2/(2*large_g_sigma[0]**2) + y**2/(2*large_g_sigma[1]**2) + x**2/(2*large_g_sigma[2]**2)))
small_g = np.exp(-(z**2/(2*small_g_sigma[0]**2) + y**2/(2*small_g_sigma[1]**2) + x**2/(2*small_g_sigma[2]**2)))


"""Make effective psf as sum of small and large gaussian according to ratio"""
ratio = 0.0
psf = ratio*large_g + (1-ratio)*small_g

""" Single frame decon """
data_path = r'X:\Andreas\3D_MonaLisa_Data_analysis\...\.tif'

data = DataIO_tools.load_data(data_path)
# data = image_mock_mito
bg = 0
data = data - bg #Subtract background if needed
# data = data - data.min() + 1e-12
data = data.clip(0) + 1e-12 #Remove negative values
data = np.pad(data, ((10, 10),(10, 10),(10, 10)))

#plt.figure()
#plt.imshow(data)

Ht_norm = Ht(np.ones_like(data), psf)

"""Choose number of iterations"""
iterations = 30

guess_array = []
dfg_array = []
err_array =[]
norm_punish_array = []


"""Check if 2D or 3D data and allocate arrays accordingly"""
if len(np.shape(data)) > 2:
    guess = np.ones([data.shape[0], data.shape[1], data.shape[2]])
    dfg = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
    err = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
    normalized_punishment = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
else:
    guess = np.ones([data.shape[0], data.shape[1]])
    dfg = np.zeros([data.shape[0], data.shape[1]])
    err = np.zeros([data.shape[0], data.shape[1]])
    normalized_punishment = np.zeros([data.shape[0], data.shape[1]])
    
    
last_saved = 0

"""Make cupy arrays for gpu accelerated ffts"""
data = cp.array(data)
psf = cp.array(psf)
guess = cp.array(guess)
normalized_punishment = cp.array(normalized_punishment)

for i in range(1, iterations+1):
    dfg = H(guess, psf)
    err = cp.divide(data, dfg)
    normalized_punishment = cp.divide(Ht(err, psf), Ht_norm)
    guess = cp.multiply(guess, normalized_punishment)
    
    if i > last_saved:
        last_saved = i
        print('Saving... i = ', i)
        guess_array.append(cp.asnumpy(guess))
        # dfg_array.append(dfg)
        # err_array.append(err)
        # norm_punish_array.append(normalized_punishment)

guess_array.append(cp.asnumpy(guess))
save_guess = copy.copy(np.asarray(guess_array))


"""Save deconvolved data"""
splitpath = os.path.splitext(data_path)

savepath = splitpath[0] + '_Deconvolved_0_' + '.tif'
i = 0
while os.path.isfile(savepath):
    i += 1
    splitpath = os.path.splitext(savepath)
    savepath = splitpath[0][0:-2] + str(i) + '_' + splitpath[1]
    
DataIO_tools.save_data(save_guess, savepath)
""" --- """
paramdict = {'Voxel size (nm)': px_size_nm,
             'Large Gaussian FWHM (nm)': large_g_fwhm,
             'Small Gaussian FWHM (nm)': small_g_fwhm,
             'Ratio': ratio,
             'Background removed': bg}

json_path = os.path.splitext(savepath)[0] + '_parameters.json'


"""Save deconvolution parameters as json"""
with open(json_path, 'w') as f:
        json.dump(paramdict, f)

