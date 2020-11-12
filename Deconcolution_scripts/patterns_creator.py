# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:07:50 2019

@author: andreas.boden
"""

import numpy as np

def GaussIm(shape, fwhm):
    
    c, r = np.meshgrid(np.linspace(0, shape[0] - 1, shape[0]), np.linspace(0, shape[1] - 1, shape[1]))
    
    c = c - (shape[0] - 1) / 2
    r = r - (shape[1] - 1) / 2
    
    s = fwhm/2.355
    
    g = np.exp(-(np.power(c, 2) + np.power(r, 2))/(2*np.power(s, 2)))
    
    return g

def Circular_mask(shape, radius):
    
    c, r = np.meshgrid(np.linspace(0, shape[0] - 1, shape[0]), np.linspace(0, shape[1] - 1, shape[1]))
    
    c = c - (shape[0] - 1) / 2
    r = r - (shape[1] - 1) / 2
    
    r = np.sqrt(np.power(c, 2) + np.power(r, 2))
    
    im = np.zeros_like(r)
    im[r < radius] = 1
    
    return im

def Circular_fourier_mask(shape, radius):
    
    c, r = np.meshgrid(np.linspace(0, shape[0] - 1, shape[0]), np.linspace(0, shape[1] - 1, shape[1]))
    
    c = c - (shape[0] - 1) // 2
    r = r - (shape[1] - 1) // 2
    
    r = np.sqrt(np.power(c, 2) + np.power(r, 2))
    
    im = np.zeros_like(r)
    im[r < radius] = 1
    
    return im

def BallProjection(shape, radius):
    
    c, r = np.meshgrid(np.linspace(0, shape[0] - 1, shape[0]), np.linspace(0, shape[1] - 1, shape[1]))
    
    c = c - (shape[0] - 1) / 2
    r = r - (shape[1] - 1) / 2

    r = np.sqrt(np.power(c, 2) + np.power(r, 2))

    y = np.sqrt(radius**2 - r**2)
    y[np.isnan(y)] = 0
    
    return 2*y

def AiryIm(shape, fwhm):
    
    c, r = np.meshgrid(np.linspace(0, shape[0] - 1, shape[0]), np.linspace(0, shape[1] - 1, shape[1]))
    
    c = c - (shape[0] - 1) / 2
    r = r - (shape[1] - 1) / 2
    d = np.sqrt(np.power(c, 2) + np.power(r, 2))
    
    g = np.power(np.sinc(d*0.886/fwhm), 2)
    
    return g


def RadialOf2DGauss(size, fwhm):
    
    r = np.linspace(0, size-1,size)
    s = fwhm/2.355
    g = np.exp(-np.power(r, 2)/np.power(s, 2))
    
    return g


def RadialOf2DAiry(size, fwhm):
    
    r = np.linspace(0, size-1,size)
    g = np.power(np.sinc(r*0.886/fwhm), 2)
    
    return g


def LightSheet(vol_size, vx_size, center_z, tilt_angle, fwhm):
    
    z_vec = np.linspace(-vol_size[0]/2, vol_size[0]/2, vol_size[0]//vx_size + 1)
    x_vec = np.linspace(-vol_size[1]/2, vol_size[1]/2, vol_size[1]//vx_size + 1)
    y_vec = np.linspace(-vol_size[2]/2, vol_size[2]/2, vol_size[2]//vx_size + 1)
    
    mesh_z, mesh_y, mesh_x = np.meshgrid(z_vec, x_vec, y_vec, indexing='ij')
    

    """Rotate coordinate system around y-axis"""
    mesh_z_prime = -mesh_x*np.sin(-tilt_angle) + (mesh_z - center_z)*np.cos(-tilt_angle)
#    mesh_x_prime = mesh_x*np.cos(-tilt_angle) + mesh_z*np.sin(-tilt_angle)
#    mesh_y_prime = mesh_y
    
    sheet_sigma = fwhm/2.355
    sheet = np.exp(-mesh_z_prime**2/(2*sheet_sigma**2))

    return sheet
    
    
    
    
    
    
    
    
    
    
    