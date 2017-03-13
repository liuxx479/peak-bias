import numpy as np
from scipy import *
from pylab import *
import os
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys, os
import hmf

z0 = 0.4
ngal_mean = 20 # arcmin^-2
zlo,zhi=0, 2.0

Pz = lambda z: 0.5*z**2/z0**3*exp(-z/z0)
z_choices = linspace(zlo,zhi,101)
prob = Pz(z_choices)
prob /= sum(prob)
Ngal_gen = lambda N: np.random.poisson(N)

redshift_gen = lambda N: np.random.choice(z_choices, size=N, p=prob)

### halo mass function gen 
#dndm_arr = array([hmf.MassFunction(z=iz, Mmin=10, Mmax=15, dlog10m=0.05).dndm for iz in z_choices])
#save('dndm_arr.npy',dndm_arr)
z_arr, M_arr = z_choices, arange(10,15,0.05)
dndm_arr = load('dndm_arr.npy')
#dndm = interpolate.RectBivariateSpline(z_arr, M_arr, dndm_arr)


