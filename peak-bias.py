import numpy as np
from scipy import *
from pylab import *
import os
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys, os

z0 = 0.4
ngal_mean = 20 # arcmin^-2
Pz = lambda z: 0.5*z**2/z0**3*exp(-z/z0)

#z_arr = linspace(0,3,100)
#plot(z_arr,Pz(z_arr)); xlabel('z');ylabel('P(z)')
#savefig('Pz.png');close()

def redshift_gen (N, zlo=0, zhi=2, Pz=Pz):
    '''Generate the N redshifts with with distribution Pz
    '''
    z_choices = linspace(zlo,zhi,2001)
    prob = Pz(z_choices)
    sample = np.random.choice(z_choices, size=N, p=prob/sum(prob))
    return sample

#### test
#sample=redshift_gen(100)
#hist(sample);xlabel('z');ylabel('hist');savefig('test_reshiftgen.png');close()






