import numpy as np
from scipy import *
from pylab import *
import os
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys, os
import hmf ## from https://github.com/steven-murray/hmf
from peak_bias_func import *

###### input parameters #####
### redshift
z0 = 0.4 ## mean redshift 
ngal_mean = 20 ## number density in unit of arcmin^-2
zlo, zhi= 0, 2.0 ## the redshift cuts
### halo mass function limits and steps, for lens members
Mmin, Mmax, dlog10m=12, 15, 0.05

###### constants ######
Mpc=3.086e+24#cm
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s^2
H0 = 70.0#67.74
h = H0/100.0
OmegaM = 0.30#1#Planck15 TT,TE,EE+lowP+lensing+ext
OmegaV = 1.0-OmegaM
M_sun = 1.989e33#gram

####### small functions #######
Hcgs = lambda z: H0*sqrt(OmegaM*(1+z)**3+OmegaV)*3.24e-20
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
rho_cz = lambda z: 0.375*Hcgs(z)**2/pi/Gnewton # critical density
####### interpolate comoving distance DC 
DC_integral = lambda z: c*quad(H_inv, 0, z)[0]
z_arr = linspace(0.0, 3.5, 301)
DC_arr0 = array([DC_integral(z) for z in z_arr])
####### various distances
DC = interpolate.interp1d(z_arr, DC_arr0) # comoving distance
DA = lambda z: DC(z)/(1.0+z) # angular diameter distance
DL = lambda z: DC(z)*(1.0+z) # luminosity distance
####### Bryan & Norman 1998 fitting formula to get Rvir = [M/ (4pi/3 rho Delta_vir)]^0.33 
dd = lambda z: OmegaM*(1+z)**3/(OmegaM*(1+z)**3+OmegaV)
Delta_vir = lambda z: 18.0*pi**2+82.0*dd(z)-39.0*dd(z)**2
Rvir_fcn = lambda Mvir, z: (0.75/pi * Mvir*M_sun/(Delta_vir(z)*rho_cz(z)))**0.3333

###### source galaxies ######
Pz = lambda z: 0.5*z**2/z0**3*exp(-z/z0) ## P(z)
z_choices = linspace(zlo,zhi,501) ## center of z bins
prob = Pz(z_choices)
prob /= sum(prob)
### generate a poisson distribution with mean N
Ngal_gen = lambda N: np.random.poisson(N) 
### generate N galaxies with Pz distribution
redshift_gen = lambda N: np.random.choice(z_choices, size=N, p=prob) 

###### halo mass function for lens members ######
#dndm_arr = array([hmf.MassFunction(z=iz, Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m).dndm for iz in z_choices])
# save('dndm_arr.npy',dndm_arr)
#z_arr, M_arr = z_choices, arange(Mmin,Mmax,dlog10m)
#dndm_arr = load('dndm_arr.npy')
#dndm_arr /= sum(dndm_arr,axis=1).reshape(-1,1)
#### dndm = interpolate.RectBivariateSpline(z_arr, M_arr, dndm_arr)
dndm_arr = lambda zlens: hmf.MassFunction(z=zlens, Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m).dndm

###### lens galaxies ######
#M_lens = np.random.choice(M_arr, p=dndm_arr[z_arr==z_lens][0]) ## find lens mass
###### HOD #############
A, B, C = 47.0, 0.85, -0.1
N_lens_fcn = lambda logM, z: A*(10**(logM-14.0)/h)**B*(1+z)**C - 1.0
### Cvir from Dutton & Maccio (2014)
Cvir = lambda logM, z: 10**(0.537+0.488*exp(-0.718*z**1.08)+(-0.097+0.024*z)* (logM-12-log10(2.0/h)))
#### projection weight
def Gx_fcn (x, cNFW):#=5.0):
    '''projection function for a halo with cNFW, at location x=theta/theta_s.
    '''
    if x < 1:
        out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)+1.0/(1.0-x**2)**1.5*arccosh((x**2+cNFW)/x/(cNFW+1.0))
    elif x == 1:
        out = sqrt(cNFW**2-1.0)/(cNFW+1.0)**2*(cNFW+2.0)/3.0
    elif 1 < x <= cNFW:
        out = 1.0/(x**2-1.0)*sqrt(cNFW**2-x**2)/(cNFW+1.0)-1.0/(x**2-1.0)**1.5*arccos((x**2+cNFW)/x/(cNFW+1.0))
    elif x > cNFW:
        out = 0
    return out
##### projected kappa due to lens with logM, at source location
def kappa_proj (logM,  z_lens, z_source_arr, x_source_arr, y_source_arr, x_lens=0, y_lens=0, thetaG=1.0):
    '''calculate the projected mass of the forground halo.
    input: 
    logM, z_lens - lens mass and redshift
    z_source_arr, x_source_arr, y_source_arr - [z, x, y] of the source galaxies, in an array. x, y are in arcmin.
    thetaG - smoothing scale in arcmin, default is 1 arcmin.
    output:
    kappa_p - an array of size = len(x), the projected kappa at the location of each source. 
    source_contribution - the contribution from each kappa_i to the total kappa, weighted by the smoothing kernel of size thetaG.
    ''' 
    Mvir = 10**logM
    Rvir = Rvir_fcn(Mvir,z_lens)
    DC_lens, DC_source = DC(z_lens), DC(z_source_arr)
    cNFW = Cvir(logM, z)
    f = 1.0/(log(1.0+cNFW)-cNFW/(1.0+cNFW))# = 1.043 with cNFW=5.0
    two_rhos_rs = Mvir*M_sun*f*cNFW**2/(2*pi*Rvir**2)#cgs, see LK2014 footnote
    Dl_cm = 3.08567758e24*DC_lens/(1.0+z_lens)
    ## note: 3.08567758e24cm = 1Mpc###    
    SIGMAc = 347.29163*DC_source*(1+z_lens)/(DC_lens*(DC_source-DC_lens))
    ## note: SIGMAc = 1.07163e+27/DlDlsDs
    ## (c*1e5)**2/4.0/pi/Gnewton = 1.0716311756473212e+27
    ## 347.2916311625792 = 1.07163e+27/3.08567758e24
    theta_arcmin = sqrt((x_lens-x_source_arr)**2+(y_lens-y_source_arr)**2)
    theta = radians(theta_arcmin/60.0)
    x = cNFW*theta*Dl_cm/Rvir 
    ## note: x=theta/theta_s, theta_s = theta_vir/c_NFW
    ## theta_vir=Rvir/Dl_cm
    Gx_arr = array([Gx_fcn(ix, cNFW) for ix in x])
    kappa_p = two_rhos_rs/SIGMAc*Gx_arr  
    kappa_p[z_source_arr<z_lens]=0
    source_contribution = exp(-0.5*theta**2/radians(thetaG/60.0)**2)
    kappa = sum(kappa_p * source_contribution)/sum(source_contribution)
    #return kappa, kappa_p
    return source_contribution, kappa_p
