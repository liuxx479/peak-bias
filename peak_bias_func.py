import numpy as np
from scipy import *
from pylab import *
import os
from scipy import interpolate,stats
from scipy.integrate import quad
import scipy.optimize as op
import sys, os
import hmf ## from https://github.com/steven-murray/hmf
from scipy.spatial import cKDTree  

#############################
###### input parameters #####
#############################

### redshift
z0 = 0.4 ## mean redshift 
ngal_mean = 20 ## number density in unit of arcmin^-2
zlo, zhi= 0, 2.0 ## the redshift cuts

### magnitude cut ###
ibandOBS = 6
aMlimOBS = 24.5

beta = 0.6 ## size bias
sslope = 0.5 + beta ## magnification bias
q=5.0*sslope-2.0
rblend = 5.0/60.0 # arcsec, M13 gals at z=0.5 ~ 2.5 arcsec in size
sigma_kappa = 0.35
Rgal2halo = 0.015
###############################
###### constants ##############
###############################

Mpc=3.086e+24#cm
c = 299792.458#km/s
Gnewton = 6.674e-8#cgs cm^3/g/s^2
H0 = 70.0#67.74
h = H0/100.0
OmegaM = 0.30#1#Planck15 TT,TE,EE+lowP+lensing+ext
OmegaV = 1.0-OmegaM
M_sun = 1.989e33#gram

###############################
####### small functions #######
###############################

########## generate galaxy noise
kappa_noise_gen = lambda N: normal(0.0, sigma_kappa, size=N)

Hcgs = lambda z: H0*sqrt(OmegaM*(1+z)**3+OmegaV)*3.24e-20
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV))
rho_cz = lambda z: 0.375*Hcgs(z)**2/pi/Gnewton # critical density

####### Cvir from Dutton & Maccio (2014)
Cvir = lambda logM, z: 10**(0.537+0.488*exp(-0.718*z**1.08)+(-0.097+0.024*z)* (logM-12-log10(2.0/h)))

####### interpolate comoving distance DC 
DC_integral = lambda z: c*quad(H_inv, 0, z)[0]
z_arr = linspace(0.0, 3.5, 301)
DC_arr0 = array([DC_integral(z) for z in z_arr])

####### various distances
DC = interpolate.interp1d(z_arr, DC_arr0) # comoving distance
DA = lambda z: DC(z)/(1.0+z) # angular diameter distance
DL = lambda z: DC(z)*(1.0+z) # luminosity distance

####### Bryan & Norman 1998 fitting formula to get Rvir = [M/ (4pi/3 rho Delta_vir)]^0.33 
#dd = lambda z: OmegaM*(1+z)**3/(OmegaM*(1+z)**3+OmegaV)
#Delta_vir = lambda z: 18.0*pi**2+82.0*dd(z)-39.0*dd(z)**2
#Rvir_fcn = lambda Mvir, z: (0.75/pi * Mvir*M_sun/(Delta_vir(z)*rho_cz(z)))**0.3333
####### Mvir is defined as the M200c, 200 times the critical density
Rvir_fcn = lambda Mvir, z: (0.75/pi * Mvir*M_sun/(200.0*rho_cz(z)))**0.3333 ## unit: cm

####### generate source galaxies: N, Pz
Pz = lambda z: 0.5*z**2/z0**3*exp(-z/z0) ## P(z)
z_choices = linspace(zlo,zhi,501) ## center of z bins
prob = Pz(z_choices)
prob /= sum(prob)
### generate a poisson distribution with mean N
Ngal_gen = lambda N: np.random.poisson(N) 
### generate N galaxies with Pz distribution
redshift_gen = lambda N: np.random.choice(z_choices, size=N, p=prob) 

#######################################
######### lensing projection ##########
#######################################

#### projection weight
def Gx_fcn (x, cNFW):
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
def kappa_proj (logM,  zlens, z_source_arr, x_source_arr, y_source_arr, x_lens=0, y_lens=0, thetaG=1.0):
    '''calculate the projected mass of the forground halo.
    input: 
    logM, zlens - lens mass and redshift
    z_source_arr, x_source_arr, y_source_arr - [z, x, y] of the source galaxies, in an array. x, y are in arcmin.
    thetaG - smoothing scale in arcmin, default is 1 arcmin.
    output:
    kappa_p - an array of size = len(x), the projected kappa at the location of each source. 
    source_contribution - the contribution from each kappa_i to the total kappa, weighted by the smoothing kernel of size thetaG.
    ''' 
    Mvir = 10**logM
    Rvir = Rvir_fcn(Mvir,zlens)
    DC_lens, DC_source = DC(zlens), DC(z_source_arr)
    cNFW = Cvir(logM, z)
    f = 1.0/(log(1.0+cNFW)-cNFW/(1.0+cNFW))# = 1.043 with cNFW=5.0
    two_rhos_rs = Mvir*M_sun*f*cNFW**2/(2*pi*Rvir**2)#cgs, see LK2014 footnote
    Dl_cm = 3.08567758e24*DC_lens/(1.0+zlens)
    ## note: 3.08567758e24cm = 1Mpc###    
    SIGMAc = 347.29163*DC_source*(1+zlens)/(DC_lens*(DC_source-DC_lens))
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
    kappa_p[z_source_arr<zlens]=0
    source_contribution = exp(-0.5*theta**2/radians(thetaG/60.0)**2)
    kappa = sum(kappa_p * source_contribution)/sum(source_contribution)
    #return kappa, kappa_p
    return source_contribution, kappa_p

###################################################
######## luminosity function (Gabasch+2006) #######
###################################################

##### DO NOT TOUCH THIS BLOCK ##############
##### band names
U1band, U2band, U3band, Bband, Gband, Rband, Iband, Zband = range (8)
band_str=('U1 band', 'U2 band', 'U3 band', 'B band', 'G band', 'R band', 'I band', 'Z band')
# 1500.0, 2800.0, 3546.0 are U, 4344.0 is B, 4670.0 is g, 6156.0 is r, 7472.0 is i, 8917, z band
alambdaB = np.array([1500.0, 2800.0, 3546.0, 4344.0, 4670.0, 6156.0, 7472.0, 8917.0],dtype=float64)
alambdaOBS = alambdaB [ibandOBS]

##### Gabasch+2006 Table 9 case 3
##### Gabasch+2004 Table 4
aMstar0 = np.array([-17.4, -18.16, -18.95, -20.92, -21.0, -21.49, -21.97, -22.22])
phistar0 = np.array([2.71e-2, 2.46e-2, 2.19e-2, 0.82e-2, 0.83e-2, 0.42e-2, 0.34e-2, 0.33e-2])
alpha_arr = np.array([-1.01, -1.06, -1.1, -1.24, -1.26, -1.33, -1.33, -1.33])
aLF = np.array([-2.19, -2.05, -1.8, -1.03, -1.08, -1.25, -0.85, -0.81])
bLF = np.array([-1.76, -1.74, -1.7, -1.27, -1.29, -0.85, -0.66, -0.63])
###### convert between obs and 
aMrest_fcn = lambda aMlimOBS, z: aMlimOBS  - 5.0*log10(DL(z)) - 25.0
aMobs_fcn = lambda aMlimREST, z: aMlimREST + 5.0*log10(DL(z)) + 25.0

###### redshift evolution of LF
aMstar_fcn = lambda z, i: aMstar0[i] + aLF[i]*log(1.0+z) # M* for band i, redshift z
phistar_fcn = lambda z, i: phistar0[i]*(1.0+z)**(bLF[i]) # phi* for band i, redshift z
alphahere_fcn = lambda i: alpha_arr[i] # alpha for band i
alambda_fcn = lambda alambdaOBS, z: alambdaOBS/(1.0+z) # redshift the band to rest
# luminosity function dphi/ dM, aM is the rest Magnitude
arg=lambda aM, z, i: 10.0**(2.0/5.0*(aMstar_fcn(z,i)-aM))
##### the LF for galaxies with Mag_rest=aM, z, in i band.
LF0 = lambda aM, z, i: 2.0/5.0*phistar_fcn(z,i)*(log(10.0))*arg(aM,z,i)**(alphahere_fcn(i)+1.0)*exp(-arg(aM,z,i)) 

def find_nearest(array,value):
    '''find the 2 values in array closes to value, 
    return 1st closes value, 2nd closeset value, 1st index, 2nd index
    '''
    idx = (np.abs(array-value)).argsort()
    if value > array[idx[0]]:
        idx2=idx[0]+1
    else:
        idx2=idx[0]-1
    return array[idx[0]],array[idx2],idx[0],idx2

def LF(aMlimOBS, z, i=Iband, return_Mlim_hmf=0):
    '''For limiting magnitude aMlimOBS, return the luminosity function
    '''
    aM=aMrest_fcn(aMlimOBS,z)
    Restlambda=alambdaB[i]/(1+z)    
    wL, wR, idxL, idxR = find_nearest(alambdaB, Restlambda)
    if Restlambda < wL and wL == alambdaB[0]:
        phi=LF0(aM, z, 0)
    else:
        ratioL = abs(Restlambda-wL)/abs(wL-wR)
        ratioR = abs(Restlambda-wR)/abs(wL-wR)
    if return_Mlim_hmf:
        Mlim_hmf_rest = ratioR*aMstar0[idxR] + ratioL*aMstar0[idxL] + 4.0
        Mlim_hmf_obs = aMobs_fcn(Mlim_hmf_rest, z)
        return Mlim_hmf_obs
    else:
        phi=ratioR*LF0(aM, z, idxL)+ratioL*LF0(aM, z, idxR)
        return phi
###############################################

###########################################
######################### HOD #############
###########################################

############ number of lens members
A, B, C = 47.0, 0.85, -0.1
Nlens_fcn = lambda logM, z: A*(10**(logM-14.0))**B*(1+z)**C - 1.0

###### halo mass function for lens members ######
Mmin = 13.0 ### complete for M200c>1e13 M_sun
#M200c is defined by the spherical overdensity mass with respect to 200 times critical density.
Mmax = 15.5
dlog10m = 0.01
Mlens_arr = arange(Mmin, Mmax, dlog10m)
dndm_arr = lambda zlens: hmf.MassFunction(z=zlens, Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m).dndm
### generate N lens masses with distribution following the halo mass function dndm_arr
Mlens_gen = lambda N, zlens: np.random.choice(Mlens_arr, size=N, p=dndm_arr (zlens)/sum(dndm_arr (zlens)))  

############ find the rest LF, assign size
### (1) caculate an array of LF for Mlim_obs
### (2) find Mlim_hmf, which is shifting from M*+4 in rest frame to observation frame
### (3) calculate an array of LM for Mlim_hmf
### (4) cut out N_lim most massive lenses that can make into the observation
### (5) assign a size to the selected lenses based on their mass and redshift

def Nlim_fcn (N, z, aMlimOBS=24.5, i=Iband):
    '''For one halo with N members, at redshift z, find the number of members Nlim that can make into the observed sample, where observation is conduced at aMlimOBS and i band.
    '''
    Mlim_hmf = LF(aMlimOBS, z, i=i, return_Mlim_hmf=1)
    integrand = lambda aM: LF(aM, z, i=i, return_Mlim_hmf=0)
    integral_Mlim_obs = quad(integrand, Mlim_hmf-9, aMlimOBS)[0]
    integral_Mlim_hmf = quad(integrand, Mlim_hmf-9, Mlim_hmf)[0]
    Nlim = int(N * integral_Mlim_obs/integral_Mlim_hmf + 0.5)
    return amin([Nlim, N]) ## make sure we don't return more than N galaxies

def gal_size_fcn(logM, z):
    '''For one lens member of mass logM, redhisft z, return its size in arcmin.'''    
    Rvir = Rgal2halo * Rvir_fcn(10**logM, z) ## unit: cm
    Rvir_Mpc = Rvir/Mpc
    theta_gal = degrees(Rvir_Mpc/DA(z)) * 60.0 ## unit: arcmin
    return theta_gal

######################################
########## MC ########################
######################################

def sampling (log10M, zlens, q=q, side=10.0, iseed=10027):
    '''For one lens halo with mass log10M, redshift zlens, do the following:
    (1) generate source galaxies with distribution Pz
    (2) generate member galaxies with (x, y, Mvir)
    (3) cut out member galaxies that fall fainter than Mlim
    (4) assign sizes to the remaining member galaxies
    (5) magnification bias: change the source number density at z>zlens
    (6) blending: remove galaxies overlap in size
    Return 
    '''
    seed(iseed)
    ### (1) generate source galaxies with distribution Pz
    area = side**2 ## arcmin^2
    N_gal = Ngal_gen(ngal_mean * area)
    z_source_arr = redshift_gen(N_gal)
    x_source_arr = rand(N_gal) * side
    y_source_arr = rand(N_gal) * side
    
    ### (2) generate member galaxies with (x, y, r, Mvir)
    Mvir = 10**log10M
    ### number of lens members
    Nlens = int(Nlens_fcn(log10M, zlens) + 0.5)
    ### assign a mass
    Mlenses = Mlens_gen (Nlens, zlens) 
    ### assign x, y, according to concentration
    cNFW = Cvir(log10M, z)
    ngal_like_fcn = lambda cNFW: array([Gx_fcn(ix, cNFW) for ix in linspace(0.01, cNFW, 1001)])
    ngal_like = ngal_like_fcn(cNFW)/sum(ngal_like_fcn(cNFW))    
    Rvir = Rvir_fcn(Mvir, z)
    theta_vir = degrees(Rvir/Mpc/DC(zlens))*60.0
    rlenses = theta_vir * np.random.choice(linspace(0.01, 1.0, 1001), size=Nlens, p=ngal_like)# sieze of radius in arcmin
    ang_lenses = rand(Nlens)*2*pi
    xlens = rlenses * sin(ang_lenses) ## in arcmin
    ylens = rlenses * cos(ang_lenses) ## in arcmin
    
    ### (3) cut out member galaxies that fall fainter than Mlim
    Nlim = Nlim_fcn (Nlens, zlens)
    idx_lim = argsort(Mlenses)[::-1][:Nlim]
    xlens = xlens[idx_lim]
    ylens = ylens[idx_lim]
    Mlenses = Mlenses[idx_lim]
    
    ### (4) assign sizes to the remaining member galaxies
    gal_sizes = gal_size_fcn(Mlenses, zlens) #### unit: arcmin
        
    ##### (5) magnification bias: change the source number density at z>zlens
    r_impact = theta_vir ### impact of magnification bias is only within virial radius
    N_source_back = sum( (z_source_arr>zlens) & ( sqrt((x_source_arr-side/2)**2 + (y_source_arr-side/2)**2) < r_impact))
    kappa_real =  kappa_proj (log10M,  zlens, z_source_arr, x_source_arr, y_source_arr, x_lens=side/2, y_lens=side/2)
    N_source_new = N_source_back * q * sum(kappa_real[1])
    N_source_new = int(N_source_new+0.5)
    ## new position and redshift, but limit to higher redshift
    z_source_new = np.random.choice(z_choices[z_choices>zlens], size=N_source_new, 
                                    p=prob[z_choices>zlens]/sum (prob[z_choices>zlens]))
    ang_new = rand(N_source_new)*2*pi
    x_source_new = r_impact * rand(N_source_new) * sin(ang_new)
    y_source_new = r_impact * rand(N_source_new) * cos(ang_new)
    
    ######  (6) blending: remove galaxies overlap in size
    xy = concatenate([[xlens+side/2, ylens+side/2],
                        [x_source_arr, y_source_arr],
                        [x_source_new+side/2, y_source_new+side/2]],axis=1).T
    kdt = cKDTree(xy)
    ########## these removes all galaxies within 5 arcsec, typical size of 10^13 halos at z=0.5
    idx_blend = (~isinf(kdt.query(xy,distance_upper_bound=rblend,k=2)[0][:,1]))
    idx_blend_member = []
    for iii in xrange(xlens):
        iidx = where(sqrt(sum(xy**2-array([xlens[iii],ylens[iii]])**2)) < galaxy[iii])[0]
        if len (iidx) > 1:
            idx_blend_member.append(iidx)
    idx_blend_tot=### merge idx_blend and idx_blend_member
    ############## above block to be tested, 5/4/2017
    x_blend, y_blend = xy[idx_blend_tot].T
    z_blend = concatenate([ones(Nlens)*zlens, z_source_arr, z_source_new])[~idx_blend]
    
    ###### test impacts
    x_all = concatenate([x_source_arr, xlens+side/2, x_source_new+side/2])
    y_all = concatenate([y_source_arr, ylens+side/2, y_source_new+side/2])
    z_all = concatenate([z_source_arr, ones(Nlens)*zlens, z_source_new])
    kappa_all = kappa_proj (log10M,  zlens, z_all, x_all, y_all, x_lens=side/2.0, y_lens=side/2.0)[1]
    noise_all = kappa_noise_gen(len(x_all))

    member, mag, blended = ones(shape=(3, len(x_all)))
    member [len(x_source_arr):len(x_source_arr)+len(xlens)] = 0 #### 1 are the sources
    if len(x_source_new)>0:
        mag [-len(x_source_new):] = 0 ### 1 is the ones not magnified
    blended [idx_blend] = 0 ## 1 is the ones not blended

    r_all = hypot(x_all-side/2.0, y_all-side/2.0)
    weight = exp(-0.5*r_all**2)

    kappa_sim = average(kappa_all, weights = weight*mag*member)
    kappa_noisy = average(kappa_all + noise_all, weights = weight*mag*member)
    noise = average(noise_all, weights = weight*mag*member)
    kappa_member = average(kappa_all + noise_all, weights = weight*mag)
    kappa_mag = average(kappa_all + noise_all, weights = weight*member)
    kappa_blend = average(kappa_all + noise_all, weights = weight*member*mag*blended)
    kappa_3eff = average(kappa_all + noise_all,weights =  weight*blended)
    return kappa_sim, kappa_noisy, noise, kappa_member, kappa_mag, kappa_blend, kappa_3eff
