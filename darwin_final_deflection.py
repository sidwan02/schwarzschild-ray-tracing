import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk as ec
from scipy.special import ellipkinc as ei

# Given b find the closest approach R


def Rmin(b):
    coeffs = [1.0, 0.0, -b*b, 2.0*b*b]
    roots_array = np.roots(coeffs)
    return np.amax(roots_array)

# Given R find total deflection angle


def theta(R):
    Q = np.sqrt((R-2.)*(R+6.))
    aa = 4. * np.sqrt(R/Q)
    ksq = (6.+Q-R) / (2.*Q)
    tmp = np.sqrt((2.+Q-R)/(6.+Q-R))
    amp = np.arcsin(tmp)
    Fc = ec(ksq)
    Fi = ei(amp, ksq)
    return aa*(Fc - Fi) - np.pi


# Vectorize functions for speedup
vec_Rmin = np.vectorize(Rmin)
vec_theta = np.vectorize(theta)


def get_final_deflection(b_vec):
    # b_vec = [5.517827601612367]

    r_vec = vec_Rmin(b_vec)
    theta_vec = vec_theta(r_vec) / np.pi * 180

    return theta_vec
    # print('theta_vec: ', theta_vec)


# print(get_final_deflection([10]))
