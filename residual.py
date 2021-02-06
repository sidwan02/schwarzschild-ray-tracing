import numpy as np
from scipy.special import ellipk as ec
from scipy.special import ellipkinc as ei
from ray_tracing import ray_tracing
from read_input import get_constraints_from_file

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


all_b = []
all_F = []
all_C = []

M = 1


content_array = get_constraints_from_file()

debug = content_array[0]
delta0_opts = content_array[3]
beam_loc = content_array[5]
y_opts = content_array[6]

for y_beam in y_opts:
    r0 = np.sqrt(np.square(beam_loc) + np.square(y_beam))

    for delta0 in delta0_opts:
        B = r0 / np.sqrt(1 - (2 * M / r0))
        b = B * np.sin(delta0)
