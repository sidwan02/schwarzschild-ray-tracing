import numpy as np
from scipy.special import ellipk as ec
from scipy.special import ellipkinc as ei
from ray_tracing import ray_tracing
from read_input import get_constraints_from_file
from darwin_final_deflection import get_final_deflection
import matplotlib.pyplot as plt


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
fascal = content_array[1]
graphing = content_array[2]
defevol = content_array[3]
delta0_opts = content_array[4]
screen_loc = content_array[5]
beam_loc = content_array[6]
y_opts = content_array[7]

for y_beam in y_opts:
    r0 = np.sqrt(np.square(beam_loc) + np.square(y_beam))

    for delta0 in delta0_opts:
        B = r0 / np.sqrt(1 - (2 * M / r0))
        b = B * np.sin(delta0)

        all_b.append(b)

darwin_delta_max = get_final_deflection(all_b)

delta_max = ray_tracing(content_array)

print('all_b: ', all_b)
print('darwin_delta_max : ', darwin_delta_max)
print('delta_max: ', delta_max)

fig, axs = plt.subplots(2, 1, sharex=True)

ref_x = all_b - 3 * np.sqrt(3)

residual = ((delta_max - darwin_delta_max) / darwin_delta_max) * 100

axs[0].plot(ref_x, darwin_delta_max, label="darwin calculation")
axs[0].plot(ref_x, delta_max, label="my calculation")

axs[1].plot(ref_x, residual, label="percent residual")

axs[0].legend(loc='upper right')

axs[0].set_xlabel('b - 3sqrt(3)')
axs[0].set_ylabel('angle change in degrees')
axs[1].set_ylabel('residual percent')


axs[1].legend(loc='upper right')

plt.show()
