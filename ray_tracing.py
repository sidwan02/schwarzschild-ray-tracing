import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import takewhile
from array import array
import re

# https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array-without-truncation
np.set_printoptions(threshold=np.inf)

mpl.rcParams['agg.path.chunksize'] = 10000


# global variables
content_array = []


# Getting inputs from input file


def file_read():
    with open("schwarzschild_ray_tracing\input.txt") as f:
        for line in f:
            if not line.startswith("#") and not line == "\n" and not line.startswith("=>"):
                content_array.append((line.rstrip()).lstrip())

    if (len(content_array) < 6 or len(content_array) > 6):
        print("Error: Please provide 6 inputs")
        return False
    else:
        assign_variables()
        return True


def parse_array(var, t):
    def clean_1(s):
        return s.replace(',', '').replace('[', '').replace(']', '')

    def clean_2(s):
        return s.replace(';', '').replace('[', '').replace(']', '')

    if var == 'delta0_opts':
        if t == 'Radial' or t == 'Parallel':
            return t
        else:
            t_split = t.split()
            if ';' not in t:
                t_cleaned_1 = list(map(clean_1, t_split))
                try:
                    return np.array(list(map(float, t_cleaned_1)))
                except ValueError:
                    raise Exception("Please recheck your input for 'defevol'")
            else:
                t_cleaned_2 = list(map(clean_2, t_split))
                try:

                    return np.linspace(float(t_cleaned_2[0]), float(
                        t_cleaned_2[1]), int(t_cleaned_2[2]))
                except ValueError:
                    raise Exception("Please recheck your input for 'defevol'")
    elif var == 'y_opts':
        t_split = t.split()
        print('t_split', t_split)
        if ';' not in t:
            t_cleaned_1 = list(map(clean_1, t_split))
            try:
                return np.array(list(map(float, t_cleaned_1)))
            except ValueError:
                raise Exception("Please recheck your input for 'y_opts'")
        else:
            t_cleaned_2 = list(map(clean_2, t_split))
            try:

                return np.linspace(
                    float(t_cleaned_2[0]), float(t_cleaned_2[1]), int(t_cleaned_2[2]))
            except ValueError:
                raise Exception("Please recheck your input for 'y_opts'")


def assign_variables():
    global debug
    global fascal
    global defevol
    global delta0_opts
    global beam_loc
    global y_opts

    print(content_array)

    # assigning values from content_array indexes
    # https://stackoverflow.com/questions/20844347/how-would-i-make-a-custom-error-message-in-python
    if (content_array[0] != 'Off' and content_array[0] != 'On'):
        raise Exception("Please recheck your input for 'debug'")
    else:
        debug = content_array[0]

    if (content_array[1] != 'Off' and content_array[1] != 'On'):
        raise Exception("Please recheck your input for 'fascal'")
    else:
        fascal = content_array[1]

    if (content_array[2] != 'Off' and content_array[2] != 'On'):
        raise Exception("Please recheck your input for 'defevol'")
    else:
        defevol = content_array[2]

    delta0_opts = parse_array('delta0_opts', content_array[3])
    if type(delta0_opts) == str:
        delta0_opts = [delta0_opts]
    else:
        delta0_opts = delta0_opts * np.pi / 180

    try:
        beam_loc = float(content_array[4])
    except ValueError:
        raise Exception("Please recheck your input for 'beam_loc'")

    y_opts = parse_array('y_opts', content_array[5])
    print(y_opts)
    # y_opts = y_opts * np.pi / 180


file_read()
print('debug', debug)
print('fascal', fascal)
print('defevol', defevol)
print('delta0_opts', delta0_opts)
print('beam_loc', beam_loc)
print('y_opts', y_opts)


def diffEquationNumSolver(phi, y):
    u = y[0]
    w = y[1]

    dudphi = w
    dwdphi = (3 * (u * u)) - u

    return [dudphi, dwdphi]

# https://scicomp.stackexchange.com/questions/16325/dynamically-ending-ode-integration-in-scipy


def lmit_200(t, y):
    # when x' = 100 i.e. x * np.cos(theta) + y * np.sin(theta) = 100
    # i.e. r * np.cos(phi) * np.cos(theta) + r * np.sin(phi) * np.sin(theta) = 100
    # i.e. M/u * np.cos(phi) * np.cos(theta) + M/u * np.sin(phi) * np.sin(theta) = 100
    # <=> u + (M * np.cos(phi) * np.cos(theta) + M * np.sin(phi) * np.sin(theta))/-100 = 0
    global flag
    phi = t
    u = y[0]
    # fprint('x', (M/y[0]) * np.cos(t))
    if flag == 1:
        if y_beam >= 0:
            test = u + (M * np.cos(phi) * np.cos(theta) +
                        M * np.sin(phi) * np.sin(theta))/-200
        elif y_beam < 0:
            test = u + (M * np.cos(phi) * np.cos(theta) -
                        M * np.sin(phi) * np.sin(theta))/-200
        # test = y[0] + (M * np.cos(t))/200
        # test = u - M/200
        # test = np.ndarray.tolist(test)
        flag = 0
    else:
        # test = y[0] + (M * np.cos(t))/200
        # test = u - M/200
        if y_beam >= 0:
            test = u + (M * np.cos(phi) * np.cos(theta) +
                        M * np.sin(phi) * np.sin(theta))/-200
        elif y_beam < 0:
            test = u + (M * np.cos(phi) * np.cos(theta) -
                        M * np.sin(phi) * np.sin(theta))/-200
    return test


def limit_neg_100(t, y):
    # when x' = -100 i.e. x * np.cos(theta) + y * np.sin(theta) = -100
    # i.e. r * np.cos(phi) * np.cos(theta) + r * np.sin(phi) * np.sin(theta) = -100
    # i.e. M/u * np.cos(phi) * np.cos(theta) + M/u * np.sin(phi) * np.sin(theta) = -100
    # <=> u + (M * np.cos(phi) * np.cos(theta) + M * np.sin(phi) * np.sin(theta))/100 = 0
    global flag
    phi = t
    u = y[0]
    # fprint('x', (M/y[0]) * np.cos(t))
    if flag == 1:
        if y_beam >= 0:
            test = u + (M * np.cos(phi) * np.cos(theta) +
                        M * np.sin(phi) * np.sin(theta))/100
        elif y_beam < 0:
            test = u + (M * np.cos(phi) * np.cos(theta) -
                        M * np.sin(phi) * np.sin(theta))/100
        # test = y[0] + (M * np.cos(t))/100
        # test = u - M/200
        # test = np.ndarray.tolist(test)
        flag = 0
    else:
        # test = y[0] + (M * np.cos(t))/100
        # test = u - M/200
        if y_beam >= 0:
            test = u + (M * np.cos(phi) * np.cos(theta) +
                        M * np.sin(phi) * np.sin(theta))/100
        elif y_beam < 0:
            test = u + (M * np.cos(phi) * np.cos(theta) -
                        M * np.sin(phi) * np.sin(theta))/100
    return test


def limit_2(t, y):  # when r i.e. M/u = 2 <=> u - M/2 = 0
    global flag
    if flag == 1:
        test = y[0] - M/2
        # test = np.ndarray.tolist(test)
        flag = 0
    else:
        test = y[0] - M/2
    return test

# https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
# https://math.stackexchange.com/questions/1332311/finding-the-jacobian-of-a-system-of-1st-order-odes
# https://stackoverflow.com/questions/50599105/using-jacobian-matrix-in-solve-ivp


def jac(t, y):
    jac_mat = np.matrix([[0, 1], [(6 * y[0]), 0]])
    return jac_mat


# Define terminal condition and type-change flag
# limit_200.terminal = True
limit_neg_100.terminal = True
lmit_200.terminal = True
limit_2.terminal = True
global flag
flag = 1

# Interval of Integration
phi0 = 0
phif = 1000*np.pi


# Initial Constants:
M = 1
# delta0_opts = np.linspace(0, 180, 19) * np.pi / 180
# delta0_opts = [np.pi, np.pi/2]
# print('delta0_opts', delta0_opts / np.pi * 180)
# delta0_opts = np.linspace(120, 170, 6) * np.pi / 180
# delta0_opts = [170 * np.pi / 180]

# beam_loc = (100 - 1e-1)*M

# if you define this in the for loop only the last scatter will display
if defevol == 'On':
    fig, axs = plt.subplots(2)
else:
    fig, axs = plt.subplots(1)


plt.suptitle(
    f"Schwarzschild Ray Tracing {{(x, y) : x = {beam_loc}, y ∈ {content_array[5]}}}; \nEmission Angles = {content_array[3]}; Display -100 <= x <= 200; r >= 2M; \ndebug = {debug}; fascal = {fascal}; defevol = {defevol}")
# https://www.edureka.co/community/18327/nameerror-name-raw-input-is-not-defined#:~:text=There%20may%20a%20problem%20with%20your%20python%20version.&text=From%20Python3.,int()%20or%20float().
step = 0

if fascal == 'On':
    step = 1e-1
elif fascal == 'Off':
    step = np.inf


if debug == 'Off':
    for y_beam in y_opts:
        # print("y_beam", y_beam)
        r0 = np.sqrt(np.square(beam_loc) + np.square(y_beam))
        # for some reason not having negaitve makes the source in the negative of the y_beam param
        theta = -np.arctan(y_beam / beam_loc)
        # print('r0', r0)
        # print('theta', theta)
        for delta0 in delta0_opts:
            print('delta0', delta0)
            if delta0 == 'Parallel':
                delta0 = np.pi - np.abs(theta)
            print('delta0', delta0)

            B = r0 / np.sqrt(1 - (2 * M / r0))

            b = B * np.sin(delta0)

            escape_to_inf = False
            # if (r0 < 2):
            #     escape_to_inf = False
            if (2*M < r0 < 3*M) & (delta0 < 2*np.pi) & (np.sin(delta0) < 3*np.sqrt(2)*M/B):
                escape_to_inf = True
            elif (r0 == 3*M) & (delta0 < np.pi):
                escape_to_inf = True
            elif (r0 > 3*M) & (delta0 < np.pi/2 or delta0 > np.pi/2) & (np.sin(delta0) > 3*np.sqrt(2)*M/B):
                escape_to_inf = True

            print("escape_to_inf:", escape_to_inf)

            if escape_to_inf:
                if delta0 == 0:
                    m = (y_beam - 0) / (beam_loc - 0)
                    x = np.linspace(beam_loc, 200, 2)
                    y = m * x

                    axs[0].plot(x, y) if defevol == 'On' else axs.plot(x, y)
                else:
                    # Initial State
                    u0 = 1/r0
                    w0 = -u0/(np.tan(delta0))

                    y0 = [u0, w0]

                    sol = solve_ivp(diffEquationNumSolver, t_span=[
                        phi0, phif], y0=y0, method='LSODA', dense_output=True, events=(limit_neg_100, lmit_200, limit_2), jac=jac, max_step=step)

                    # print(sol)
                    u = sol.y[0]
                    w = sol.y[1]
                    phi = sol.t
                    # print('phi', phi)
                    # print(len(phi))

                    # print('delta', delta)

                    # convert polar to cartesian coordinates
                    r = M/u
                    # print("r: ", r)

                    x = r * np.cos(phi)
                    y = r * np.sin(phi)

                    # https://www.sparknotes.com/math/precalc/conicsections/section5/#:~:text=So%20far%2C%20we%20have%20only,%2B%20Ey%20%2B%20F%20%3D%200.&text=Figure%20%25%3A%20The%20x%20and,x%27%20and%20y%27%20axes.
                    x_prime = x * np.cos(theta) + y * np.sin(theta)
                    y_prime = -x * np.sin(theta) + y * np.cos(theta)

                    # need to reflect in the primed axis to get all delta 360 coverage otherwise reflection will be in the unprimed axis and will original from two different sources (half each)
                    minus_x_prime = x * np.cos(theta) + (-y) * np.sin(theta)
                    minus_y_prime = -x * np.sin(theta) + (-y) * np.cos(theta)

                    if y_beam >= 0:
                        # print(x_prime)
                        # print(-100 in x_prime)
                        if np.any(x_prime < -6):
                            if defevol == 'On':
                                axs[0].plot(x_prime, y_prime,
                                            label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")
                            else:
                                axs.plot(x_prime, y_prime,
                                         label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")

                            if defevol == 'On':
                                delta = (np.arctan(-u / w) / np.pi * 180)
                                # print('delta: ', delta)

                                # https://www.kite.com/python/answers/how-to-replace-elements-in-a-numpy-array-if-a-condition-is-met-in-python#:~:text=Use%20numpy.,that%20do%20not%20with%20y%20.
                                delta = np.where(delta < 0, 180 + delta, delta)

                                axs[1].plot(x_prime, delta,
                                            label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")
                    else:
                        if np.any(minus_x_prime < -6):
                            axs[0].plot(minus_x_prime, minus_y_prime,
                                        label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")

                            if defevol == 'On':
                                delta = (np.arctan(-u / w) / np.pi * 180)
                                # print('delta: ', delta)

                                # https://www.kite.com/python/answers/how-to-replace-elements-in-a-numpy-array-if-a-condition-is-met-in-python#:~:text=Use%20numpy.,that%20do%20not%20with%20y%20.
                                delta = np.where(delta < 0, 180 + delta, delta)
                                axs[1].plot(minus_x_prime, delta,
                                            label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")

                    # print('delta', delta)

                    axs[0].legend(loc='upper right') if defevol == 'On' else axs.legend(
                        loc='upper right')
elif debug == 'On':
    for y_beam in y_opts:
        # print("y_beam", y_beam)
        r0 = np.sqrt(np.square(beam_loc) + np.square(y_beam))
        # for some reason not having negative makes the source in the negative of the y_beam param
        theta = -np.arctan(y_beam / beam_loc)
        # print('r0', r0)
        # print('theta', theta)
        for delta0 in delta0_opts:
            if delta0 == 'Parallel':
                delta0 = np.pi - np.abs(theta)
            # print('delta0', delta0)

            if delta0 == 0:
                m = (y_beam - 0) / (beam_loc - 0)
                x = np.linspace(beam_loc, 200, 2)
                y = m * x

                axs[0].plot(x, y) if defevol == 'On' else axs.plot(x, y)
            else:
                # Initial State
                u0 = 1/r0
                w0 = -u0/(np.tan(delta0))

                y0 = [u0, w0]

                sol = solve_ivp(diffEquationNumSolver, t_span=[
                    phi0, phif], y0=y0, method='LSODA', dense_output=True, events=(limit_neg_100, lmit_200, limit_2), jac=jac, max_step=step)

                # print(sol)
                u = sol.y[0]
                w = sol.y[1]
                phi = sol.t
                # print('phi', phi)
                # print(len(phi))

                # convert polar to cartesian coordinates
                r = M/u
                # print("r: ", r)

                x = r * np.cos(phi)
                y = r * np.sin(phi)

                # https://www.sparknotes.com/math/precalc/conicsections/section5/#:~:text=So%20far%2C%20we%20have%20only,%2B%20Ey%20%2B%20F%20%3D%200.&text=Figure%20%25%3A%20The%20x%20and,x%27%20and%20y%27%20axes.
                x_prime = x * np.cos(theta) + y * np.sin(theta)
                y_prime = -x * np.sin(theta) + y * np.cos(theta)

                # need to reflect in the primed axis to get all delta 360 coverage otherwise reflection will be in the unprimed axis and will original from two different sources (half each)
                minus_x_prime = x * np.cos(theta) + (-y) * np.sin(theta)
                minus_y_prime = -x * np.sin(theta) + (-y) * np.cos(theta)
                # print('len', minus_x_prime)

                # print('hi')

                if y_beam >= 0:
                        # print(x_prime)
                        # print(-100 in x_prime)
                    # if np.any(x_prime < -6):
                    if defevol == 'On':
                        axs[0].plot(x_prime, y_prime,
                                    label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")
                    else:
                        axs.plot(x_prime, y_prime,
                                 label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")

                    if defevol == 'On':
                        delta = (np.arctan(-u / w) / np.pi * 180)
                        # print('delta: ', delta)

                        # https://www.kite.com/python/answers/how-to-replace-elements-in-a-numpy-array-if-a-condition-is-met-in-python#:~:text=Use%20numpy.,that%20do%20not%20with%20y%20.
                        delta = np.where(delta < 0, 180 + delta, delta)

                        axs[1].plot(x_prime, delta,
                                    label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")
                else:
                    # if np.any(minus_x_prime < -6):
                    if defevol == 'On':
                        axs[0].plot(minus_x_prime, minus_y_prime,
                                    label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")
                    else:
                        axs.plot(minus_x_prime, minus_y_prime,
                                 label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")

                    if defevol == 'On':
                        delta = (np.arctan(-u / w) / np.pi * 180)
                        # print('delta: ', delta)

                        # https://www.kite.com/python/answers/how-to-replace-elements-in-a-numpy-array-if-a-condition-is-met-in-python#:~:text=Use%20numpy.,that%20do%20not%20with%20y%20.
                        delta = np.where(
                            delta < 0, 180 + delta, delta)
                        axs[1].plot(minus_x_prime, delta,
                                    label="(" + str(round(y_beam, 2)) + ", " + str(round(beam_loc, 2)) + "); " + str(round(delta0 / np.pi * 180, 2)) + "°")

                axs[0].legend(loc='upper right') if defevol == 'On' else axs.legend(
                    loc='upper right')


axs[0].scatter([0], [0]) if defevol == 'On' else axs.scatter([0], [0])   # AGN
# axs[0].plot([beam_loc, beam_loc], [-100, 100])
axs[0].set_xlabel('x') if defevol == 'On' else axs.set_xlabel('x')
axs[0].set_ylabel('y') if defevol == 'On' else axs.set_ylabel('y')
# axs[0].set_zlabel('z')

# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
if defevol == 'On':
    axs[1].set_xlabel('phi')
    axs[1].set_ylabel('delta')

# plt.gca().set_aspect('equal', adjustable='box')

plt.show()
