import functools
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import takewhile
from array import array
import re
import math
import os

# global variables
content_array = []


# Getting inputs from input file
my_path = os.path.dirname(__file__)


def file_read():
    with open(my_path + "/input.txt") as f:
        for line in f:
            if not line.startswith("#") and not line == "\n" and not line.startswith("=>"):
                content_array.append((line.rstrip()).lstrip())

    if (len(content_array) < 8 or len(content_array) > 8):
        print("Error: Please provide 8 inputs")
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
    global graphing
    global defevol
    global delta0_opts
    global screen_loc
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
        raise Exception("Please recheck your input for 'graphing'")
    else:
        graphing = content_array[2]

    if (content_array[3] != 'Off' and content_array[3] != 'On'):
        raise Exception("Please recheck your input for 'defevol'")
    else:
        defevol = content_array[3]

    delta0_opts = parse_array('delta0_opts', content_array[4])
    if type(delta0_opts) == str:
        delta0_opts = [delta0_opts]
    else:
        delta0_opts = delta0_opts * np.pi / 180

    try:
        beam_loc = float(content_array[5])
    except ValueError:
        raise Exception("Please recheck your input for 'beam_loc'")

    try:
        screen_loc = float(content_array[6])
    except ValueError:
        raise Exception("Please recheck your input for 'screen_loc'")

    y_opts = parse_array('y_opts', content_array[7])
    # print(y_opts)
    # y_opts = y_opts * np.pi / 180


def get_constraints_from_file():
    file_read()

    print('debug', debug)
    print('fascal', fascal)
    print('graphing', graphing)
    print('defevol', defevol)
    print('delta0_opts', delta0_opts)
    print('beam_loc', beam_loc)
    print('screen_loc', screen_loc)
    print('y_opts', y_opts)

    return [debug, fascal, graphing, defevol, delta0_opts, screen_loc, beam_loc, y_opts]
