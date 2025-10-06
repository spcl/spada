# Stencils in GT4Py syntax
from math import sqrt
import numpy as np
from gt4py import computation, interval, PARALLEL, FORWARD, BACKWARD

Field3D = np.ndarray

def scalar_arg_f(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = factor * in_field[0, 0, 0] - in_field[1, 0, 0]

def scalar_arg_g(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = factor * in_field[0, 0, 0]

def scalar_arg_h(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[1, 0, 0] + in_field[0, 0, 0] * factor

def scalar_arg_l(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[0, 0, 0] * factor

def scalar_arg_k(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = factor
        
def scalar_arg_i(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = factor * in_field[1, 0, 0]
        
def scalar_arg_j(in_field: Field3D, factor: float, out_field: Field3D):
    with computation(FORWARD), interval(1, None):
        out_field = factor * in_field[0, 0, -1]
        
        
def pure_vertical_test(in_field: Field3D, out_field: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            in_field = in_field[0, 0, 0]
        with interval(1, None):
            in_field = in_field[0, 0, 0] - 0.5 * in_field[0, 0, -1]
