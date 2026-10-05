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


# Level 0 is not written and must keep its input value; level k - 1 must be read after its update.
def forward_partial_inout(a: Field3D):
    with computation(FORWARD), interval(1, None):
        a = a[0, 0, 0] - 0.5 * a[0, 0, -1]


# Level k + 1 has not been updated yet, so the input value is read; the top level keeps its input value.
def forward_lookahead_inout(a: Field3D, b: Field3D):
    with computation(FORWARD), interval(0, -1):
        a = 0.5 * (a[0, 0, 1] + a[0, 0, 0]) + b[0, 0, 0]


# A temporary read at k - 1 after being rewritten in the same iteration; at k = 1 the value comes from the
# previous interval.
def forward_carried_temporary(a: Field3D, out: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            t = a[0, 0, 0]
            out = t[0, 0, 0]
        with interval(1, None):
            t = 2.0 * a[0, 0, 0]
            out = t[0, 0, 0] + t[0, 0, -1]


# The Thomas-sweep pattern of vertical advection: k + 1 must observe the final value of the previous iteration,
# not the intermediate value assigned earlier in the current iteration.
def backward_carried_after_write(a: Field3D, acc: Field3D):
    with computation(BACKWARD):
        with interval(-1, None):
            acc = a[0, 0, 0]
        with interval(0, -1):
            acc = 2.0 * a[0, 0, 0]
            acc = acc[0, 0, 0] + 0.5 * acc[0, 0, 1]
