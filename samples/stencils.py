# Stencils in GT4Py syntax
from math import sqrt
import numpy as np
from gt4py import computation, interval, PARALLEL, FORWARD, BACKWARD

Field3D = np.ndarray


def one_d_diff(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = 2 * in_field[0, 0, 0] - in_field[-1, 0, 0]

def laplacian(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = - 4.0 * in_field[0, 0, 0] + (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0])

def pure_vertical(in_field: Field3D, out_field: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            in_field = in_field[0, 0, 0]
        with interval(1, None):
            in_field = in_field[0, 0, 0] - 0.5 * in_field[0, 0, -1]


# See https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
def vertical_advection(
    utens_stage: Field3D,
    u_stage: Field3D,
    wcon: Field3D,
    u_pos: Field3D,
    utens: Field3D,
    dtr_stage: float,
):
    with computation(FORWARD):
        with interval(0, 1):
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
            cs = gcv * 0.5  # = BET_M

            ccol = gcv * 0.5  # = BET_P
            bcol = dtr_stage - ccol[0, 0, 0]

            # update the d column
            correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term)

            # Thomas forward
            divided = 1.0 / bcol[0, 0, 0]
            ccol = ccol[0, 0, 0] * divided
            dcol = dcol[0, 0, 0] * divided

        with interval(1, -1):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

            as_ = gav * 0.5  # = BET_M
            cs = gcv * 0.5  # = BET_M

            acol = gav * 0.5  # = BET_P
            ccol = gcv * 0.5  # = BET_P
            bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0]) - cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term)

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            ccol = ccol[0, 0, 0] * divided
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with interval(-1, None):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            as_ = gav * 0.5  # = BET_M
            acol = gav * 0.5  # = BET_P
            bcol = dtr_stage - acol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
            dcol = (dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term)

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

    with computation(BACKWARD):
        with interval(-1, None):
            datacol = dcol[0, 0, 0]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

        with interval(0, -1):
            datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])



def uvbke(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D, arg5: Field3D):
    with computation(PARALLEL), interval(...):
        i16 = (arg1[-1, 0, 0] + arg1[0, 0, 0]) * arg2[0, 0, 0]
        i19 = arg0[0, -1, 0] + arg0[0, 0, 0]
        i21 = 112.5 * (i19 - i16)
        arg4 = arg3[0, 0, 0] * i21

        i16 = (arg0[0, -1, 0] + arg0[0, 0, 0]) * arg2[0, 0, 0]
        i19 = arg1[-1, 0, 0] + arg1[0, 0, 0]
        i21 = 112.5 * (i19 - i16)
        arg5 = arg3[0, 0, 0] * i21
