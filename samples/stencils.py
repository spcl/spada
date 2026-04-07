# Stencils in GT4Py syntax
from math import sqrt
import numpy as np

# from gt4py.cartesian import computation, interval, PARALLEL, FORWARD, BACKWARD
import gt4py.storage
import gt4py.cartesian.gtscript as gtscript
import sys

dtype = np.float32
Field3D = gtscript.Field[dtype]
backend = "gt:gpu"


def one_d_diff(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = 2 * in_field[0, 0, 0] - in_field[-1, 0, 0]


@gtscript.stencil(backend)
def laplacian(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = -4.0 * in_field[0, 0, 0] + (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
        )


@gtscript.stencil(backend)
def pure_vertical(in_field: Field3D, out_field: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            in_field = in_field[0, 0, 0]
        with interval(1, None):
            in_field = in_field[0, 0, 0] - 0.5 * in_field[0, 0, -1]


# See https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
@gtscript.stencil(backend)
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
            dcol = dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term

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
            dcol = dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term

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
            dcol = dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term

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


@gtscript.stencil(backend)
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


if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Callable

    @dataclass
    class Kernel:
        func: Callable
        nargs: int
        origin: tuple[int, int, int]

    kernels = {
        "laplacian": Kernel(laplacian, 2, (1, 1, 0)),
        "pure_vertical": Kernel(pure_vertical, 2, (0, 0, 0)),
        "uvbke": Kernel(uvbke, 6, (1, 1, 0)),
        "vertical_advection": Kernel(vertical_advection, 5, (1, 1, 0)),
    }
    if len(sys.argv) < 2:
        print("USAGE: python stencils.py <KERNEL> [HORIZONTAL POINTS] [VERTICAL POINTS]")
        print(f"Allowed kernels: {kernels.keys()}")
        exit(1)
    if sys.argv[1] not in kernels:
        print(f"Allowed kernels: {kernels.keys()}")
        exit(1)
    kernel = kernels[sys.argv[1]]
    N = int(sys.argv[2]) if len(sys.argv) >= 3 else 512
    K = int(sys.argv[3]) if len(sys.argv) >= 4 else 80
    shape = (N, N, K)
    origin = kernel.origin

    print(f"Running {sys.argv[1]} with shape {shape} and dtype {dtype}")

    indices = np.arange(N)
    kindices = np.arange(K)
    ii = np.zeros((N, N, K)) + np.reshape(indices, (N, 1, 1))
    jj = np.zeros((N, N, K)) + np.reshape(indices, (1, N, 1))
    kk = np.zeros((N, N, K)) + np.reshape(kindices, (1, 1, K))

    xx = ii / N
    yy = jj / N
    zz = kk / N

    in_data = 5.0 + 8.0 * (2.0 + np.cos(np.pi * (xx + 1.5 * yy)) + np.sin(2 * np.pi * (xx + 1.5 * yy))) / 4.0
    data = [np.copy(in_data) for _ in range(kernel.nargs)]
    storage = [gt4py.storage.from_array(d, dtype, backend=backend, aligned_index=origin) for d in data]

    for _ in range(1000):
        if sys.argv[1] == "vertical_advection":
            kernel.func(*storage, 2.0, origin=origin)
        else:
            kernel.func(*storage, origin=origin)
