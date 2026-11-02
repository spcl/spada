# Additional GT4Py stencils (parsed by spada.syntax.gt4py, never executed) that stress the
# geometry of the stencil-to-spatial lowering: asymmetric halos, chained offsets, single-axis
# communication, mixed horizontal/vertical computations, and multiple outputs.
import numpy as np
from gt4py import computation, interval, PARALLEL, FORWARD, BACKWARD

Field3D = np.ndarray


def east_only(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[1, 0, 0] - in_field[0, 0, 0]


def north_only(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[0, 1, 0] + in_field[0, 0, 0]


def west_south(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[-1, 0, 0] + in_field[0, -1, 0]


def x_diffusion(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[1, 0, 0] - 2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0]


def y_diffusion(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = in_field[0, 1, 0] - 2.0 * in_field[0, 0, 0] + in_field[0, -1, 0]


def chained_east(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = in_field[1, 0, 0] + in_field[0, 0, 0]
        out_field = tmp[1, 0, 0] - tmp[0, 0, 0]


def chained_laplacian(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        lap = -4.0 * in_field[0, 0, 0] + (
            in_field[1, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[0, 1, 0]
            + in_field[0, -1, 0]
        )
        out_field = -4.0 * lap[0, 0, 0] + (
            lap[1, 0, 0] + lap[-1, 0, 0] + lap[0, 1, 0] + lap[0, -1, 0]
        )


def two_outputs(in_field: Field3D, out_a: Field3D, out_b: Field3D):
    with computation(PARALLEL), interval(...):
        out_a = in_field[1, 0, 0] + in_field[0, 0, 0]
        out_b = in_field[0, -1, 0] * in_field[0, 0, 0]


def scaled_laplacian(in_field: Field3D, alpha: float, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = alpha * (
            -4.0 * in_field[0, 0, 0]
            + in_field[1, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[0, 1, 0]
            + in_field[0, -1, 0]
        )


def vertical_intervals(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL):
        with interval(0, 1):
            out_field = in_field[0, 0, 0]
        with interval(1, -1):
            out_field = in_field[0, 0, 1] + in_field[0, 0, -1]
        with interval(-1, None):
            out_field = 2.0 * in_field[0, 0, 0]


def last_level(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(-1, None):
        out_field = in_field[0, 0, 0] + in_field[1, 0, 0]


def horizontal_then_forward(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = in_field[1, 0, 0] + in_field[0, 1, 0]
    with computation(FORWARD):
        with interval(0, 1):
            out_field = tmp[0, 0, 0]
        with interval(1, None):
            out_field = tmp[0, 0, 0] + 0.5 * out_field[0, 0, -1]


def backward_with_halo(a: Field3D, b: Field3D, out: Field3D):
    with computation(BACKWARD):
        with interval(-1, None):
            out = a[0, 0, 0] + b[-1, 0, 0]
        with interval(0, -1):
            out = a[0, 0, 0] + 0.5 * out[0, 0, 1]


def fw_lookahead(a: Field3D, b: Field3D):
    with computation(FORWARD), interval(0, -1):
        a = 0.5 * (a[0, 0, 1] + a[0, 0, 0]) + b[0, 0, 0]


def fw_partial_inout(a: Field3D):
    with computation(FORWARD), interval(1, None):
        a = a[0, 0, 0] - 0.5 * a[0, 0, -1]


def bw_carried(a: Field3D, acc: Field3D):
    with computation(BACKWARD):
        with interval(-1, None):
            acc = a[0, 0, 0]
        with interval(0, -1):
            acc = 2.0 * a[0, 0, 0]
            acc = acc[0, 0, 0] + 0.5 * acc[0, 0, 1]


def fw_temp_carried(a: Field3D, out: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            t = a[0, 0, 0]
            out = t[0, 0, 0]
        with interval(1, None):
            t = 2.0 * a[0, 0, 0]
            out = t[0, 0, 0] + t[0, 0, -1]
