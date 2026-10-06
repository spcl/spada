"""
NumPy references for the vertical (FORWARD/BACKWARD) stencils in ``samples/gt4py_test_instances.py``
and ``samples/stencils.py``.

Usage: python vertical_stencils.py <kernel> <input .npy files in kernel argument order> -o <output .npy>
"""

import argparse

import numpy as np


def pure_vertical(in_field):
    out = in_field.copy()
    for k in range(1, out.shape[2]):
        out[:, :, k] = out[:, :, k] - 0.5 * out[:, :, k - 1]
    return out


def forward_partial_inout(a):
    return pure_vertical(a)


def forward_lookahead_inout(a, b):
    out = a.copy()
    for k in range(a.shape[2] - 1):
        out[:, :, k] = 0.5 * (a[:, :, k + 1] + out[:, :, k]) + b[:, :, k]
    return out


def forward_carried_temporary(a):
    t = np.empty_like(a)
    out = np.empty_like(a)
    t[:, :, 0] = a[:, :, 0]
    out[:, :, 0] = t[:, :, 0]
    for k in range(1, a.shape[2]):
        t[:, :, k] = 2.0 * a[:, :, k]
        out[:, :, k] = t[:, :, k] + t[:, :, k - 1]
    return out


def backward_carried_after_write(a, acc):
    acc = np.empty_like(a)
    acc[:, :, -1] = a[:, :, -1]
    for k in range(a.shape[2] - 2, -1, -1):
        acc[:, :, k] = 2.0 * a[:, :, k]
        acc[:, :, k] = acc[:, :, k] + 0.5 * acc[:, :, k + 1]
    return acc


KERNELS = {
    "pure_vertical": pure_vertical,
    "pure_vertical_test": pure_vertical,
    "forward_partial_inout": forward_partial_inout,
    "forward_lookahead_inout": forward_lookahead_inout,
    "forward_carried_temporary": forward_carried_temporary,
    "backward_carried_after_write": backward_carried_after_write,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vertical stencil references")
    parser.add_argument("kernel", choices=KERNELS.keys())
    parser.add_argument("inputs", nargs="+", type=str, help="Input file paths")
    parser.add_argument(
        "--output", "-o", type=str, default="vertical_out.npy", help="Output file path"
    )
    args = parser.parse_args()

    inputs = [np.load(f) for f in args.inputs]
    np.save(args.output, KERNELS[args.kernel](*inputs))
