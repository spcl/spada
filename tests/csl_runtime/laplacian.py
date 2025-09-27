import argparse
import numpy as np
import numpy.typing as npt


def laplacian(x: npt.ArrayLike) -> npt.ArrayLike:
    """
    Implements the Laplace operator.
    """
    return x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2] - 4 * x[1:-1, 1:-1]


def main(args):
    x = np.load(args.input)
    result = laplacian(x)
    np.save(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laplacian operator")
    parser.add_argument("input", type=str, help="Input file path")
    parser.add_argument("--output", "-o", type=str, default='lap_out.npy', help="Output file path")
    args = parser.parse_args()

    main(args)
