"""
Parametric (domain-size independent) lowering from GT4Py to spatial IR.

The lowering runs the integer pipeline on the symbolic domain ``(I, J, K)``, represented by anchored
coordinates (:class:`~spada.syntax.common.symbolic.Coord`). Every decision that depends on the domain
size is either independent of it or valid for all sizes above a bound, which is recorded in a
:class:`~spada.syntax.common.symbolic.ConstraintLog`. The result is a kernel with parameters
``I, J, K`` and the minimum sizes for which it is valid.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass

import spada.syntax.spatial_ir.irnodes as spa
from spada.syntax.common.symbolic import ConstraintLog, Coord

PARAMETERS = ("I", "J", "K")

PRECONDITION_PREFIX = "// spada-precondition:"
_BOUND = re.compile(r"^([A-Za-z_]\w*)>=(-?\d+)$")


def parse_precondition(source: str) -> dict[str, int] | None:
    """
    Reads the precondition header of a spatial IR source file.

    The parser ignores comments, so this inspects the text directly.

    :param source: The spatial IR source text.
    :return: The minimum value of each parameter, or None if the file has no precondition header.
    :raises ValueError: If a header line is malformed.
    """
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith(PRECONDITION_PREFIX):
            continue
        bounds = {}
        for term in stripped[len(PRECONDITION_PREFIX) :].split():
            match = _BOUND.match(term)
            if match is None:
                raise ValueError(f"Malformed precondition term {term!r} in: {stripped}")
            bounds[match.group(1)] = int(match.group(2))
        return bounds
    return None


def check_precondition(bounds: dict[str, int], values: dict[str, int]) -> None:
    """
    Checks concrete parameter values against a precondition.

    :param bounds: The minimum value of each parameter (see :func:`parse_precondition`).
    :param values: The concrete parameter values.
    :raises ValueError: If a value is missing or below its bound.
    """
    violated = [
        f"{p}={values.get(p)} (requires {p}>={b})"
        for p, b in bounds.items()
        if not isinstance(values.get(p), int) or values[p] < b
    ]
    if violated:
        raise ValueError(
            "Parameter values violate the kernel precondition: "
            + ", ".join(violated)
            + ". Lower the stencil for this size with concrete dimensions instead."
        )


@dataclass
class ParametricKernel:
    """
    A parametric spatial IR kernel together with its admissible domain sizes.
    """

    kernel: spa.Kernel
    #: Minimum value of each parameter for which the kernel is valid.
    bounds: dict[str, int]

    def admits(self, size: tuple[int, int, int]) -> bool:
        """
        :param size: A concrete domain size ``(x, y, z)``.
        :return: Whether the kernel is valid for this size.
        """
        return all(v >= self.bounds.get(p, 1) for p, v in zip(PARAMETERS, size))

    def precondition(self) -> str:
        """
        :return: The precondition header line, e.g. ``// spada-precondition: I>=3 J>=3 K>=1``.
        """
        terms = " ".join(f"{p}>={self.bounds.get(p, 1)}" for p in PARAMETERS)
        return f"{PRECONDITION_PREFIX} {terms}"

    def as_ir(self) -> str:
        """
        :return: The kernel source, starting with the precondition header.
        """
        return f"{self.precondition()}\n{self.kernel.as_ir()}"

    def sidecar(self) -> str:
        """
        :return: The JSON sidecar describing the parameters and their minimum values.
        """
        return json.dumps(
            {
                "kernel": self.kernel.name,
                "parameters": list(PARAMETERS),
                "min": {p: self.bounds.get(p, 1) for p in PARAMETERS},
            },
            indent=2,
        )


def symbolic_domain() -> tuple[Coord, Coord, Coord]:
    """
    :return: The symbolic domain size ``(I, J, K)``.
    """
    return tuple(Coord(p) for p in PARAMETERS)


def lower_gt4py_parametric(
    program, verbose: bool = False, **kwargs
) -> ParametricKernel:
    """
    Lowers a GT4Py stencil to a parametric spatial IR kernel.

    :param program: The parsed GT4Py function (see ``spada.syntax.gt4py.parser``).
    :param verbose: Whether to keep the diagnostic output of the lowering passes.
    :param kwargs: Additional keyword arguments for ``lower_stencil_to_spatial``.
    :return: The kernel and the minimum domain sizes for which it is valid.
    :raises SymbolicCoordError: If the lowering makes a decision that depends on the domain size in a
                                way that is not monotone (e.g., on its parity).
    """
    from spada.lowering import gt4py_to_stencil_ir
    from spada.lowering.stencil_to_spatial import lower_stencil_to_spatial
    from spada.syntax.stencil_ir import type_inference

    out = (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    with ConstraintLog() as log, out:
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(
            program, domain=symbolic_domain()
        )
        type_inference.infer_field_extents(irprogram)
        type_inference.infer_field_domains(irprogram)
        kernel = lower_stencil_to_spatial(
            irprogram, parameters=list(PARAMETERS), **kwargs
        )
    bounds = {p: max(1, log.bounds.get(p, 1)) for p in PARAMETERS}
    return ParametricKernel(kernel, bounds)
