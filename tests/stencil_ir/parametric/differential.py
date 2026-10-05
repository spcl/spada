"""
Differential test of the parametric lowering against the concrete (integer) lowering.

For every corpus stencil, the parametric kernel is lowered once. For every corpus size it admits,
the kernel is specialized with the sptlc front-end passes (parameter concretization, constant
folding, metaprogramming expansion) and compared with the integer lowering at that size. The
comparison is textual; if the texts differ, the per-PE semantic comparison is reported as well.

Usage: ``python3 -m tests.stencil_ir.parametric.differential [<report file>]``.
"""

import contextlib
import copy
import io
import multiprocessing
import sys
import traceback

from tests.stencil_ir.parametric.corpus import (
    GT4PY_SOURCES,
    SIZES,
    corpus_entries,
    lower_entry,
)


def specialize(kernel, size: tuple[int, int, int]):
    """
    Specializes a parametric kernel to a concrete domain size, as sptlc does before lowering.

    :param kernel: A parametric spatial IR kernel with parameters ``I, J, K``.
    :param size: The domain size ``(x, y, z)``.
    :return: A concrete kernel without parameters or metaprogramming blocks.
    """
    from spada.syntax.spatial_ir import canonicalization, passes

    kernel = copy.deepcopy(kernel)
    kernel = passes.concretize_parameters(kernel, I=size[0], J=size[1], K=size[2])
    kernel = passes.constexpr_propagation(kernel)
    return canonicalization.inline_metaprogramming(kernel)


def check_stencil(args: tuple[str, str]) -> list[tuple[str, tuple, str]]:
    """
    Runs the differential test for one stencil over all corpus sizes.

    :param args: ``(source key, function name)``.
    :return: One ``(name, size, status)`` triple per size, where status is ``identical``,
             ``equivalent`` (same per-PE semantics, different text), ``skipped`` (size not admitted
             or concrete lowering fails), ``differs: ...``, or ``parametric failed: ...``.
    """
    from spada.lowering.parametric import lower_gt4py_parametric
    from spada.syntax.gt4py import parser
    from tests.stencil_ir.parametric.compare import block_summary, compare_summaries

    key, name = args
    label = f"{key}__{name}"
    program = parser.parse_file(str(GT4PY_SOURCES[key]))[name]
    try:
        parametric = lower_gt4py_parametric(program)
    except Exception:
        reason = traceback.format_exc().splitlines()[-1]
        return [(label, size, f"parametric failed: {reason}") for size in SIZES]

    b = tuple(parametric.bounds[p] for p in ("I", "J", "K"))
    boundary = [
        b,
        (b[0] + 1, b[1], b[2]),
        (b[0], b[1] + 1, b[2]),
        (b[0], b[1], b[2] + 1),
        (b[0] + 1, b[1] + 1, b[2] + 1),
    ]
    results = []
    for size in SIZES + [s for s in boundary if s not in SIZES]:
        if not parametric.admits(size):
            results.append(
                (
                    label,
                    size,
                    f"skipped (precondition {parametric.precondition()[24:]})",
                )
            )
            continue
        try:
            concrete = lower_entry(key, name, size)
        except Exception:
            results.append((label, size, "skipped (concrete lowering fails)"))
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            special = specialize(parametric.kernel, size)
        if special.as_ir() == concrete.as_ir():
            results.append((label, size, "identical"))
            continue
        problems = compare_summaries(block_summary(concrete), block_summary(special))
        if problems:
            results.append((label, size, "differs: " + " | ".join(problems[:3])))
        else:
            results.append((label, size, "equivalent"))
    return results


def main(report: str | None = None) -> int:
    stencils = sorted({(k, n) for k, n, _ in corpus_entries()})
    with multiprocessing.Pool() as pool:
        results = [r for rs in pool.map(check_stencil, stencils) for r in rs]
    lines = [f"{label} {size}: {status}" for label, size, status in results]
    if report:
        with open(report, "w") as f:
            f.write("\n".join(lines) + "\n")
    counts: dict[str, int] = {}
    for _, _, status in results:
        kind = status.split(":")[0].split(" (")[0]
        counts[kind] = counts.get(kind, 0) + 1
    for line in lines:
        if line.split(": ", 1)[1].startswith(("differs", "parametric failed")):
            print(line[:400])
    print(", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 1 if counts.get("differs") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
