"""
Corpus of (stencil, domain size) pairs for regression-testing the stencil-to-spatial lowering.

Usage as a script: ``python -m tests.stencil_ir.parametric.corpus <outdir>`` lowers every corpus entry
with the current lowering and writes one ``.sptl`` file and one ``.blocks.json`` block summary per
entry (or a ``.err`` file with the exception if lowering fails). Two such directories can be
compared with ``compare.py``.
"""

import contextlib
import io
import json
import multiprocessing
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

GT4PY_SOURCES = {
    "stencils": REPO / "samples" / "stencils.py",
    "test_instances": REPO / "samples" / "gt4py_test_instances.py",
    "advanced": REPO / "samples" / "advanced_stencils.py",
    "corpus": Path(__file__).resolve().parent / "corpus_stencils.py",
}

SIZES = [
    (4, 4, 4),
    (5, 4, 3),
    (4, 5, 4),
    (5, 5, 5),
    (6, 7, 5),
    (8, 8, 4),
    (9, 8, 6),
    (8, 9, 7),
    (16, 16, 4),
    (16, 17, 4),
    (17, 16, 5),
    (17, 17, 6),
    (33, 20, 10),
]


def corpus_entries() -> list[tuple[str, str, tuple[int, int, int]]]:
    """
    Enumerate the corpus.

    :return: A list of ``(source key, function name, domain size)`` triples.
    """
    from spada.syntax.gt4py import parser

    entries = []
    for key, path in GT4PY_SOURCES.items():
        for name in parser.parse_file(str(path)).keys():
            for size in SIZES:
                entries.append((key, name, size))
    return entries


def lower_entry(key: str, name: str, size: tuple[int, int, int]):
    """
    Lower one corpus entry with the concrete (integer-size) pipeline of ``gt4py_to_spatial``.

    :param key: Source key in ``GT4PY_SOURCES``.
    :param name: GT4Py function name.
    :param size: Domain size ``(x, y, z)``.
    :return: The Spatial IR kernel.
    """
    from spada.syntax.gt4py import parser
    from spada.lowering import gt4py_to_stencil_ir
    from spada.lowering.stencil_to_spatial import lower_stencil_to_spatial
    from spada.syntax.stencil_ir import type_inference

    program = parser.parse_file(str(GT4PY_SOURCES[key]))[name]
    with contextlib.redirect_stdout(io.StringIO()):
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program, domain=size)
        type_inference.infer_field_extents(irprogram)
        type_inference.infer_field_domains(irprogram)
        kernel = lower_stencil_to_spatial(irprogram)
    return kernel


def entry_filename(key: str, name: str, size: tuple[int, int, int]) -> str:
    return f"{key}__{name}__{size[0]}_{size[1]}_{size[2]}"


def _work(args):
    key, name, size, outdir = args
    base = Path(outdir) / entry_filename(key, name, size)
    try:
        from tests.stencil_ir.parametric.compare import block_summary

        kernel = lower_entry(key, name, size)
        base.with_suffix(".sptl").write_text(kernel.as_ir())
        base.with_name(base.name + ".blocks.json").write_text(
            json.dumps(block_summary(kernel))
        )
        return base.name, True
    except Exception:
        base.with_suffix(".err").write_text(
            traceback.format_exc().splitlines()[-1] + "\n"
        )
        return base.name, False


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    work = [(k, n, s, str(outdir)) for k, n, s in corpus_entries()]
    with multiprocessing.Pool() as pool:
        results = pool.map(_work, work)
    failed = [name for name, ok in results if not ok]
    print(f"{len(results) - len(failed)} lowered, {len(failed)} failed")


if __name__ == "__main__":
    main()
