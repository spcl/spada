"""
Independent checks of the parametric and region-based stencil lowering (plan section 4).
"""

import contextlib
import functools
import io
import os
import random
import re
import subprocess
import sys

import pytest

from spada.lowering.parametric import lower_gt4py_parametric
from spada.lowering.regions import partition
from spada.syntax.common.symbolic import ConstraintLog, Coord, SymbolicCoordError
from spada.syntax.gt4py import parser
from spada.syntax.spatial_ir.grid_geometry import Rectangle
from tests.stencil_ir.parametric.compare import _per_pe, block_summary
from tests.stencil_ir.parametric.corpus import GT4PY_SOURCES, REPO, lower_entry
from tests.stencil_ir.parametric.differential import specialize

STENCILS = [
    ("stencils", "laplacian"),
    ("corpus", "east_only"),
    ("corpus", "west_south"),
    ("corpus", "chained_laplacian"),
    ("corpus", "two_outputs"),
    ("corpus", "backward_with_halo"),
    ("advanced", "horizontal_diffusion"),
]
SIZES = [(4, 4, 4), (5, 5, 5), (16, 16, 4), (16, 17, 4), (17, 16, 5), (17, 17, 6)]

_DECL = re.compile(r"stream<[^>]*> ([\w#]+) = relative_stream\((-?\d+), (-?\d+)\)")
_SEND = re.compile(r"\bsend\([^,()]+, ([\w#]+)\)")
_RECV = re.compile(r"\breceive\((?:[^,()]+, )?([\w#]+)\)")


@functools.lru_cache(maxsize=None)
def _parametric(key, name):
    return lower_gt4py_parametric(parser.parse_file(str(GT4PY_SOURCES[key]))[name])


def _specialized(key, name, size):
    with contextlib.redirect_stdout(io.StringIO()):
        return specialize(_parametric(key, name).kernel, size)


def _kernels(key, name, size):
    yield "concrete", lower_entry(key, name, size)
    yield "parametric", _specialized(key, name, size)


def _stream_use(summary):
    """
    :return: Per PE, the stream declarations ``name -> (dx, dy)``, the sets of sent / received
             stream names, and the checkerboard pairs (adjacent declarations of the same field and
             offset, i.e. the even and odd version of one stream).
    """
    result = {}
    for pe, (_, dataflow, compute) in _per_pe(summary).items():
        decls = {}
        for stmt in dataflow:
            for m in _DECL.finditer(stmt):
                decls[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        names = list(decls)
        pairs = {
            frozenset((a, b))
            for a, b in zip(names, names[1:])
            if a.split("#")[0] == b.split("#")[0] and decls[a] == decls[b]
        }
        text = "\n".join(s for block in compute for s in block)
        sends = {n for n in _SEND.findall(text) if n in decls}
        recvs = {n for n in _RECV.findall(text) if n in decls}
        result[pe] = (decls, sends, recvs, pairs)
    return result


@pytest.mark.parametrize("stencil", STENCILS, ids=lambda s: s[1])
def test_send_receive_consistency(stencil):
    """A3: every send on a relative stream is received on the same stream by the target PE, and vice versa."""
    for size in SIZES:
        for kind, kernel in _kernels(*stencil, size):
            use = _stream_use(block_summary(kernel))
            assert (
                any(sends for _, sends, _, _ in use.values())
                or stencil[1] == "backward_with_halo"
            )
            for (x, y), (decls, sends, recvs, pairs) in use.items():
                for name in sends:
                    dx, dy = decls[name]
                    target = use.get((x + dx, y + dy))
                    assert target and name in target[2], (
                        f"{kind} {size}: {name} sent at {(x, y)} not received at {(x + dx, y + dy)}"
                    )
                for name in recvs:
                    dx, dy = decls[name]
                    source = use.get((x - dx, y - dy))
                    assert source and name in source[1], (
                        f"{kind} {size}: {name} received at {(x, y)} not sent from {(x - dx, y - dy)}"
                    )
                for direction in (sends, recvs):
                    for pair in pairs:
                        assert not pair <= direction, (
                            f"{kind} {size}: PE {(x, y)} uses both checkerboard versions {sorted(pair)} in one direction"
                        )
            assert any(pairs for *_, pairs in use.values()) or stencil[1] in (
                "backward_with_halo",
            )


@pytest.mark.parametrize("stencil", STENCILS, ids=lambda s: s[1])
def test_guard_variables_do_not_leak(stencil):
    """A7: guard variables only occur in guard headers, are distinct when nested, and vanish after specialization."""
    text = _parametric(*stencil).kernel.as_ir()
    for line in text.splitlines():
        if "__parity_" in line:
            assert re.fullmatch(r"\s*for u16 __parity_[xy] in \[0:.*\] \{", line), line
    for nested in re.findall(
        r"for u16 (__parity_\w+) in [^\n]*\n\s*for u16 (__parity_\w+)", text
    ):
        assert nested[0] != nested[1]
    special = _specialized(*stencil, (16, 17, 4)).as_ir()
    assert "__parity_" not in special and "for u16 __" not in special


_DETERMINISM_SCRIPT = """
import contextlib, hashlib, io
from spada.syntax.gt4py import parser
from spada.lowering.parametric import lower_gt4py_parametric
from tests.stencil_ir.parametric.corpus import GT4PY_SOURCES, lower_entry
out = []
for key, name in [("stencils", "laplacian"), ("corpus", "backward_with_halo"), ("corpus", "two_outputs")]:
    with contextlib.redirect_stdout(io.StringIO()):
        out.append(lower_gt4py_parametric(parser.parse_file(str(GT4PY_SOURCES[key]))[name]).as_ir())
        out.append(lower_entry(key, name, (16, 17, 4)).as_ir())
print(hashlib.sha256("\\n".join(out).encode()).hexdigest())
"""


def test_determinism_across_hash_seeds():
    """B14: output does not depend on string hashing."""
    digests = set()
    for seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO))
        run = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT],
            env=env,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(run.stdout.strip())
    assert len(digests) == 1


@pytest.mark.parametrize(
    "kind,size",
    [("concrete", (4, 4, 4)), ("parametric", (5, 4, 3)), ("symbolic", None)],
)
def test_parser_round_trip(kind, size):
    """R1: printing, parsing and printing again is the identity on lowering outputs."""
    from spada.syntax.spatial_ir.parser import parse_string

    if kind == "concrete":
        kernel = lower_entry("stencils", "laplacian", size)
    elif kind == "parametric":
        kernel = _specialized("stencils", "laplacian", size)
    else:
        kernel = _parametric("corpus", "east_only").kernel
    text = kernel.as_ir()
    assert parse_string(text).as_ir() == text


def _corner_streams(size):
    """
    :return: The x- and y-stream names used by the Laplacian's last interior PE.
    """
    summary = block_summary(_specialized("stencils", "laplacian", size))
    decls, sends, recvs, _ = _stream_use(summary)[(size[0], size[1])]
    used = sends | recvs
    return (
        {n for n in used if decls[n][0] != 0},
        {n for n in used if decls[n][1] != 0},
    )


def test_parity_vector_per_axis():
    """A1: at the last interior PE, x-streams follow I's parity and y-streams follow J's parity."""
    x0, y0 = _corner_streams((16, 16, 4))
    x1, y1 = _corner_streams((16, 17, 4))
    x2, y2 = _corner_streams((17, 16, 4))
    x3, y3 = _corner_streams((17, 17, 4))
    assert x0 == x1 and x2 == x3 and x0 != x2
    assert y0 == y2 and y1 == y3 and y0 != y1


def test_sptlc_rejects_sizes_below_bound(tmp_path):
    """B1/B17: sptlc rejects each axis just below the precondition, before parsing."""
    from spada.cli.compiler import generate_program

    result = _parametric("stencils", "laplacian")
    path = tmp_path / "laplacian_param.sptl"
    path.write_text(result.as_ir())
    b = result.bounds
    for axis in ("I", "J", "K"):
        if b[axis] <= 1:
            continue
        values = dict(b, **{axis: b[axis] - 1})
        with pytest.raises(ValueError, match=f"{axis}={b[axis] - 1}"):
            generate_program(
                str(path),
                str(tmp_path / "out"),
                [f"{p}={v}" for p, v in values.items()],
            )


def test_equality_logs_constraint():
    """A5: an equality test alone records the bound that makes it false."""
    with ConstraintLog() as log:
        assert not (3 == Coord("I", 1))
    assert log.bounds == {"I": 3}
    with ConstraintLog() as log:
        assert 3 != Coord("I", 1)
    assert log.bounds == {"I": 3}


def test_low_anchored_values_are_ints():
    """A4/A6: low-anchored values hash like ints; high-minus-low stays anchored and has no truth value."""
    assert len({3, 3 + 0}) == 1
    k = Coord("K", -1) - 0
    assert isinstance(k, Coord) and k.offset == -1
    with ConstraintLog() as log:
        assert k > 0
        with pytest.raises(SymbolicCoordError):
            bool(k)
    assert log.bounds == {"K": 2}


def _check_partition(rects):
    groups = partition(rects)
    for x in range(12):
        for y in range(12):
            covering = sorted(
                r.metadata
                for r in rects
                if r.x_range[0] <= x < r.x_range[1] and r.y_range[0] <= y < r.y_range[1]
            )
            hits = [
                g
                for g in groups
                if g[0].x_range[0] <= x < g[0].x_range[1]
                and g[0].y_range[0] <= y < g[0].y_range[1]
            ]
            assert len(hits) == (1 if covering else 0)
            if covering:
                assert sorted(r.metadata for r in hits[0]) == covering
    return {
        (g[0].x_range, g[0].y_range, frozenset(r.metadata for r in g)) for g in groups
    }


def _r(x0, x1, y0, y1, k):
    return Rectangle((x0, x1, 1), (y0, y1, 1), k)


SHAPES = {
    "l_shape": [_r(0, 2, 0, 6, 0), _r(0, 6, 0, 2, 1)],
    "ring": [
        _r(0, 6, 0, 1, 0),
        _r(0, 6, 5, 6, 1),
        _r(0, 1, 0, 6, 2),
        _r(5, 6, 0, 6, 3),
    ],
    "staircase": [_r(i, i + 2, i, i + 2, i) for i in range(5)],
    "nested": [
        _r(0, 8, 0, 8, 0),
        _r(2, 6, 2, 6, 1),
        _r(3, 4, 3, 4, 2),
        _r(2, 6, 2, 6, 3),
    ],
    "touching": [
        _r(0, 3, 0, 3, 0),
        _r(3, 6, 0, 3, 1),
        _r(3, 6, 3, 6, 2),
        _r(4, 4, 0, 5, 3),
    ],
}


@pytest.mark.parametrize("shape", SHAPES)
def test_partition_shapes_and_order_independence(shape):
    """B10: merged regions cover each PE once, and do not depend on the input order."""
    rects = SHAPES[shape]
    expected = _check_partition(rects)
    rng = random.Random(0)
    for _ in range(5):
        shuffled = rects[:]
        rng.shuffle(shuffled)
        assert _check_partition(shuffled) == expected
