import json

import pytest

from spada.lowering.parametric import (
    check_precondition,
    lower_gt4py_parametric,
    parse_precondition,
)
from spada.syntax.gt4py import parser
from tests.stencil_ir.parametric.corpus import GT4PY_SOURCES
from tests.stencil_ir.parametric.differential import check_stencil

# A representative subset; the full corpus is run by ``python3 -m tests.stencil_ir.parametric.differential``.
STENCILS = [
    ("stencils", "laplacian"),
    ("stencils", "pure_vertical"),
    ("corpus", "west_south"),
    ("corpus", "horizontal_then_forward"),
    ("corpus", "backward_with_halo"),
    ("test_instances", "forward_lookahead_inout"),
]


@pytest.mark.parametrize("stencil", STENCILS, ids=lambda s: f"{s[0]}.{s[1]}")
def test_parametric_matches_concrete(stencil):
    """Theorem 1: the specialized parametric kernel prints exactly as the concrete lowering."""
    results = check_stencil(stencil)
    assert results
    for _, size, status in results:
        assert status == "identical", f"{size}: {status}"


def _parametric(key, name):
    return lower_gt4py_parametric(parser.parse_file(str(GT4PY_SOURCES[key]))[name])


def test_parity_guards_cover_corner():
    """The Laplacian corner region communicates along both high-anchored axes and gets four guarded copies."""
    text = _parametric("stencils", "laplacian").kernel.as_ir()
    corner = "compute u16 i#7, u16 j#7 in [I:(I + 1):2 , J:(J + 1):2]"
    assert text.count(corner) == 4
    assert "for u16 __parity_x in [0:(I % 2)]" in text
    assert "for u16 __parity_y in [0:(1 - (J % 2))]" in text


def test_precondition_header_and_sidecar_agree():
    result = _parametric("stencils", "laplacian")
    source = result.as_ir()
    bounds = parse_precondition(source)
    assert bounds == json.loads(result.sidecar())["min"] == result.bounds
    check_precondition(bounds, {"I": bounds["I"], "J": bounds["J"], "K": bounds["K"]})
    with pytest.raises(ValueError, match="I=3"):
        check_precondition(bounds, {"I": bounds["I"] - 1, "J": 100, "K": 100})
    with pytest.raises(ValueError):
        check_precondition(bounds, {"J": 100, "K": 100})


def test_parse_precondition():
    assert parse_precondition("kernel @k<>() {}") is None
    assert parse_precondition("// spada-precondition: I>=3 J>=1 K>=2\nkernel") == {
        "I": 3,
        "J": 1,
        "K": 2,
    }
    with pytest.raises(ValueError):
        parse_precondition("// spada-precondition: I>3")
