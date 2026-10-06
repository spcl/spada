import pytest

from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.spatial_ir import parser, passes


def _canonicalize(code: str):
    kernel = parser.parse_string(code, 'test.sptl')
    kernel = passes.constexpr_propagation(kernel)
    return s2c.canonicalize_kernel(kernel)


def test_overlapping_compute_subgrids_in_one_phase_are_rejected():
    code = """
kernel @overlapping_compute<>() {
    phase {
        compute i16 i, i16 j in [0:4, 0] {}
        compute i16 i, i16 j in [2:6, 0] {}
    }
}
"""
    with pytest.raises(
        SyntaxError,
        match=r'Overlapping compute subgrids in phase 1.*x=\(2, 4, 1\)',
    ):
        _canonicalize(code)


def test_overlapping_dataflow_subgrids_in_one_phase_are_rejected():
    code = """
kernel @overlapping_dataflow<>() {
    phase {
        dataflow i16 i, i16 j in [0:4, 0] {}
        dataflow i16 i, i16 j in [3:6, 0] {}
    }
}
"""
    with pytest.raises(
        SyntaxError,
        match=r'Overlapping dataflow subgrids in phase 1.*x=\(3, 4, 1\)',
    ):
        _canonicalize(code)


def test_disjoint_strided_compute_subgrids_are_accepted():
    code = """
kernel @disjoint_compute<>() {
    phase {
        compute i16 i, i16 j in [0:8:2, 0] {}
        compute i16 i, i16 j in [1:8:2, 0] {}
    }
}
"""
    _canonicalize(code)


def test_compute_subgrids_may_overlap_across_phases():
    code = """
kernel @compute_across_phases<>() {
    phase {
        compute i16 i, i16 j in [0:4, 0] {}
    }
    phase {
        compute i16 i, i16 j in [0:4, 0] {}
    }
}
"""
    _canonicalize(code)
