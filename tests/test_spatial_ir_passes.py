import unittest

from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization


def test_canonicalize_nochange():
    kernel = spir.Kernel(
        name='test', parameters=[], arguments=[], body=[
            spir.Phase([], [], []),
            spir.Phase([], [], []),
        ])
    ir_before = kernel.as_ir()
    ckernel = canonicalization.canonicalize_phases(kernel)
    assert ckernel.as_ir() == ir_before


def test_canonicalize_singlephase():
    kernel = spir.Kernel(
        name='test',
        parameters=[],
        arguments=[],
        body=[
            spir.DataflowBlock([spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)), 
                                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))],
                               spir.SubgridExpression(_make_range(0, 20), _make_range(0, 30)), [
                                   spir.RelativeStreamDeclaration(
                                       spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(-1),
                                       _make_number(1))
                               ]),
            spir.ComputeBlock([spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)), 
                               spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))],
                              spir.SubgridExpression(_make_range(0, 30), _make_range(0, 30)), []),
            spir.DataflowBlock([spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)), 
                                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))],
                               spir.SubgridExpression(_make_range(20, 30), _make_range(0, 30)), [
                                   spir.RelativeStreamDeclaration(
                                       spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(1),
                                       _make_number(-1))
                               ])
        ])
    ckernel = canonicalization.canonicalize_phases(kernel)
    assert ckernel.as_ir() == '''kernel @test<>() {
  phase {
    dataflow i16 i, i16 j in [0:20 , 0:30] {
      stream<f32> s = relative_stream(-1, 1)
    }
    dataflow i16 i, i16 j in [20:30 , 0:30] {
      stream<f32> s = relative_stream(1, -1)
    }
    compute i16 i, i16 j in [0:30 , 0:30] {

    }
  }
}'''


def test_canonicalize_multiphase():
    kernel = spir.Kernel(
        name='test',
        parameters=[],
        arguments=[],
        body=[
            spir.DataflowBlock([spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)), 
                                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))],
                               spir.SubgridExpression(_make_range(0, 20), _make_range(0, 30)), [
                                   spir.RelativeStreamDeclaration(
                                       spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(-1),
                                       _make_number(1))
                               ]),
            spir.ComputeBlock([spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)), 
                               spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))],
                              spir.SubgridExpression(_make_range(0, 30), _make_range(0, 30)), []),
            spir.Phase([], [], []),
            spir.DataflowBlock([spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)), 
                                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))],
                               spir.SubgridExpression(_make_range(20, 30), _make_range(0, 30)), [
                                   spir.RelativeStreamDeclaration(
                                       spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(1),
                                       _make_number(-1))
                               ])
        ])
    ckernel = canonicalization.canonicalize_phases(kernel)
    assert ckernel.as_ir() == '''kernel @test<>() {
  phase {
    dataflow i16 i, i16 j in [0:20 , 0:30] {
      stream<f32> s = relative_stream(-1, 1)
    }
    compute i16 i, i16 j in [0:30 , 0:30] {

    }
  }
  phase {
  }
  phase {
    dataflow i16 i, i16 j in [20:30 , 0:30] {
      stream<f32> s = relative_stream(1, -1)
    }
  }
}'''


def _make_number(num: int):
    return spir.Expression(spir.ConstantLiteral(num, spir.ScalarType.i16))


def _make_range(start: int, end: int):
    return spir.RangeExpression(_make_number(start), _make_number(end))


if __name__ == '__main__':
    test_canonicalize_nochange()
    test_canonicalize_singlephase()
    test_canonicalize_multiphase()
