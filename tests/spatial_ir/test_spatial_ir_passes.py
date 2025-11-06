import pytest
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, parser


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
            spir.DataflowBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(0, 20), _make_range(0, 30)), [
                spir.StreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0),
                    spir.RelativeStreamDeclaration(_make_number(-1), _make_number(1)))
            ]),
            spir.ComputeBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(0, 30), _make_range(0, 30)), []),
            spir.DataflowBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(20, 30), _make_range(0, 30)), [
                spir.StreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0),
                    spir.RelativeStreamDeclaration(_make_number(1), _make_number(-1)))
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
            spir.DataflowBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(0, 20), _make_range(0, 30)), [
                spir.StreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0),
                    spir.RelativeStreamDeclaration(_make_number(-1), _make_number(1)))
            ]),
            spir.ComputeBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(0, 30), _make_range(0, 30)), []),
            spir.Phase([], [], []),
            spir.DataflowBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(20, 30), _make_range(0, 30)), [
                spir.StreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0),
                    spir.RelativeStreamDeclaration(_make_number(1), _make_number(-1)))
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


@pytest.mark.parametrize("should_lower", [True, False])
def test_lower_to_for(should_lower: bool):
    stream = 'b' if should_lower else 'red'
    kernel_str = f"""
      kernel @test<>() {{
          place i16 i, i16 j in [0:4, 0:4] {{
              f32[8] a
              f32[8] b
          }}
          dataflow i16 i, i16 j in [0:4 , 0:4] {{
              stream<f32> red = relative_stream(-1, 0) {{
                  hops = [(-1, 0)],
                  channel = 0
              }}
              stream<f32> blue = relative_stream(-1, 0) {{
                  hops = [(-1, 0)],
                  channel = 1
              }}
          }}
          compute i16 i, i16 j in [0:4, 0:4] {{
              await foreach i16 k, f32 x in [0:8], receive({stream}) {{
                  a[k] = a[k] + x
                  await send(a[k], blue)
              }}
          }}
      }}"""
    kernel = parser.parse_string(kernel_str, "test.sptl")
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    assert len(rects) == 1
    rect = rects[0]
    dtypes = {decl.field_name: decl.dtype for decl in rect.metadata.place.statements}
    dtypes.update({decl.stream_name: decl.dtype for decl in rect.metadata.dataflow.statements})
    canonicalization.convert_foreach_data_tasks_to_loops(rect, dtypes)
    if should_lower:
        assert not any(isinstance(stmt, spir.ForeachStatement) for stmt in rect.metadata.compute.statements)
    else:
        # There should still be a data task here
        assert any(isinstance(stmt, spir.ForeachStatement) for stmt in rect.metadata.compute.statements)


if __name__ == '__main__':
    test_canonicalize_nochange()
    test_canonicalize_singlephase()
    test_canonicalize_multiphase()
    test_lower_to_for(False)
    test_lower_to_for(True)
