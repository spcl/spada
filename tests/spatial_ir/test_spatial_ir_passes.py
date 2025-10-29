from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, passes, parser


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
                spir.RelativeStreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(-1), _make_number(1))
            ]),
            spir.ComputeBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(0, 30), _make_range(0, 30)), []),
            spir.DataflowBlock([
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
                spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0))
            ], spir.SubgridExpression(_make_range(20, 30), _make_range(0, 30)), [
                spir.RelativeStreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(1), _make_number(-1))
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
                spir.RelativeStreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(-1), _make_number(1))
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
                spir.RelativeStreamDeclaration(
                    spir.StreamType(spir.ScalarType.f32), spir.Identifier('s', 0), _make_number(1), _make_number(-1))
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


def test_prune_unused_fields():
    code = '''kernel @test<N>() {
    place u16 i, u16 j in [0:N, 0:N] {
        f32 erase;
        f32 do_not_erase_1;
        f32[N] do_not_erase_2;
        f32 do_not_erase_3;
        f32[N] do_not_erase_4;
        f32[N] do_not_erase_5;
        f32[N] erase_arr;
    }
    compute u16 i, u16 j in [0:N, 0:N] {
        do_not_erase_1 = 5.0;
        await receive(do_not_erase_2, a[i, j]);
        for u16 k in [0:N] {
            do_not_erase_3 = do_not_erase_2[k] + 1.0;
        }
        completion c2 = foreach u16 k, f32 x in [0:N], receive(do_not_erase_4) {
            do_not_erase_5[k] = x + 2.0;
            await send(do_not_erase_5[k], b[i, j]);
        }
        await c2;
    }
}'''
    kernel = parser.parse_string(code)
    kernel = passes.prune_unused_fields(kernel)
    assert len(kernel.body[0].statements) == 5
    assert all(decl.field_name.name.startswith('do_not_erase') for decl in kernel.body[0].statements)


if __name__ == '__main__':
    test_canonicalize_nochange()
    test_canonicalize_singlephase()
    test_canonicalize_multiphase()
    test_prune_unused_fields()
