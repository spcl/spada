import pytest
from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, parser, passes, copy_elimination


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


def test_inline_phases_freshens_replicated_place_fields():
    kernel = spir.Kernel(
        name='test',
        parameters=[],
        arguments=[],
        body=[
            spir.Phase(
                place=[
                    spir.PlaceBlock(_make_block_vars(), _make_subgrid(), [
                        spir.FieldDeclaration(spir.ScalarType.i16, spir.Identifier('tmp', 0))
                    ])
                ],
                dataflow=[],
                compute=[
                    spir.ComputeBlock(_make_block_vars(), _make_subgrid(), [
                        spir.AssignmentStatement(
                            spir.Identifier('tmp', 0),
                            spir.Expression(spir.ConstantLiteral(1, spir.ScalarType.i16)))
                    ])
                ],
            ),
            spir.Phase(
                place=[
                    spir.PlaceBlock(_make_block_vars(), _make_subgrid(), [
                        spir.FieldDeclaration(spir.ScalarType.i16, spir.Identifier('tmp', 0))
                    ])
                ],
                dataflow=[],
                compute=[
                    spir.ComputeBlock(_make_block_vars(), _make_subgrid(), [
                        spir.AssignmentStatement(
                            spir.Identifier('tmp', 0),
                            spir.Expression(spir.ConstantLiteral(2, spir.ScalarType.i16)))
                    ])
                ],
            ),
        ],
    )

    inlined = canonicalization.inline_phases(kernel)
    place = next(block for block in inlined.body if isinstance(block, spir.PlaceBlock))
    compute = next(block for block in inlined.body if isinstance(block, spir.ComputeBlock))

    assert [statement.field_name.as_ir() for statement in place.statements] == ['tmp', 'tmp#1']
    assert compute.statements[0].destination.as_ir() == 'tmp'
    assert isinstance(compute.statements[1], spir.AwaitAllStatement)
    assert compute.statements[2].destination.as_ir() == 'tmp#1'


def test_inline_phases_shares_global_place_fields():
    global_place = spir.PlaceBlock(_make_block_vars(), _make_subgrid(), [
        spir.FieldDeclaration(spir.ScalarType.i16, spir.Identifier('tmp', 0))
    ])
    kernel = spir.Kernel(
        name='test',
        parameters=[],
        arguments=[],
        body=[
            global_place,
            spir.Phase(
                place=[
                    spir.PlaceBlock(_make_block_vars(), _make_subgrid(), [
                        spir.FieldDeclaration(spir.ScalarType.i16, spir.Identifier('tmp', 0))
                    ])
                ],
                dataflow=[],
                compute=[
                    spir.ComputeBlock(_make_block_vars(), _make_subgrid(), [
                        spir.AssignmentStatement(
                            spir.Identifier('tmp', 0),
                            spir.Expression(spir.ConstantLiteral(1, spir.ScalarType.i16)))
                    ])
                ],
            ),
            spir.Phase(
                place=[
                    spir.PlaceBlock(_make_block_vars(), _make_subgrid(), [
                        spir.FieldDeclaration(spir.ScalarType.i16, spir.Identifier('tmp', 0))
                    ])
                ],
                dataflow=[],
                compute=[
                    spir.ComputeBlock(_make_block_vars(), _make_subgrid(), [
                        spir.AssignmentStatement(
                            spir.Identifier('tmp', 0),
                            spir.Expression(spir.ConstantLiteral(2, spir.ScalarType.i16)))
                    ])
                ],
            ),
        ],
    )

    inlined = canonicalization.inline_phases(kernel)
    place = next(block for block in inlined.body if isinstance(block, spir.PlaceBlock))
    compute = next(block for block in inlined.body if isinstance(block, spir.ComputeBlock))

    assert [statement.field_name.as_ir() for statement in place.statements] == ['tmp']
    assert compute.statements[0].destination.as_ir() == 'tmp'
    assert isinstance(compute.statements[1], spir.AwaitAllStatement)
    assert compute.statements[2].destination.as_ir() == 'tmp'


def test_inline_phases_freshens_replicated_streams():
    global_place = spir.PlaceBlock(_make_block_vars(), _make_subgrid(), [
        spir.FieldDeclaration(spir.ScalarType.i16, spir.Identifier('tmp', 0))
    ])
    kernel = spir.Kernel(
        name='test',
        parameters=[],
        arguments=[],
        body=[
            global_place,
            spir.Phase(
                place=[],
                dataflow=[
                    spir.DataflowBlock(_make_block_vars(), _make_subgrid(), [
                        spir.StreamDeclaration(
                            spir.StreamType(spir.ScalarType.i16),
                            spir.Identifier('s', 0),
                            spir.RelativeStreamDeclaration(_make_number(1), _make_number(0)),
                        )
                    ])
                ],
                compute=[
                    spir.ComputeBlock(_make_block_vars(), _make_subgrid(), [
                        spir.SendStatement(spir.Identifier('tmp', 0), spir.Identifier('s', 0))
                    ])
                ],
            ),
            spir.Phase(
                place=[],
                dataflow=[
                    spir.DataflowBlock(_make_block_vars(), _make_subgrid(), [
                        spir.StreamDeclaration(
                            spir.StreamType(spir.ScalarType.i16),
                            spir.Identifier('s', 0),
                            spir.RelativeStreamDeclaration(_make_number(1), _make_number(0)),
                        )
                    ])
                ],
                compute=[
                    spir.ComputeBlock(_make_block_vars(), _make_subgrid(), [
                        spir.SendStatement(spir.Identifier('tmp', 0), spir.Identifier('s', 0))
                    ])
                ],
            ),
        ],
    )

    inlined = canonicalization.inline_phases(kernel)
    dataflow = next(block for block in inlined.body if isinstance(block, spir.DataflowBlock))
    compute = next(block for block in inlined.body if isinstance(block, spir.ComputeBlock))

    assert [statement.stream_name.as_ir() for statement in dataflow.statements] == ['s', 's#1']
    assert compute.statements[0].stream_name.as_ir() == 's'
    assert isinstance(compute.statements[1], spir.AwaitAllStatement)
    assert compute.statements[2].stream_name.as_ir() == 's#1'


def _make_number(num: int):
    return spir.Expression(spir.ConstantLiteral(num, spir.ScalarType.i16))


def _make_range(start: int, end: int):
    return spir.RangeExpression(_make_number(start), _make_number(end))


def _make_subgrid():
    return spir.SubgridExpression(_make_range(0, 4), _make_range(0, 4))


def _make_block_vars():
    return [
        spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('i', 0)),
        spir.TypedIdentifier(spir.ScalarType.i16, spir.Identifier('j', 0)),
    ]


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
    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    assert len(rects) == 1
    kernel = copy_elimination.prune_unused_fields(rects)
    assert len(rects[0].metadata.place.statements) == 5
    assert all(decl.field_name.name.startswith('do_not_erase') for decl in rects[0].metadata.place.statements)


if __name__ == '__main__':
    test_canonicalize_nochange()
    test_canonicalize_singlephase()
    test_canonicalize_multiphase()
    test_lower_to_for(False)
    test_lower_to_for(True)
    test_prune_unused_fields()
