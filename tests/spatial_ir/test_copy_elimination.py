from spatialstencil.syntax.spatial_ir import irnodes as spir, canonicalization, copy_elimination, parser


def _optimize_kernel(kernel: spir.Kernel):
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)
    assert len(rects) == 1
    return rects[0]


def _place_field_names(rect) -> list[str]:
    return [decl.field_name.name for decl in rect.metadata.place.statements]


def test_remove_redundant_scalar_copy():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 a
                f32 tmp
                f32 out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                tmp = a
                out = tmp
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "out"]
    assert len(rect.metadata.compute.statements) == 1
    stmt = rect.metadata.compute.statements[0]
    assert isinstance(stmt, spir.AssignmentStatement)
    assert isinstance(stmt.destination, spir.Identifier)
    assert stmt.destination.name == "out"
    assert isinstance(stmt.source.value, spir.Identifier)
    assert stmt.source.value.name == "a"


def test_do_not_remove_copy_if_source_changes_before_use():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 a
                f32 tmp
                f32 out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                tmp = a
                a = out
                out = tmp
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out"]
    assert len(rect.metadata.compute.statements) == 3


def test_remove_redundant_map_copy_after_array_lowering():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[8] a
                f32[8] tmp
                f32[8] out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                tmp = a
                out = tmp
            }
        }
        """,
        "test.sptl",
    )

    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    canonicalization.lower_array_assignment(rects)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)
    rect = rects[0]

    assert _place_field_names(rect) == ["a", "out"]
    assert len(rect.metadata.compute.statements) == 1
    stmt = rect.metadata.compute.statements[0]
    assert isinstance(stmt, spir.MapStatement)
    body_stmt = stmt.body[0]
    assert isinstance(body_stmt, spir.AssignmentStatement)
    assert isinstance(body_stmt.source.value, spir.ArraySlice)
    assert body_stmt.source.value.array.name == "a"


def test_remove_redundant_copy_inside_foreach_body():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 tmp
            }
            dataflow u16 i, u16 j in [0:1, 0:1] {
                stream<f32> red = relative_stream(-1, 0) {
                    hops = [(-1, 0)],
                    channel = 0
                }
                stream<f32> blue = relative_stream(1, 0) {
                    hops = [(1, 0)],
                    channel = 1
                }
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                await foreach u16 k, f32 x in [0:8], receive(red) {
                    tmp = x
                    await send(tmp, blue)
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == []
    assert len(rect.metadata.compute.statements) == 1
    foreach_stmt = rect.metadata.compute.statements[0]
    assert isinstance(foreach_stmt, spir.ForeachStatement)
    assert len(foreach_stmt.body) == 1
    send_stmt = foreach_stmt.body[0]
    assert isinstance(send_stmt, spir.SendStatement)
    assert isinstance(send_stmt.local_array, spir.Identifier)
    assert send_stmt.local_array.name == "x"


def test_prune_unused_fields_keeps_extern_fields():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                extern f32[1] input_field
                f32 tmp
            }
            compute u16 i, u16 j in [0:1, 0:1] {
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert len(rect.metadata.place.statements) == 1
    assert rect.metadata.place.statements[0].is_extern
    assert rect.metadata.place.statements[0].field_name.name == "input_field"


def test_remove_bulk_foreach_receive_send_copy():
    kernel = parser.parse_string(
        """
        kernel @copy<>() {
            place u16 i, u16 j in [0:16, 0:16] {
                f32[2] local
                extern f32[2] a
                extern f32[2] out
            }
            compute u16 i, u16 j in [0:16, 0:16] {
                await foreach u16 __k0, f32 __x in [0:2], receive(a) {
                    local[__k0] = __x
                }
                await send(local, out)
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "out"]
    assert len(rect.metadata.compute.statements) == 1
    send_stmt = rect.metadata.compute.statements[0]
    assert isinstance(send_stmt, spir.SendStatement)
    assert isinstance(send_stmt.local_array, spir.Identifier)
    assert send_stmt.local_array.name == "a"
    assert isinstance(send_stmt.stream_name, spir.Identifier)
    assert send_stmt.stream_name.name == "out"


def test_do_not_remove_bulk_receive_relay_copy_to_dataflow_stream():
    kernel = parser.parse_string(
        """
        kernel @copy<>() {
            place u16 i, u16 j in [0:16, 0:16] {
                f32[2] local
                extern f32[2] a
            }
            dataflow u16 i, u16 j in [0:16, 0:16] {
                stream<f32> relay = relative_stream(1, 0) {
                    hops = [(1, 0)],
                    channel = 0
                }
            }
            compute u16 i, u16 j in [0:16, 0:16] {
                await foreach u16 __k0, f32 __x in [0:2], receive(a) {
                    local[__k0] = __x
                }
                completion done = send(local, relay)
                await done
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["local", "a"]
    assert len(rect.metadata.compute.statements) == 3

    foreach_stmt = rect.metadata.compute.statements[0]
    send_stmt = rect.metadata.compute.statements[1]

    assert isinstance(foreach_stmt, spir.ForeachStatement)
    assert isinstance(send_stmt, spir.SendStatement)
    assert isinstance(send_stmt.local_array, spir.Identifier)
    assert send_stmt.local_array.name == "local"
    assert isinstance(send_stmt.stream_name, spir.Identifier)
    assert send_stmt.stream_name.name == "relay"


def test_remove_bulk_receive_copy_to_extern_field_with_non_stream_name():
    kernel = parser.parse_string(
        """
        kernel @copy<>() {
            place u16 i, u16 j in [0:16, 0:16] {
                f32[2] local
                extern f32[2] source
                extern f32[2] downstream
            }
            compute u16 i, u16 j in [0:16, 0:16] {
                await foreach u16 __k0, f32 __x in [0:2], receive(source) {
                    local[__k0] = __x
                }
                await send(local, downstream)
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["source", "downstream"]
    assert len(rect.metadata.compute.statements) == 1
    send_stmt = rect.metadata.compute.statements[0]
    assert isinstance(send_stmt, spir.SendStatement)
    assert isinstance(send_stmt.local_array, spir.Identifier)
    assert send_stmt.local_array.name == "source"
    assert isinstance(send_stmt.stream_name, spir.Identifier)
    assert send_stmt.stream_name.name == "downstream"


def test_do_not_elide_extern_field_in_copy_chain():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 a
                extern f32 tmp
                f32 out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                tmp = a
                out = tmp
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out"]
    assert rect.metadata.place.statements[1].is_extern
    assert len(rect.metadata.compute.statements) == 2

    first_stmt = rect.metadata.compute.statements[0]
    second_stmt = rect.metadata.compute.statements[1]
    assert isinstance(first_stmt, spir.AssignmentStatement)
    assert isinstance(second_stmt, spir.AssignmentStatement)
    assert isinstance(first_stmt.destination, spir.Identifier)
    assert isinstance(second_stmt.source.value, spir.Identifier)
    assert first_stmt.destination.name == "tmp"
    assert second_stmt.source.value.name == "tmp"


def test_do_not_remove_copy_across_for_boundary():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 a
                f32 tmp
                f32 out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                tmp = a
                for u16 k in [0:8] {
                    out = tmp
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out"]
    assert len(rect.metadata.compute.statements) == 2


def test_remove_single_element_forwarding_inside_for_body():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] tmp
                f32[4] out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp[k] = a[k]
                    out[k] = tmp[k]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "out"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 1
    assignment = loop_stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assert isinstance(assignment.source.value, spir.ArraySlice)
    assert assignment.source.value.array.name == "a"


def test_remove_single_element_forwarding_inside_expression():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] b
                f32[4] tmp
                f32[4] out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp[k] = (a[k] + b[k])
                    out[k] = (0.25 * tmp[k])
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "b", "out"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 1
    assignment = loop_stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assert isinstance(assignment.source.value, spir.BinaryOperator)
    rhs = assignment.source.value
    assert rhs.op == "*"
    assert isinstance(rhs.right.value, spir.BinaryOperator)
    nested = rhs.right.value
    assert nested.op == "+"
    assert isinstance(nested.left.value, spir.ArraySlice)
    assert nested.left.value.array.name == "a"
    assert isinstance(nested.right.value, spir.ArraySlice)
    assert nested.right.value.array.name == "b"


def test_remove_single_element_forwarding_with_multiple_consumers():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] b
                f32[4] tmp
                f32[4] out0
                f32[4] out1
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp[k] = (a[k] + b[k])
                    out0[k] = (0.25 * tmp[k])
                    out1[k] = (tmp[k] - 1.0)
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "b", "out0", "out1"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 2

    first_assignment = loop_stmt.body[0]
    second_assignment = loop_stmt.body[1]
    assert isinstance(first_assignment, spir.AssignmentStatement)
    assert isinstance(second_assignment, spir.AssignmentStatement)

    first_rhs = first_assignment.source.value
    assert isinstance(first_rhs, spir.BinaryOperator)
    assert first_rhs.op == "*"
    assert isinstance(first_rhs.right.value, spir.BinaryOperator)
    assert first_rhs.right.value.op == "+"

    second_rhs = second_assignment.source.value
    assert isinstance(second_rhs, spir.BinaryOperator)
    assert second_rhs.op == "-"
    assert isinstance(second_rhs.left.value, spir.BinaryOperator)
    assert second_rhs.left.value.op == "+"


def test_remove_single_element_forwarding_with_multiple_dependent_consumers():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 coeff
                f32[4] ref
                f32[4] wcon
                f32[4] u_stage
                f32[4] gcv
                f32[4] ccol_tmp
                f32[4] divided
                f32[4] dcol
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    gcv[k] = (0.25 * (ref[(k + 1)] + wcon[(k + 1)]))
                    ccol_tmp[k] = (gcv[k] * 0.5)
                    divided[k] = (1.0 / (coeff - ccol_tmp[k]))
                    dcol[k] = ((gcv[k] * 0.5) * (u_stage[(k + 1)] - u_stage[k]))
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["coeff", "ref", "wcon", "u_stage", "divided", "dcol"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 2

    divided_assignment = loop_stmt.body[0]
    dcol_assignment = loop_stmt.body[1]
    assert isinstance(divided_assignment, spir.AssignmentStatement)
    assert isinstance(dcol_assignment, spir.AssignmentStatement)

    divided_rhs = divided_assignment.source.value
    assert isinstance(divided_rhs, spir.BinaryOperator)
    assert divided_rhs.op == "/"
    assert isinstance(divided_rhs.right.value, spir.BinaryOperator)
    assert divided_rhs.right.value.op == "-"
    assert isinstance(divided_rhs.right.value.right.value, spir.BinaryOperator)
    assert divided_rhs.right.value.right.value.op == "*"

    dcol_rhs = dcol_assignment.source.value
    assert isinstance(dcol_rhs, spir.BinaryOperator)
    assert dcol_rhs.op == "*"
    assert isinstance(dcol_rhs.left.value, spir.BinaryOperator)
    assert dcol_rhs.left.value.op == "*"


def test_remove_single_element_forwarding_chain_inside_for_body():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] dcol
                f32[4] datacol
                f32[4] data_col_stage
                f32[4] data_col
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    datacol[k] = dcol[k]
                    data_col_stage[k] = datacol[k]
                    data_col[k] = data_col_stage[k]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["dcol", "data_col"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 1
    assignment = loop_stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assert isinstance(assignment.destination, spir.ArraySlice)
    assert assignment.destination.array.name == "data_col"
    assert isinstance(assignment.source.value, spir.ArraySlice)
    assert assignment.source.value.array.name == "dcol"


def test_remove_single_element_forwarding_arithmetic_chain_inside_for_body():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 coeff
                f32[4] u
                f32[4] v
                f32[4] w
                f32[4] corr
                f32[4] tmp0
                f32[4] tmp1
                f32[4] tmp2
                f32[4] out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp0[k] = (coeff * u[k])
                    tmp1[k] = (tmp0[k] + v[k])
                    tmp2[k] = (tmp1[k] + w[k])
                    out[k] = (tmp2[k] + corr[k])
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["coeff", "u", "v", "w", "corr", "out"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 1
    assignment = loop_stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assert isinstance(assignment.source.value, spir.BinaryOperator)
    outer = assignment.source.value
    assert outer.op == "+"
    assert isinstance(outer.right.value, spir.ArraySlice)
    assert outer.right.value.array.name == "corr"

    level2 = outer.left.value
    assert isinstance(level2, spir.BinaryOperator)
    assert level2.op == "+"
    assert isinstance(level2.right.value, spir.ArraySlice)
    assert level2.right.value.array.name == "w"

    level1 = level2.left.value
    assert isinstance(level1, spir.BinaryOperator)
    assert level1.op == "+"
    assert isinstance(level1.right.value, spir.ArraySlice)
    assert level1.right.value.array.name == "v"

    base = level1.left.value
    assert isinstance(base, spir.BinaryOperator)
    assert base.op == "*"
    assert isinstance(base.left.value, spir.Identifier)
    assert base.left.value.name == "coeff"
    assert isinstance(base.right.value, spir.ArraySlice)
    assert base.right.value.array.name == "u"


def test_remove_single_element_forwarding_inside_map_body():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] tmp
                f32[4] out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                await map i32 k in [0:4:1] {
                    tmp[k] = a[k]
                    out[k] = tmp[k]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "out"]
    map_stmt = rect.metadata.compute.statements[0]
    assert isinstance(map_stmt, spir.MapStatement)
    assert len(map_stmt.body) == 1
    assignment = map_stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assert isinstance(assignment.source.value, spir.ArraySlice)
    assert assignment.source.value.array.name == "a"


def test_remove_single_element_forwarding_inside_foreach_body():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] tmp
                f32[4] out
            }
            dataflow u16 i, u16 j in [0:1, 0:1] {
                stream<f32> red = relative_stream(-1, 0) {
                    hops = [(-1, 0)],
                    channel = 0
                }
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                await foreach i32 k, f32 x in [0:4:1], receive(red) {
                    tmp[k] = x
                    out[k] = tmp[k]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["out"]
    foreach_stmt = rect.metadata.compute.statements[0]
    assert isinstance(foreach_stmt, spir.ForeachStatement)
    assert len(foreach_stmt.body) == 1
    assignment = foreach_stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assert isinstance(assignment.source.value, spir.Identifier)
    assert assignment.source.value.name == "x"


def test_do_not_elide_extern_indexed_forwarding_temp():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                extern f32[4] tmp
                f32[4] out
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp[k] = a[k]
                    out[k] = tmp[k]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out"]
    assert rect.metadata.place.statements[1].is_extern
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 2


def test_do_not_remove_nested_scalar_copy_if_temp_used_after_loop():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 a
                f32 tmp
                f32 out
                extern f32 sink
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp = a
                    out = tmp
                }
                await send(tmp, sink)
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out", "sink"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 2
    first_stmt = loop_stmt.body[0]
    second_stmt = loop_stmt.body[1]
    assert isinstance(first_stmt, spir.AssignmentStatement)
    assert isinstance(second_stmt, spir.AssignmentStatement)
    assert isinstance(first_stmt.destination, spir.Identifier)
    assert first_stmt.destination.name == "tmp"
    assert isinstance(second_stmt.source.value, spir.Identifier)
    assert second_stmt.source.value.name == "tmp"


def test_do_not_remove_single_element_forwarding_if_temp_used_after_loop():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] tmp
                f32[4] out
                extern f32[4] sink
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp[k] = a[k]
                    out[k] = tmp[k]
                }
                await send(tmp, sink)
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out", "sink"]
    loop_stmt = rect.metadata.compute.statements[0]
    assert isinstance(loop_stmt, spir.ForStatement)
    assert len(loop_stmt.body) == 2
    first_stmt = loop_stmt.body[0]
    second_stmt = loop_stmt.body[1]
    assert isinstance(first_stmt, spir.AssignmentStatement)
    assert isinstance(second_stmt, spir.AssignmentStatement)
    assert isinstance(first_stmt.destination, spir.ArraySlice)
    assert first_stmt.destination.array.name == "tmp"
    assert isinstance(second_stmt.source.value, spir.ArraySlice)
    assert second_stmt.source.value.array.name == "tmp"


def test_do_not_remove_single_element_forwarding_if_temp_used_in_second_loop():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] tmp
                f32[4] out0
                f32[4] out1
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 k in [0:4:1] {
                    tmp[k] = a[k]
                    out0[k] = tmp[k]
                }
                for i32 k in [0:4:1] {
                    out1[k] = tmp[k]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out0", "out1"]
    first_loop = rect.metadata.compute.statements[0]
    second_loop = rect.metadata.compute.statements[1]
    assert isinstance(first_loop, spir.ForStatement)
    assert isinstance(second_loop, spir.ForStatement)
    assert len(first_loop.body) == 2
    assert len(second_loop.body) == 1
    first_loop_consumer = first_loop.body[1]
    second_loop_consumer = second_loop.body[0]
    assert isinstance(first_loop_consumer, spir.AssignmentStatement)
    assert isinstance(second_loop_consumer, spir.AssignmentStatement)
    assert isinstance(first_loop_consumer.source.value, spir.ArraySlice)
    assert first_loop_consumer.source.value.array.name == "tmp"
    assert isinstance(second_loop_consumer.source.value, spir.ArraySlice)
    assert second_loop_consumer.source.value.array.name == "tmp"


def test_protected_fields_propagate_into_nested_scalar_loop_region():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32 a
                f32 tmp
                f32 out
                f32 later
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 ii in [0:2:1] {
                    for i32 jj in [0:2:1] {
                        tmp = a
                        out = tmp
                    }
                }
                for i32 kk in [0:2:1] {
                    later = tmp
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out", "later"]
    outer_loop = rect.metadata.compute.statements[0]
    later_loop = rect.metadata.compute.statements[1]
    assert isinstance(outer_loop, spir.ForStatement)
    assert isinstance(later_loop, spir.ForStatement)
    assert len(outer_loop.body) == 1

    inner_loop = outer_loop.body[0]
    assert isinstance(inner_loop, spir.ForStatement)
    assert len(inner_loop.body) == 2
    first_stmt = inner_loop.body[0]
    second_stmt = inner_loop.body[1]
    assert isinstance(first_stmt, spir.AssignmentStatement)
    assert isinstance(second_stmt, spir.AssignmentStatement)
    assert isinstance(first_stmt.destination, spir.Identifier)
    assert first_stmt.destination.name == "tmp"
    assert isinstance(second_stmt.source.value, spir.Identifier)
    assert second_stmt.source.value.name == "tmp"


def test_protected_fields_propagate_into_nested_indexed_loop_region():
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                f32[4] a
                f32[4] tmp
                f32[4] out
                f32[4] later
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                for i32 ii in [0:2:1] {
                    for i32 jj in [0:4:1] {
                        tmp[jj] = a[jj]
                        out[jj] = tmp[jj]
                    }
                }
                for i32 kk in [0:4:1] {
                    later[kk] = tmp[kk]
                }
            }
        }
        """,
        "test.sptl",
    )

    rect = _optimize_kernel(kernel)

    assert _place_field_names(rect) == ["a", "tmp", "out", "later"]
    outer_loop = rect.metadata.compute.statements[0]
    later_loop = rect.metadata.compute.statements[1]
    assert isinstance(outer_loop, spir.ForStatement)
    assert isinstance(later_loop, spir.ForStatement)
    assert len(outer_loop.body) == 1

    inner_loop = outer_loop.body[0]
    assert isinstance(inner_loop, spir.ForStatement)
    assert len(inner_loop.body) == 2
    first_stmt = inner_loop.body[0]
    second_stmt = inner_loop.body[1]
    assert isinstance(first_stmt, spir.AssignmentStatement)
    assert isinstance(second_stmt, spir.AssignmentStatement)
    assert isinstance(first_stmt.destination, spir.ArraySlice)
    assert first_stmt.destination.array.name == "tmp"
    assert isinstance(second_stmt.source.value, spir.ArraySlice)
    assert second_stmt.source.value.array.name == "tmp"
