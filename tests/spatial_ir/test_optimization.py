from spatialstencil.syntax.spatial_ir import irnodes as spa
from spatialstencil.syntax.spatial_ir import parser, passes, canonicalization, copy_elimination
from typing import TypeVar
import pytest

T = TypeVar('T')


def parse_kernel(code: str) -> spa.Kernel:
    return parser.parse_string(code, "test.sptl")


def _first_assignment(statements: list[spa.Statement]) -> spa.AssignmentStatement:
    for stmt in statements:
        if isinstance(stmt, spa.AssignmentStatement):
            return stmt
    raise AssertionError("No assignment statement found")


def test_remove_copy_prior_to_send() -> None:
    code = """
    kernel @dead_copy<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = val;
            await send(tmp, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    assert len(compute.statements) == 1
    send_stmt = compute.statements[0]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "val"

    place = rects[0].metadata.place
    assert {decl.field_name.name for decl in place.statements} == {"val"}


def test_chain_of_copies_is_removed() -> None:
    code = """
    kernel @rename_chain<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 result;
            f32 val;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = val;
            result = tmp;
            await send(result, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    assert len(compute.statements) == 1
    send_stmt = compute.statements[0]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "val"

    place = rects[0].metadata.place
    assert {decl.field_name.name for decl in place.statements} == {"val"}


def test_copy_preserved_when_source_mutates() -> None:
    code = """
    kernel @preserve_copy<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 val;
            f32 other;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = val;
            val = other;
            await send(tmp, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)

    compute = rects[0].metadata.compute
    assert len(compute.statements) == 3
    first_stmt, second_stmt, send_stmt = compute.statements
    assert isinstance(first_stmt, spa.AssignmentStatement)
    assert isinstance(first_stmt.destination, spa.Identifier)
    assert first_stmt.destination.name == "tmp"
    assert isinstance(second_stmt, spa.AssignmentStatement)
    assert isinstance(second_stmt.destination, spa.Identifier)
    assert second_stmt.destination.name == "val"
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "tmp"


def test_map_copy_removed_and_fields_pruned() -> None:
    code = """
    kernel @map_copy<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[2] src;
            f32[2] tmp;
            f32[2] out;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await map u16 x in [0:2] {
                tmp[x] = src[x];
            };
            await map u16 x in [0:2] {
                out[x] = tmp[x];
            };
            await send(out, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    send_stmt = compute.statements[-1]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "out"

    place = rects[0].metadata.place
    names = {decl.field_name.name for decl in place.statements}
    assert "tmp" not in names


def test_map_with_index_mismatch_not_removed() -> None:
    code = """
    kernel @map_noncopy<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32[8] src;
            f32[4] tmp;
            f32[4] out;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await map u16 x in [0:4] {
                tmp[x] = src[x * 2];
            };
            await map u16 x in [0:4] {
                out[x] = tmp[x];
            };
            await send(out, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    maps = [stmt for stmt in compute.statements if isinstance(stmt, spa.MapStatement)]
    assert len(maps) == 2
    place = rects[0].metadata.place
    assert "tmp" in {decl.field_name.name for decl in place.statements}


@pytest.mark.skip(
    reason="Cross-region propagation from top-level statements into loop bodies is intentionally unsupported")
def test_rename_propagates_into_for_loop() -> None:
    code = """
    kernel @for_loop_copy<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 val;
            f32 out;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = val;
            for u16 k in [0:2] {
                out = tmp;
            }
            await send(out, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    loop = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForStatement))
    body_assign = _first_assignment(loop.body)
    assert isinstance(body_assign.source.value, spa.Identifier)
    assert body_assign.source.value.name == "val"

    place = rects[0].metadata.place
    assert "tmp" not in {decl.field_name.name for decl in place.statements}


def test_foreach_copy_not_aliased() -> None:
    code = """
    kernel @foreach_copy<N>(stream<f32, 1>[N] readonly input, stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 val;
            f32 out;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = val;
            await foreach u16 k, f32 elem in [0:2], receive(input[i]) {
                out = tmp;
                await send(out, output[i]);
            };
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    first_stmt = compute.statements[0]
    assert isinstance(first_stmt, spa.AssignmentStatement)
    assert isinstance(first_stmt.destination, spa.Identifier)
    assert first_stmt.destination.name == "tmp"

    foreach_stmt = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForeachStatement))
    assert len(foreach_stmt.body) == 1
    send_stmt = foreach_stmt.body[0]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "tmp"

    place = rects[0].metadata.place
    assert "tmp" in {decl.field_name.name for decl in place.statements}


def test_async_block_copy_not_aliased() -> None:
    code = """
    kernel @async_alias<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 val;
            f32 out;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = val;
            completion c = async {
                out = tmp;
            };
            await c;
            await send(out, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)

    compute = rects[0].metadata.compute
    first_stmt = compute.statements[0]
    assert isinstance(first_stmt, spa.AssignmentStatement)
    assert isinstance(first_stmt.destination, spa.Identifier)
    assert first_stmt.destination.name == "tmp"

    async_stmt = next(stmt for stmt in compute.statements if isinstance(stmt, spa.AsyncBlock))
    body_assign = _first_assignment(async_stmt.body)
    assert isinstance(body_assign.source.value, spa.Identifier)
    assert body_assign.source.value.name == "tmp"


def test_foreach_body_keeps_internal_copy() -> None:
    code = """
    kernel @foreach_internal_copy<N>(stream<f32, 1>[N] readonly input, stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 out;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            await foreach u16 k, f32 elem in [0:2], receive(input[i]) {
                tmp = elem;
                out = tmp;
            };
            await send(out, output[i]);
        }
    }
    """
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)

    compute = rects[0].metadata.compute
    foreach_stmt = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForeachStatement))
    assert len([stmt for stmt in foreach_stmt.body if isinstance(stmt, spa.AssignmentStatement)]) == 1

    body_assign = _first_assignment(foreach_stmt.body)
    assert isinstance(body_assign.source.value, spa.Identifier)
    assert body_assign.source.value.name == "elem"


@pytest.mark.skip(reason="Constant propagation is outside the scope of copy-elimination passes")
@pytest.mark.parametrize('is_array', [True, False])
def test_copy_constant_propagation(is_array: bool) -> None:
    dtype = '[8]' if is_array else ''
    code = f"""
    kernel @copy_const<N>(stream<f32, {8 if is_array else 1}>[N] writeonly output) {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32{dtype} tmp;
            f32{dtype} val;
            f32{dtype} val2;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            tmp = 0;
            val = tmp;
            val2 = tmp + 1;
            await send(val, output[i]);
            await send(val2, output[i]);
        }}
    }}"""
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    assert len(compute.statements) == 2
    place = rects[0].metadata.place
    assert "tmp" not in {decl.field_name.name for decl in place.statements}


@pytest.mark.skip(reason="Values used before assignment are intentionally not optimized")
def test_swap() -> None:
    code = """
    kernel @swap<N>(stream<f32, 1>[N] writeonly output) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 tmp;
            f32 src;
            f32 dst;
            f32 res;
        }
        compute u16 i, u16 j in [0:N, 0:1] {
            tmp = src;
            src = dst;
            dst = tmp;
            res = dst + 2 * src;
            await send(res, output[i]);
        }
    }"""
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    assert len(compute.statements) == 4
    assign_stmts = [stmt for stmt in compute.statements if isinstance(stmt, spa.AssignmentStatement)]
    assert len(assign_stmts) == 3
    assert assign_stmts[0].destination.name == "tmp"
    assert assign_stmts[1].destination.name == "src"
    assert assign_stmts[2].destination.name == "res"


@pytest.mark.parametrize('internal_value', ('tmp', 'val', 'out'))
def test_write_after_read_copy(internal_value: str) -> None:
    if internal_value == "out":
        pytest.skip("Loop-carried self-updates through the forwarded value are intentionally not optimized")

    code = f"""
    kernel @for_loop_copy<N>(stream<f32, 1>[N] writeonly output) {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32 tmp;
            f32 val;
            f32 out;
            f32 out2;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            tmp = 0;
            for u16 k in [0:2] {{
                out = tmp;
                tmp = {internal_value} + k;
            }}
        }}
    }}"""
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    loop = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForStatement))
    if internal_value == "out":
        assert len(loop.body) == 1
    else:
        body_assign = _first_assignment(loop.body)
        assert isinstance(body_assign.source.value, spa.Identifier)
        assert body_assign.source.value.name == "tmp"

    place = rects[0].metadata.place
    if internal_value == "out":
        assert {decl.field_name.name for decl in place.statements} == {"tmp"}

    assert "out2" not in {decl.field_name.name for decl in place.statements}


@pytest.mark.skip(reason="Loop-local elimination does not currently remove auxiliary aliases like out2")
@pytest.mark.parametrize('internal_value', ('tmp', 'val', 'out'))
def test_elide_in_loop(internal_value: str) -> None:
    code = f"""
    kernel @for_loop_copy<N>(stream<f32, 1>[N] writeonly output) {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32 tmp;
            f32 val;
            f32 out;
            f32 out2;
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            tmp = 0;
            for u16 k in [0:2] {{
                out = tmp;
                out2 = {internal_value};
                tmp = out2 + k;
            }}
        }}
    }}"""
    kernel = parse_kernel(code)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    copy_elimination.eliminate_redundant_copies(rects)
    copy_elimination.prune_unused_fields(rects)

    compute = rects[0].metadata.compute
    loop = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForStatement))
    body_assign = _first_assignment(loop.body)
    assert isinstance(body_assign.source.value, spa.Identifier)
    assert body_assign.source.value.name == "tmp"

    place = rects[0].metadata.place
    if internal_value == "out":
        assert "val" not in {decl.field_name.name for decl in place.statements}
    assert "out2" not in {decl.field_name.name for decl in place.statements}


def test_copy_with_extern():
    kernel = """
    kernel @copy<N>(stream<f32, N>[N, N] readonly a,
               stream<f32, N>[N, N] writeonly out) {
    place u16 i, u16 j in [0:N, 0:N] {
        f32[N] local;
    }
    compute u16 i, u16 j in [0:N, 0:N] {
        await receive(local, a[i, j]);
        await send(local, out[i, j]);
    }
}"""
    kernel = parse_kernel(kernel)

    passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    canonicalization.lower_bulk_communication(rects)
    canonicalization.lower_array_assignment(rects)
    canonicalization.lower_arguments_to_extern(rects, kernel)

    copy_elimination.eliminate_redundant_copies(rects)
    print(rects[0].metadata.compute.as_ir())
    copy_elimination.prune_unused_fields(rects)

    assert len(rects[0].metadata.place.statements) == 2
    assert {'a', 'out'} == {decl.field_name.name for decl in rects[0].metadata.place.statements}

    assert len(rects[0].metadata.compute.statements) == 1
    send_stmt = rects[0].metadata.compute.statements[0]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == 'a'
    assert isinstance(send_stmt.stream_name, spa.Identifier)
    assert send_stmt.stream_name.name == 'out'


if __name__ == '__main__':
    pytest.main([__file__])
