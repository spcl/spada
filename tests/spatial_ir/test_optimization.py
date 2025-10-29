from spatialstencil.syntax.spatial_ir import irnodes as spa
from spatialstencil.syntax.spatial_ir import parser, passes


def parse_kernel(code: str) -> spa.Kernel:
    return parser.parse_string(code, "test.sptl")


def _get_block(kernel: spa.Kernel, block_type: type) -> spa.SpatialNode:
    for stmt in kernel.body:
        if isinstance(stmt, block_type):
            return stmt
    raise AssertionError(f"No block of type {block_type.__name__} found")


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

    passes.eliminate_extraneous_copies(kernel)
    passes.prune_unused_fields(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    assert len(compute.statements) == 1
    send_stmt = compute.statements[0]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "val"

    place = _get_block(kernel, spa.PlaceBlock)
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

    passes.eliminate_extraneous_copies(kernel)
    passes.prune_unused_fields(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    assert len(compute.statements) == 1
    send_stmt = compute.statements[0]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "val"

    place = _get_block(kernel, spa.PlaceBlock)
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

    passes.eliminate_extraneous_copies(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    assert len(compute.statements) == 2
    first_stmt, send_stmt = compute.statements
    assert isinstance(first_stmt, spa.AssignmentStatement)
    assert isinstance(first_stmt.destination, spa.Identifier)
    assert first_stmt.destination.name == "tmp"
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

    passes.eliminate_extraneous_copies(kernel)
    passes.prune_unused_fields(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    send_stmt = compute.statements[-1]
    assert isinstance(send_stmt, spa.SendStatement)
    assert isinstance(send_stmt.local_array, spa.Identifier)
    assert send_stmt.local_array.name == "src"

    place = _get_block(kernel, spa.PlaceBlock)
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

    passes.eliminate_extraneous_copies(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    maps = [stmt for stmt in compute.statements if isinstance(stmt, spa.MapStatement)]
    assert len(maps) == 1
    place = _get_block(kernel, spa.PlaceBlock)
    assert "tmp" in {decl.field_name.name for decl in place.statements}


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

    passes.eliminate_extraneous_copies(kernel)
    passes.prune_unused_fields(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    loop = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForStatement))
    body_assign = _first_assignment(loop.body)
    assert isinstance(body_assign.source.value, spa.Identifier)
    assert body_assign.source.value.name == "val"

    place = _get_block(kernel, spa.PlaceBlock)
    assert "tmp" not in {decl.field_name.name for decl in place.statements}


def test_rename_propagates_into_foreach_loop() -> None:
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

    passes.eliminate_extraneous_copies(kernel)
    passes.prune_unused_fields(kernel)

    compute = _get_block(kernel, spa.ComputeBlock)
    foreach_stmt = next(stmt for stmt in compute.statements if isinstance(stmt, spa.ForeachStatement))
    body_assign = _first_assignment(foreach_stmt.body)
    assert isinstance(body_assign.source.value, spa.Identifier)
    assert body_assign.source.value.name == "val"

    place = _get_block(kernel, spa.PlaceBlock)
    assert "tmp" not in {decl.field_name.name for decl in place.statements}


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
