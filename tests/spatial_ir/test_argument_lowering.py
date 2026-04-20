import pytest
from spada.syntax.spatial_ir import canonicalization, irnodes as spir, parser, passes


@pytest.mark.parametrize("streaming", (False, True))
def test_lower_arguments(streaming):
    suffix = ", 1" if not streaming else ""
    kernel_code = f"""
kernel @copy<N>(stream<f32{suffix}>[N, N] readonly a,
               stream<f32{suffix}>[N, N] writeonly out) {{
    place u16 i, u16 j in [0:N, 0:N] {{
        f32 local;
    }}
    compute u16 i, u16 j in [0:N, 0:N] {{
        await receive(local, a[i, j]);
        await send(local, out[i, j]);
    }}
}}
    """
    kernel = parser.parse_string(kernel_code, "test.sptl")
    kernel = passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    assert len(rects) == 1
    canonicalization.lower_arguments_to_extern(rects, kernel)

    if streaming:
        # Check dataflow block for extern streams
        dataflow_block = rects[0].metadata.dataflow
        extern_streams = [
            stmt for stmt in dataflow_block.statements if isinstance(stmt.stream, spir.ExternStreamDeclaration)
        ]
        assert len(extern_streams) == 2
        extern_a = extern_streams[0]
        extern_out = extern_streams[1]
        assert extern_a.stream_name.name == "a"
        assert extern_out.stream_name.name == "out"
        assert extern_a.stream.direction == "in"
        assert extern_out.stream.direction == "out"
    else:
        # Check place block for extern fields
        place_block = rects[0].metadata.place
        extern_fields = [stmt for stmt in place_block.statements if stmt.is_extern]
        assert len(extern_fields) == 2
        extern_a = extern_fields[0]
        extern_out = extern_fields[1]
        assert extern_a.field_name.name == "a"
        assert extern_out.field_name.name == "out"


@pytest.mark.parametrize("streaming", (False, True))
def test_lower_arguments_with_scalar(streaming):
    suffix = ", 1" if not streaming else ""
    kernel_code = f"""
kernel @mult_scalar<N>(stream<f32{suffix}>[N, N] readonly a, f32 coeff, 
                       stream<f32{suffix}>[N, N] writeonly out) {{
    place u16 i, u16 j in [0:N, 0:N] {{
        f32 local_a;
    }}
    compute u16 i, u16 j in [0:N, 0:N] {{
        await receive(local_a, a[i, j]);
        local_a = local_a * coeff;
        await send(local_a, out[i, j]);
    }}
}}
    """
    kernel = parser.parse_string(kernel_code, "test.sptl")
    kernel = passes.concretize_parameters(kernel, N=8)
    rects = canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)
    assert len(rects) == 1
    canonicalization.lower_arguments_to_extern(rects, kernel)

    if streaming:
        # Check dataflow block for extern streams
        dataflow_block = rects[0].metadata.dataflow
        extern_streams = [
            stmt for stmt in dataflow_block.statements if isinstance(stmt.stream, spir.ExternStreamDeclaration)
        ]
        assert len(extern_streams) == 2
        extern_a = extern_streams[0]
        extern_out = extern_streams[1]
        assert extern_a.stream_name.name == "a"
        assert extern_out.stream_name.name == "out"
        assert extern_a.stream.direction == "in"
        assert extern_out.stream.direction == "out"
    else:
        # Check place block for extern fields
        place_block = rects[0].metadata.place
        extern_fields = [stmt for stmt in place_block.statements if stmt.is_extern]
        assert len(extern_fields) == 2
        extern_a = extern_fields[0]
        extern_out = extern_fields[1]
        assert extern_a.field_name.name == "a"
        assert extern_out.field_name.name == "out"


if __name__ == "__main__":
    pytest.main([__file__])
