import pytest

from spatialstencil.syntax.spatial_ir import canonicalization, irnodes as spir, parser, passes


class MetaForCounter(spir.NodeVisitor):

    def __init__(self):
        super().__init__()
        self.count = 0

    def visit_MetaForBlock(self, node: spir.MetaForBlock):
        self.count += 1
        return self.generic_visit(node)


class RelativeStreamOffsetCollector(spir.NodeVisitor):

    def __init__(self):
        super().__init__()
        self.offsets = []

    def visit_RelativeStreamDeclaration(self, node: spir.RelativeStreamDeclaration):
        self.offsets.append((node.dx.eval(), node.dy.eval()))
        return self.generic_visit(node)


def _count_metafor_blocks(kernel: spir.Kernel) -> int:
    counter = MetaForCounter()
    counter.visit(kernel)
    return counter.count


def _collect_relative_stream_offsets(kernel: spir.Kernel):
    collector = RelativeStreamOffsetCollector()
    collector.visit(kernel)
    return sorted(collector.offsets)


def _inline_metaprogramming(kernel: spir.Kernel, **parameters: int) -> spir.Kernel:
    if parameters:
        kernel = passes.concretize_parameters(kernel, **parameters)
    return canonicalization.inline_metaprogramming(kernel)


def _collect_rectangles(kernel: spir.Kernel):
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.inline_phases(kernel)
    return canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)


def _rectangle_bounds(rectangles):
    return sorted(rect.metadata.compute.get_grid_rect() for rect in rectangles)


def _parse_inline_collect(kernel_str: str, **parameters: int):
    kernel = parser.parse_string(kernel_str, 'test.sptl')
    kernel = _inline_metaprogramming(kernel, **parameters)
    rectangles = _collect_rectangles(kernel)
    return kernel, rectangles


@pytest.mark.parametrize(
    ('loop_range', 'tile_width', 'inline_parameters'),
    [
        ('[0:4:2, 0:2]', '2', {}),
        ('[M:N:K, 0:K]', 'K', {
            'M': 0,
            'N': 4,
            'K': 2
        }),
    ],
)
def test_metafor_in_phase(loop_range, tile_width, inline_parameters):
    parameter_list = '<M, N, K>' if inline_parameters else '<>'
    kernel, rectangles = _parse_inline_collect(
        f"""
        kernel @test{parameter_list}() {{
            phase {{
                for i16 meta_i, i16 meta_j in {loop_range} {{
                    place i16 i, i16 j in [meta_i:(meta_i + {tile_width}), meta_j:(meta_j + 1)] {{
                        f32 value
                    }}
                    dataflow i16 i, i16 j in [meta_i:(meta_i + {tile_width}), meta_j:(meta_j + 1)] {{
                        stream<f32> blue = relative_stream(0, 1) {{
                            hops = [(0, 1)],
                            channel = 0
                        }}
                    }}
                    compute i16 i, i16 j in [meta_i:(meta_i + {tile_width}), meta_j:(meta_j + 1)] {{
                        value = 1
                    }}
                }}
            }}
        }}""",
        **inline_parameters,
    )

    assert _count_metafor_blocks(kernel) == 0
    assert _rectangle_bounds(rectangles) == [
        (0, 2, 0, 1),
        (0, 2, 1, 2),
        (2, 4, 0, 1),
        (2, 4, 1, 2),
    ]
    assert all(len(rect.metadata.place.statements) == 1 for rect in rectangles)
    assert all(len(rect.metadata.dataflow.statements) == 1 for rect in rectangles)
    assert all(len(rect.metadata.compute.statements) == 1 for rect in rectangles)


def test_metafor_outside_phase():
    kernel, rectangles = _parse_inline_collect(
        """
        kernel @test<>() {
            for i16 meta_i, i16 meta_j in [0:2, 0:2] {
                place i16 i, i16 j in [meta_i:(meta_i + 1), meta_j:(meta_j + 1)] {
                    f32 value
                }
                compute i16 i, i16 j in [meta_i:(meta_i + 1), meta_j:(meta_j + 1)] {
                    value = 1
                }
            }
        }
        """,)

    assert _count_metafor_blocks(kernel) == 0
    assert _rectangle_bounds(rectangles) == [
        (0, 1, 0, 1),
        (0, 1, 1, 2),
        (1, 2, 0, 1),
        (1, 2, 1, 2),
    ]


def test_metafor_outside_phase_with_inner_phase():
    hop_expr = "1 << stage"  # 2^stage
    active_nodes_expr = "1 << (K - stage + 1)"  # 2^(K - stage + 1)
    kernel, rectangles = _parse_inline_collect(
        f"""
        kernel @tree_reduce<K>() {{
            for i16 stage in [0:K] {{
                phase {{
                    place i16 i, i16 j in [0:{active_nodes_expr}, 0:1] {{
                        f32 partial
                    }}
                    dataflow i16 i, i16 j in [0:{active_nodes_expr}, 0:1] {{
                        stream<f32> parent = relative_stream({hop_expr}, 0)
                    }}
                    compute i16 i, i16 j in [0:{active_nodes_expr}, 0:1] {{
                        await send(partial, parent)
                    }}
                }}
            }}
        }}
        """,
        K=3,
    )

    assert _count_metafor_blocks(kernel) == 0
    assert _collect_relative_stream_offsets(kernel) == [(1, 0), (2, 0), (4, 0)]
    assert _rectangle_bounds(rectangles) == [
        (0, 1, 0, 1),
        (0, 2, 0, 1),
        (0, 4, 0, 1),
    ]
    assert all(len(rect.metadata.place.statements) == 1 for rect in rectangles)
    assert all(len(rect.metadata.dataflow.statements) == 1 for rect in rectangles)
    assert all(len(rect.metadata.compute.statements) == 1 for rect in rectangles)


def test_metafor_multiblock():
    kernel, rectangles = _parse_inline_collect(
        """
        kernel @test<>() {
            for i16 meta_i in [0:2] {
                dataflow i16 i, i16 j in [meta_i:(meta_i + 1), 0:1] {
                    stream<f32> blue = relative_stream(1, 0) {
                        hops = [(1, 0)],
                        channel = 0
                    }
                }
                compute i16 i, i16 j in [meta_i:(meta_i + 1), 0:1] {
                    await send(i, blue)
                }
            }
        }
        """,)

    assert _count_metafor_blocks(kernel) == 0
    assert _rectangle_bounds(rectangles) == [
        (0, 1, 0, 1),
        (1, 2, 0, 1),
    ]
    assert all(len(rect.metadata.dataflow.statements) == 1 for rect in rectangles)
    assert all(len(rect.metadata.compute.statements) == 1 for rect in rectangles)


def test_nested_metafor():
    kernel, rectangles = _parse_inline_collect(
        """
        kernel @test<>() {
            for i16 meta_i in [0:2] {
                for i16 meta_j in [0:2] {
                    place i16 i, i16 j in [meta_i:(meta_i + 1), meta_j:(meta_j + 1)] {
                        f32 value
                    }
                    compute i16 i, i16 j in [meta_i:(meta_i + 1), meta_j:(meta_j + 1)] {
                        value = 1
                    }
                }
            }
        }
        """,)

    assert _count_metafor_blocks(kernel) == 0
    assert _rectangle_bounds(rectangles) == [
        (0, 1, 0, 1),
        (0, 1, 1, 2),
        (1, 2, 0, 1),
        (1, 2, 1, 2),
    ]
    assert all(len(rect.metadata.place.statements) == 1 for rect in rectangles)


def test_recursive_metafor_with_phase_and_inner_metafor():
    kernel, rectangles = _parse_inline_collect(
        """
        kernel @test<>() {
            for i16 meta_i in [0:2] {
                phase {
                    for i16 meta_j in [0:2] {
                        place i16 i, i16 j in [meta_i:(meta_i + 1), meta_j:(meta_j + 1)] {
                            f32 value
                        }
                        compute i16 i, i16 j in [meta_i:(meta_i + 1), meta_j:(meta_j + 1)] {
                            value = 1
                        }
                    }
                }
            }
        }
        """,)

    assert _count_metafor_blocks(kernel) == 0
    assert _rectangle_bounds(rectangles) == [
        (0, 1, 0, 1),
        (0, 1, 1, 2),
        (1, 2, 0, 1),
        (1, 2, 1, 2),
    ]
    assert all(len(rect.metadata.place.statements) == 1 for rect in rectangles)
    assert all(len(rect.metadata.compute.statements) == 1 for rect in rectangles)


def test_metafor_invalidrange():
    """
    Tests that a metafor with a non-compile-time-evaluatable bound raises an error.
    """
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            for i16 meta_i in [0:unknown] {
                compute i16 i, i16 j in [0:1, 0:1] {
                }
            }
        }
        """,
        'test.sptl',
    )

    with pytest.raises((TypeError, ValueError), match='compile|constant|integral|unknown'):
        _inline_metaprogramming(kernel)


if __name__ == '__main__':
    pytest.main([__file__])
