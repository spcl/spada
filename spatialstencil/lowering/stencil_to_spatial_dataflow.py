from dataclasses import dataclass

import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.spatial_ir.irnodes as spa
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement

from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.common.types import ScalarType
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle, split_rectangles, group_rectangles_by_domain

from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector


@dataclass(frozen=True)
class StreamMetadata:
    """
    Metadata for a stream.
    """
    stream_type: spa.StreamType
    identifier: spa.Identifier
    dx: int
    dy: int


AbstractStream = Rectangle[StreamMetadata]


def declare_dataflow_for_computation(comp: sast.ComputationBlock,
                                     versioning: Versioning[spa.Identifier],
                                     offset_domain: tuple[int, int]) -> list[spa.DataflowBlock]:

    # For every statement generate a stream for each non-zero extent

    abstract_streams = []

    for stmt in comp.body:
        if isinstance(stmt, sast.StatementBlock):
            for acess, access_type in zip(stmt.inputs, stmt.operation_type.source):
                if isinstance(access_type, sast.ViewType):
                    for extent in access_type.extent.extents:
                        dx = -extent.values[0]
                        dy = -extent.values[1]
                        assert isinstance(dx, int)
                        assert isinstance(dy, int)
                        if dx or dy:
                            stream_type = spa.StreamType(access_type.dtype)
                            identifier = versioning.next_version(f'_stream_{acess.name}')

                            metadata = StreamMetadata(
                                stream_type,
                                identifier,
                                dx,
                                dy
                            )
                            # Generate stream
                            assert isinstance(access_type.domain, sast.Cartesian)
                            x_range = (access_type.domain.x[0]+offset_domain[0], access_type.domain.x[1]+offset_domain[1])
                            y_range = (access_type.domain.y[0]+offset_domain[0], access_type.domain.y[1]+offset_domain[1])
                            astream = AbstractStream(x_range, y_range, metadata)
                            abstract_streams.append(astream)

    abstract_streams = split_rectangles(abstract_streams)
    grouped = group_rectangles_by_domain(abstract_streams)

    blocks = []

    for group in grouped:
        # Generate a dataflow block from the abstract declaration
        declarations = []

        x_range = group[0].x_range
        y_range = group[0].y_range

        for rect in group:

            stream = spa.RelativeStreamDeclaration(
                dtype=rect.metadata.stream_type,
                stream_name=rect.metadata.identifier,
                dx=spa.Expression(spa.ConstantLiteral(rect.metadata.dx, dtype=ScalarType.i32), ScalarType.i32),
                dy=spa.Expression(spa.ConstantLiteral(rect.metadata.dy, dtype=ScalarType.i32), ScalarType.i32)
            )
            declarations.append(stream)

        var_i = versioning.next_version("_i")
        var_j = versioning.next_version("_j")

        subgrid = spa.SubgridExpression.from_tuple(x_range, y_range)

        block = spa.DataflowBlock(variables=[var_i, var_j],
                                  subgrid=subgrid,
                                  statements=declarations)
        blocks.append(block)

    return blocks
