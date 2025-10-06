from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

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


class ProgramDataflow:

    domain_collector: DomainCollector
    versioning: Versioning[spa.Identifier]
    # Maps [input_field][output_field][offset] -> stream
    # the destination field is the first 
    _stream_map: dict[sast.Identifier, dict[sast.Identifier, dict[sast.Offset, spa.Identifier]]]

    # stream -> x-y range where the stream sends
    stream_send_range_map: dict[spa.Identifier, tuple[tuple[int, int, int], tuple[int, int, int]]]
    # stream -> x-y range where the stream receives
    stream_receive_range_map: dict[spa.Identifier, tuple[tuple[int, int, int], tuple[int, int, int]]]

    def __init__(self,
                 domain_shift: tuple,
                 versioning: Versioning[spa.Identifier],
                 grid_var_type: ScalarType = ScalarType.u16):
        self.versioning = versioning
        self.domain_shift = domain_shift
        self._stream_map = defaultdict(lambda: defaultdict(dict))
        self.grid_var_t = grid_var_type
        self.stream_send_range_map = dict()
        self.stream_receive_range_map = dict()

    def get_stream(self,
                   input_id: sast.Identifier,
                   output_id: sast.Identifier,
                   offset: sast.Offset) -> spa.Identifier | None:
        if input_id not in self._stream_map:
            return None
        if output_id not in self._stream_map[input_id]:
            return None
        if offset not in self._stream_map[input_id][output_id]:
            return None
        return self._stream_map[input_id][output_id][offset]

    def _set_stream(self,
                    input_id: sast.Identifier,
                    output_id: sast.Identifier,
                    offset: sast.Offset,
                    stream: spa.Identifier):
        self._stream_map[input_id][output_id][offset] = stream

    def declare_dataflow_for_computation(self,
                                         comp: sast.ComputationBlock) -> list[spa.DataflowBlock]:
        """
        Generate dataflow blocks for a computation block.
        """

        # Keep track of a mapping from statements (or views) to participating streams
        # For every statement generate a stream for each non-zero extent

        abstract_streams = []

        for stmt in comp.body:

            if isinstance(stmt, sast.MaterializeOp):
                out_t = stmt.operation_type.destination[0]

                for extent in stmt.operation_type.destination[0].extent.extents:
                    dx = -extent.values[0]
                    dy = -extent.values[1]
                    assert isinstance(dx, int)
                    assert isinstance(dy, int)
                    if dx or dy:
                        stream_type = spa.StreamType(stmt.operation_type.destination[0].dtype)
                        identifier = self.versioning.next_version(f'_stream_{stmt.result.name}')

                        metadata = StreamMetadata(
                            stream_type,
                            identifier,
                            dx,
                            dy
                        )
                        self._set_stream(stmt.value, stmt.result, extent, identifier)

                        # Generate stream
                        x_range, y_range = self.get_x_y_range(out_t, -dx, -dy)
                        self.stream_send_range_map[identifier] = self.get_x_y_send_range(out_t, -dx, -dy)
                        self.stream_receive_range_map[identifier] = self.get_x_y_receive_range(out_t, -dx, -dy)

                        astream = AbstractStream(x_range, y_range, metadata)
                        abstract_streams.append(astream)

            if isinstance(stmt, sast.StatementBlock):
                out_t = stmt.operation_type.destination[0]

                for access, access_type in zip(stmt.inputs, stmt.operation_type.source):
                    if isinstance(access_type, sast.ViewType) and any(access == inp for inp in comp.inputs):
                        for extent in access_type.extent.extents:
                            dx = -extent.values[0]
                            dy = -extent.values[1]
                            assert isinstance(dx, int)
                            assert isinstance(dy, int)
                            if dx or dy:
                                stream_type = spa.StreamType(access_type.dtype)
                                identifier = self.versioning.next_version(f'_stream_{access.name}')

                                metadata = StreamMetadata(
                                    stream_type,
                                    identifier,
                                    dx,
                                    dy
                                )

                                self._set_stream(access, stmt.outputs[0], extent, identifier)

                                # Generate stream
                                x_range, y_range = self.get_x_y_range(out_t, -dx, -dy)

                                self.stream_send_range_map[identifier] = self.get_x_y_send_range(out_t, -dx, -dy)
                                self.stream_receive_range_map[identifier] = self.get_x_y_receive_range(out_t,- dx, -dy)

                                astream = AbstractStream(x_range, y_range, metadata)
                                abstract_streams.append(astream)

        blocks = self._abstract_declarations_to_block(abstract_streams)

        return blocks


    def _abstract_declarations_to_block(self, abstract_streams: list[AbstractStream]) -> list[spa.DataflowBlock]:

        abstract_streams = split_rectangles(abstract_streams)
        grouped = group_rectangles_by_domain(abstract_streams)

        blocks = []
        # Generate dataflow blocks from the abstract declarations
        for group in grouped:
            declarations = []

            x_range = group[0].x_range
            y_range = group[0].y_range

            for rect in group:

                stream = spa.RelativeStreamDeclaration(
                    dtype=rect.metadata.stream_type,
                    stream_name=rect.metadata.identifier,
                    dx=spa.Expression(spa.ConstantLiteral(rect.metadata.dx, dtype=ScalarType.i32)),
                    dy=spa.Expression(spa.ConstantLiteral(rect.metadata.dy, dtype=ScalarType.i32)),
                )
                declarations.append(stream)

            var_i = self.versioning.next_version("i")
            var_j = self.versioning.next_version("j")

            subgrid = spa.SubgridExpression.from_tuple(x_range, y_range)

            block = spa.DataflowBlock(variables=[spa.TypedIdentifier(self.grid_var_t, var_i),
                                                 spa.TypedIdentifier(self.grid_var_t, var_j)],
                                      subgrid=subgrid,
                                      statements=declarations)
            blocks.append(block)

        return blocks

    def get_x_y_send_range(self, out_t: sast.ViewType | sast.FieldType, dx: int, dy: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Defines the subgrid that sends for the given type and stream offset (dx, dy)
        """
        assert isinstance(out_t.domain, sast.Cartesian)
        
        # We need a buffer of + the extent around the domain
        send_domain = out_t.domain.add((dx, dy, 0))
        x_range = (send_domain.x[0] + self.domain_shift[0],
                   send_domain.x[1] + self.domain_shift[0],
                   1)
        y_range = (send_domain.y[0] + self.domain_shift[1],
                   send_domain.y[1] + self.domain_shift[1],
                   1)

        assert x_range[0] >= 0, f"Type {out_t.as_ir()} at {dx}, {dy} has invalid send range"
        assert x_range[1] >= x_range[0], f"Type {out_t.as_ir()} at {dx}, {dy} has invalid send range"
        assert y_range[0] >= 0, f"Type {out_t.as_ir()} at {dx}, {dy} has invalid send range"
        assert y_range[1] >= y_range[0], f"Type {out_t.as_ir()} at {dx}, {dy} has invalid send range"

        return x_range, y_range
    
    def get_x_y_receive_range(self, out_t: sast.ViewType | sast.FieldType, dx: int, dy: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Defines the subgrid receives for the given type and stream offset (dx, dy)
        """
        assert isinstance(out_t.domain, sast.Cartesian)
                
        # We need a buffer of - the extent around the domain
        send_domain = out_t.domain
        x_range = (send_domain.x[0] + self.domain_shift[0],
                   send_domain.x[1] + self.domain_shift[0],
                   1)
        y_range = (send_domain.y[0] + self.domain_shift[1],
                   send_domain.y[1] + self.domain_shift[1],
                   1)

        assert x_range[0] >= 0, f"Type {out_t.as_ir()} at {dx}, {dy} has invalid receive range"
        assert x_range[1] >= x_range[0], f"Type {out_t.as_ir()} at {dx}, {dy} has invalid receive range"
        assert y_range[0] >= 0, f"Type {out_t.as_ir()} at {dx}, {dy} has invalid receive range"
        assert y_range[1] >= y_range[0], f"Type {out_t.as_ir()} at {dx}, {dy} has invalid receive range"

        return x_range, y_range

    def get_x_y_range(self, out_t: sast.ViewType | sast.FieldType, dx: int, dy: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Defines the subgrid sends OR receives for the given type and stream offset (dx, dy)
        """
        assert isinstance(out_t.domain, sast.Cartesian)

        # We need a buffer of +- the extent around the domain
        send_domain = out_t.domain.union(out_t.domain.add((dx, dy, 0)))
        #print(f"Send {send_domain}")
        x_range = (send_domain.x[0] + self.domain_shift[0],
                   send_domain.x[1] + self.domain_shift[0],
                   1)
        y_range = (send_domain.y[0] + self.domain_shift[1],
                   send_domain.y[1] + self.domain_shift[1],
                   1)

        assert x_range[0] >= 0, f"Type {out_t.as_ir()} at {dx}, {dy} has invalid range {x_range}"
        assert x_range[1] >= x_range[0], f"Type {out_t.as_ir()} at {dx}, {dy} has invalid range"
        assert y_range[0] >= 0, f"Type {out_t.as_ir()} at {dx}, {dy} has invalid range"
        assert y_range[1] >= y_range[0], f"Type {out_t.as_ir()} at {dx}, {dy} has invalid range"

        return x_range, y_range
