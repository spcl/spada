import copy
import itertools
from enum import Enum, auto
from spada.lowering.versioning import Versioning
import spada.syntax.spatial_ir.irnodes as spa
from spada.syntax.common.symbolic import is_symbolic
from spada.syntax.spatial_ir.canonicalization import canonicalize_phases, inline_phases
from spada.syntax.spatial_ir.symbolic_grid import coord_of_expr


class ChannelStrategy(Enum):
    NONE = 0
    TRIVIAL = 1


class KernelRouting:
    """
    Lowering Pass for generating routing assignments for a SPADA kernel
    """
    versioning: Versioning[spa.Identifier]

    def __init__(self, versioning: Versioning[spa.Identifier]):
        self.versioning = versioning

    def split_blocks(self, kernel: spa.Kernel) -> spa.Kernel:
        """Splits the blocks of the kernel to allow introduction of a checkerboard pattern.
        That is, for each dimension that is being communicated in, it splits all blocks into two blocks, an even and an odd block.

        For now, only single hop communication is supported!
        However, multiple single-hop steps can be chained.
        Args:
            kernel (spa.Kernel): Kernel to split

        Returns:
            spa.Kernel: The split kernel, semantically equivalent.
        """

        # Determine the active dimensions (which dimensions to split)
        # Also, assert that |dx|<= 1 and |dy|<=1, we do not support multiple hops (yet)

        dxdy_visitor = DxDyVisitor()
        dxdy_visitor.visit(kernel)

        if dxdy_visitor.max_dx > 1 or dxdy_visitor.max_dy > 1:
            raise NotImplementedError("Only single hop communication is supported.")

        active_dimensions = [dxdy_visitor.max_dx > 0, dxdy_visitor.max_dy > 0]

        # Do the split

        transformer = SplitTransformer(active_dimensions, self.versioning)

        # Canonicalize: Implicitly sorts the kernel to ensure we visit compute blocks last
        kernel = canonicalize_phases(kernel)
        kernel = inline_phases(kernel)

        # Split the kernel in preparation for coloring
        transformed_kernel = transformer.visit(kernel)

        return transformed_kernel

    def generate_routing(self,
                         kernel: spa.Kernel,
                         channel_strategy: ChannelStrategy = ChannelStrategy.TRIVIAL) -> spa.Kernel:
        """Generates routing blocks, possibly splitting and restructuring the kernel blocks

        Args:
            kernel (spa.Kernel): _description_

        Returns:
            spa.Kernel: _description_
        """
        if channel_strategy is not ChannelStrategy.NONE:
            kernel = self.split_blocks(kernel)

        # Gather the stream declarations
        stream_visitor = StreamVisitor()
        stream_visitor.visit(kernel)

        # Trivial Coloring
        channel_map: dict[spa.Identifier, int] = dict()
        hops_map: dict[str, list[tuple[int, int]]] = dict()

        if channel_strategy.name == ChannelStrategy.TRIVIAL.name:
            color = 0
            for stream in stream_visitor.streams.keys():
                channel_map[stream] = color
                color += 1

        for stream in stream_visitor.streams.keys():
            if stream not in hops_map:
                if isinstance(stream_visitor.streams[stream].stream, spa.RelativeStreamDeclaration):
                    hops_map[stream] = self._shortest_path_routing(stream_visitor.streams[stream].stream.dx.eval(),
                                                                   stream_visitor.streams[stream].stream.dy.eval())

        routing_trans = StreamRoutingTransformer(channel_map, hops_map)
        routing_trans.visit(kernel)

        return kernel

    @staticmethod
    def _shortest_path_routing(dx: int, dy: int) -> list[spa.RoutingHop]:

        if dx > 0:
            result = [spa.RoutingHop((1, 0)) for _ in range(dx)]
        else:
            result = [spa.RoutingHop((-1, 0)) for _ in range(-dx)]

        if dy > 0:
            result.extend([spa.RoutingHop((0, 1)) for _ in range(dy)])
        else:
            result.extend([spa.RoutingHop((0, -1)) for _ in range(-dy)])

        return result


class StreamRoutingTransformer(spa.NodeTransformer):
    channel_map: dict[spa.Identifier, int] = dict()
    hops_map: dict[str, list[tuple[int, int]]] = dict()

    def __init__(self, channel_map: dict, hops_map: dict):
        super().__init__()
        self.channel_map = channel_map
        self.hops_map = hops_map

    def visit_StreamDeclaration(self, stmt: spa.StreamDeclaration):

        routing = spa.RoutingDeclaration(
            hops=copy.deepcopy(self.hops_map[stmt.stream_name]),
            channel=self.channel_map[stmt.stream_name] if stmt.stream_name in self.channel_map else 'auto')

        stmt.stream.routing = routing
        return stmt


class StreamVisitor(spa.NodeVisitor):

    streams: dict[spa.Identifier, spa.StreamDeclaration]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streams = dict()

    def visit_StreamDeclaration(self, stmt: spa.StreamDeclaration):
        if stmt.stream_name not in self.streams:
            self.streams[stmt.stream_name] = stmt


class DxDyVisitor(spa.NodeVisitor):

    max_dx: int
    max_dy: int

    def __init__(self):
        self.max_dx = 0
        self.max_dy = 0
        super().__init__()

    def visit_RelativeStreamDeclaration(self, s: spa.RelativeStreamDeclaration):
        self.max_dx = max(self.max_dx, abs(s.dx.eval()))
        self.max_dy = max(self.max_dy, abs(s.dy.eval()))


class _StreamUseCollector(spa.NodeVisitor):
    """
    Collects the identifiers of all streams that are sent to or received from in a block.
    """

    def __init__(self):
        super().__init__()
        self.stream_names: list[spa.Identifier] = []

    def _add(self, name):
        if isinstance(name, spa.Identifier) and name not in self.stream_names:
            self.stream_names.append(name)

    def visit_SendStatement(self, stmt: spa.SendStatement):
        self._add(stmt.stream_name)

    def visit_ReceiveStatement(self, stmt: spa.ReceiveStatement):
        self._add(stmt.stream_name)

    def visit_ReceiveGenerator(self, gen: spa.ReceiveGenerator):
        self._add(gen.stream_name)


class SplitTransformer(spa.NodeTransformer):
    """
    Splits the blocks of a kernel according to a checkerboard pattern.
    Each active dimension is split in 2.
    This results in 1, 2, or 4 blocks.
    
    Each stream s is duplicated into s_even and s_odd. The key insight: messages from 
    even-coordinate PEs travel only through even-coordinate intermediate PEs, while messages 
    from odd-coordinate PEs traverse only odd-coordinate PEs. This spatial separation 
    eliminates all routing conflicts.

    Note: even/odd streams are implemented using versioning of the stream identifier.

    Stream Selection:
    At each send or receive operation on stream s = relative_stream(dx, dy), we replace s 
    with either s_even or s_odd deterministically:

        1. Identify the communication dimension: x if |dx| > 0, otherwise y
        2. Compute block parity: p = i mod 2 for x-communication, p = j mod 2 for y-communication
        3. Determine direction: σ = 1 if dx > 0 (or dy > 0), otherwise σ = 0
        4. If receiving, flip direction: σ = 1 - σ (logical reversal)
        5. Select: use s_even if p = σ, otherwise s_odd
    
    """

    ## Keep track of the mapping from original stream names to their split streams (even & odd stream)
    stream_map: dict[spa.Identifier, list[spa.StreamDeclaration]]

    # Dimensions to split
    active_dimensions: list[bool]

    versioning: Versioning[spa.Identifier]

    _active_compute_block: spa.ComputeBlock | None

    def __init__(self,
                 active_dimensions: list[bool],
                 versioning: Versioning[spa.Identifier],
                 int_type: spa.ScalarType = spa.ScalarType.i16):
        super().__init__()
        self.active_dimensions = active_dimensions
        self.stream_map = dict()
        self.versioning = versioning
        self._active_compute_block = None
        self._parity_override: dict[str, int] = {}
        self.int_type = int_type

    def _split_subgrid_x(self, subgrid: spa.SubgridExpression) -> list[spa.SubgridExpression]:
        assert self.active_dimensions[0]

        x_step = spa.Expression(spa.ConstantLiteral(2, self.int_type))

        first = spa.SubgridExpression(
            spa.RangeExpression(subgrid.x_range.start, subgrid.x_range.stop, x_step), subgrid.y_range)

        result = [first]

        x_start = coord_of_expr(subgrid.x_range.start) + 1

        if x_start < coord_of_expr(subgrid.x_range.stop):

            second = spa.SubgridExpression(
                spa.RangeExpression(
                    spa.expression_of(x_start, self.int_type), subgrid.x_range.stop, x_step),
                subgrid.y_range)

            result.append(copy.deepcopy(second))

        return result

    def _split_subgrid_y(self, subgrid: spa.SubgridExpression) -> list[spa.SubgridExpression]:
        assert self.active_dimensions[1]

        y_step = spa.Expression(spa.ConstantLiteral(2, self.int_type))

        first = spa.SubgridExpression(
            subgrid.x_range,
            spa.RangeExpression(subgrid.y_range.start, subgrid.y_range.stop, y_step),
        )

        result = [copy.deepcopy(first)]

        y_start_2 = coord_of_expr(subgrid.y_range.start) + 1

        if y_start_2 < coord_of_expr(subgrid.y_range.stop):
            y_start = spa.expression_of(y_start_2, self.int_type)

            second = spa.SubgridExpression(
                subgrid.x_range,
                spa.RangeExpression(y_start, subgrid.y_range.stop, y_step),
            )

            result.append(copy.deepcopy(second))

        return result

    def _split_subgrid(self, subgrid: spa.SubgridExpression) -> list[spa.SubgridExpression]:
        """
        Splits a subgrid according to the active dimensions.
        
        Creates 2^(active_dimensions) many splits
        
        """
        assert subgrid.x_range.step == None or subgrid.x_range.step.eval() == 1
        assert subgrid.y_range.step == None or subgrid.y_range.step.eval() == 1
        if not self.active_dimensions[0] and not self.active_dimensions[1]:
            return [subgrid]

        grids = [subgrid]
        if self.active_dimensions[0]:
            grids = self._split_subgrid_x(subgrid)
        if self.active_dimensions[1]:
            new_grids = []
            for g in grids:
                new_grids.extend(self._split_subgrid_y(g))
            grids = new_grids

        return grids

    def _split_block(
        self, block: spa.PlaceBlock | spa.DataflowBlock | spa.ComputeBlock
    ) -> list[spa.PlaceBlock | spa.DataflowBlock | spa.ComputeBlock]:
        """Splits a block according to the active dimensions (copies its contents 1:1)

        Args:
            block (spa.PlaceBlock | spa.DataflowBlock | spa.ComputeBlock): The block to split

        Returns:
            list[spa.PlaceBlock | spa.DataflowBlock | spa.ComputeBlock]: The list of split blocks (1, 2 or 4)
        """
        grids = self._split_subgrid(block.subgrid)

        blocks = [copy.deepcopy(block) for _ in range(len(grids))]
        for g, b in zip(grids, blocks):
            b.subgrid = g

        return blocks

    def visit_PlaceBlock(self, block: spa.PlaceBlock):
        # Place blocks are merely split, no other changes
        return self._split_block(block)

    def visit_DataflowBlock(self, block: spa.DataflowBlock):

        # The dataflow blocks get the new streams (even & odd)
        block = self.generic_visit(block)

        # Now, we actually do the split
        blocks = self._split_block(block)
        return blocks

    def visit_StreamDeclaration(self, stmt: spa.StreamDeclaration):
        assert isinstance(stmt.stream, spa.RelativeStreamDeclaration)
        active_x = abs(stmt.stream.dx.eval())
        active_y = abs(stmt.stream.dy.eval())
        if active_x + active_y > 1:
            raise NotImplementedError("Currently, only 1-hop communication is supported")

        if active_y:
            assert self.active_dimensions[1]
        if active_x:
            assert self.active_dimensions[0]

        assert active_x + active_y > 0

        if stmt.stream_name in self.stream_map:

            return copy.deepcopy(self.stream_map[stmt.stream_name])

        else:
            # New stream names
            even = self.versioning.next_version(stmt.stream_name.name)
            odd = self.versioning.next_version(stmt.stream_name.name)

            # Create & keep copy streams
            stmt_1 = copy.deepcopy(stmt)
            stmt_2 = copy.deepcopy(stmt)
            stmt_1.stream_name = even
            stmt_2.stream_name = odd

            self.stream_map[stmt.stream_name] = [stmt_1, stmt_2]

            return [stmt_1, stmt_2]

    def visit_ComputeBlock(self, block: spa.ComputeBlock):
        # The compute blocks need adjustment, use the even / odd stream depending on the parity of the block and the sign of the communication

        # First, split:
        split_blocks = self._split_block(block)
        result = []

        for b in split_blocks:
            axes = self._symbolic_parity_axes(b)
            if not axes:
                self._active_compute_block = b
                result.append(self.generic_visit(b))
            else:
                # The stream choice depends on the parity of symbolic block starts: emit one copy per
                # parity vector, each guarded by compile-time loops that keep exactly the matching copy.
                for parities in itertools.product((0, 1), repeat=len(axes)):
                    variant = copy.deepcopy(b)
                    self._active_compute_block = variant
                    self._parity_override = dict(zip(axes, parities))
                    variant = self.generic_visit(variant)
                    result.append(self._guard(variant, axes, parities))
                    self._parity_override = {}

            self._active_compute_block = None

        return result

    def _symbolic_parity_axes(self, block: spa.ComputeBlock) -> list[str]:
        """
        :param block: A split compute block.
        :return: The axes (``'x'``, ``'y'``) along which the block communicates through relative streams
                 and whose block start is a symbolic coordinate, in the order x, y.
        """
        collector = _StreamUseCollector()
        collector.visit(block)
        used_axes = set()
        for name in collector.stream_names:
            streams = self.stream_map.get(name)
            if streams is None or not isinstance(streams[0].stream, spa.RelativeStreamDeclaration):
                continue
            used_axes.add('x' if streams[0].stream.dx.value.eval() != 0 else 'y')
        return [axis for axis in ('x', 'y')
                if axis in used_axes and is_symbolic(coord_of_expr(self._axis_range(block, axis).start))]

    @staticmethod
    def _axis_range(block: spa.ComputeBlock, axis: str) -> spa.RangeExpression:
        return block.subgrid.x_range if axis == 'x' else block.subgrid.y_range

    def _guard(self, block: spa.ComputeBlock, axes: list[str], parities: tuple[int, ...]) -> spa.MetaForBlock:
        """
        Wraps a block in nested compile-time loops ``for u16 __parity_a in [0 : p]``, one per axis, where
        ``p`` is ``(start_a % 2)`` for parity 1 and ``1 - (start_a % 2)`` for parity 0. After parameter
        concretization, each loop runs once if the block start has the given parity and zero times
        otherwise.
        """
        inner = block
        for axis, parity in reversed(list(zip(axes, parities))):
            start = self._axis_range(block, axis).start
            two = spa.Expression(spa.ConstantLiteral(2, self.int_type))
            start_parity = spa.Expression(spa.BinaryOperator(copy.deepcopy(start), '%', two))
            if parity == 1:
                stop = start_parity
            else:
                one = spa.Expression(spa.ConstantLiteral(1, self.int_type))
                stop = spa.Expression(spa.BinaryOperator(one, '-', start_parity))
            zero = spa.Expression(spa.ConstantLiteral(0, self.int_type))
            inner = spa.MetaForBlock(
                variables=[spa.TypedIdentifier(spa.ScalarType.u16, spa.Identifier(f'__parity_{axis}', 0))],
                range_expression=[spa.RangeExpression(zero, stop)],
                body=[inner])
        return inner

    def _block_parity(self, axis: str) -> int:
        if axis in self._parity_override:
            return self._parity_override[axis]
        return coord_of_expr(self._axis_range(self._active_compute_block, axis).start) % 2

    def visit_SendStatement(self, stmt: spa.SendStatement):
        stmt.stream_name = self._resolve_communication_stream(stmt.stream_name, is_receive=False)
        return stmt

    def visit_ReceiveStatement(self, stmt: spa.ReceiveStatement):
        stmt.stream_name = self._resolve_communication_stream(stmt.stream_name, is_receive=True)
        return stmt

    def visit_ReceiveGenerator(self, gen: spa.ReceiveGenerator):
        gen.stream_name = self._resolve_communication_stream(gen.stream_name, is_receive=True)
        return gen

    def _resolve_communication_stream(self, stream_name: spa.Identifier, is_receive: bool):
        if not isinstance(stream_name, spa.Identifier):
            # Arguments - do not change
            return stream_name
        
        streams = self.stream_map[stream_name]
        if not isinstance(streams[0].stream, spa.RelativeStreamDeclaration):
            return stream_name  # Non-relative streams are not changed

        dx = streams[0].stream.dx.value.eval()
        dy = streams[0].stream.dy.value.eval()

        assert abs(dx) + abs(dy) <= 1

        if abs(dx) > 0:
            sign = dx > 0
            block_parity: int = self._block_parity('x')
        else:
            assert abs(dy) > 0
            sign = dy > 0
            block_parity: int = self._block_parity('y')

        assert isinstance(block_parity, int)

        # Logically flip the direction of stream when receiving
        if is_receive:
            sign = not sign

        channel_idx = block_parity if sign else (block_parity + 1) % 2

        return streams[channel_idx].stream_name
