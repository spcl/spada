"""
Contains analysis functions for Spatial IR, such as statement dependency analysis.
"""
from spatialstencil.syntax.spatial_ir import irnodes as spir
from dataclasses import dataclass
from typing import Literal
import networkx as nx  # TODO: Switch to igraph
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle


@dataclass(frozen=True)
class CompletionDAGNode:
    """
    Object representing a completion DAG node.
    """
    optype: Literal['post', 'wait']
    statement_id: int


def to_completion_dag(compute: spir.ComputeBlock) -> nx.DiGraph:
    """
    Converts a compute block to a directed graph of statement dependencies,
    as defined in the specifications (local order) and based on completions
    and code order.

    The resulting graph contains ``CompletionDAGNode`` objects, which refer to
    whether the node is posting an asynchronous task or waiting for one
    (based on the node's ``optype`` field), and which statement index in
    the compute block's statements it refers to.
    """
    result = nx.DiGraph()

    # Keep track of last node for sequential dependencies in local order
    last_node: CompletionDAGNode | None = None
    # Keep track of unawaited completions
    incomplete_completions: dict[str, CompletionDAGNode] = {}

    for stmt_id, stmt in enumerate(compute.statements):
        node: CompletionDAGNode
        completion_node: CompletionDAGNode | None = None

        # TODO: If seeing foreach or for and communication is inside, raise NotImplementedError

        # awaitall
        if isinstance(stmt, spir.AwaitAllStatement):
            # If there is nothing to wait for, skip node
            if not incomplete_completions:
                continue

            # Connect all previous incomplete nodes to this node
            node = CompletionDAGNode('wait', stmt_id)
            result.add_node(node)
            for compnode in incomplete_completions.values():
                result.add_edge(compnode, node)
            incomplete_completions.clear()

        # await completion
        elif isinstance(stmt, spir.AwaitCompletionStatement):
            # Create a completion node and connect it to the poster
            compname = stmt.completion_name.as_ir()
            if compname not in incomplete_completions:
                raise SyntaxError(f'Trying to await completion "{stmt.completion_name.as_ir()}", which does not exist '
                                  'or was already awaited for.')
            node = CompletionDAGNode('wait', stmt_id)
            result.add_node(node)
            result.add_edge(incomplete_completions[compname], node)
            del incomplete_completions[compname]

        # Asynchronous nodes (completion comp = ...)
        else:
            # Create poster node
            node = CompletionDAGNode('post', stmt_id)
            result.add_node(node)
            completion: spir.Completion | None = stmt.completion_name
            if completion is None:
                # Create another completion node immediately after this one and connect it
                completion_node = CompletionDAGNode('wait', stmt_id)
                result.add_node(completion_node)
                result.add_edge(node, completion_node)
            else:
                # Add to unawaited completions
                incomplete_completions[completion.name.as_ir()] = node

        # Potentially connect previous node if not already connected (local code order)
        if last_node is not None and last_node not in result.predecessors(node):
            result.add_edge(last_node, node)

        # Set new last node
        if completion_node is not None:
            last_node = completion_node
        else:
            last_node = node

    return result


def get_identifier_sizes(place: spir.PlaceBlock) -> dict[spir.Identifier, list[int]]:
    """
    Returns a dictionary mapping each identifier to its dimensions, or an empty list if scalar.
    """
    result = {}
    for decl in place.statements:
        if isinstance(decl.dtype, spir.ScalarType):
            result[decl.field_name] = []
        else:  # Array type
            evaluated_shape = []
            for s in decl.dtype.shape:
                if isinstance(s, int):
                    evaluated_shape.append(s)
                else:
                    evaluated_shape.append(s.eval())
            result[decl.field_name] = evaluated_shape
    return result


def get_identifier_types(place: spir.PlaceBlock) -> dict[spir.Identifier, spir.ScalarType]:
    """
    Returns a dictionary mapping each identifier to its data type.
    """
    result = {}
    for decl in place.statements:
        if isinstance(decl.dtype, spir.ScalarType):
            result[decl.field_name] = decl.dtype
        else:  # Array type
            result[decl.field_name] = decl.dtype.base_type.element_type
    return result


class _SendRecvCollector(spir.NodeVisitor):

    def __init__(self):
        super().__init__()
        self.sends: set[spir.Identifier] = set()
        self.receives: set[spir.Identifier] = set()

    def _get_underlying_stream(self, node: spir.Identifier | spir.ArraySlice) -> spir.Identifier:
        if isinstance(node, spir.ArraySlice):
            return node.array
        return node

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        self.receives.add(self._get_underlying_stream(node.stream_name))

    def visit_ReceiveGenerator(self, node: spir.ReceiveGenerator):
        self.receives.add(self._get_underlying_stream(node.stream_name))

    def visit_SendStatement(self, node: spir.SendStatement):
        self.sends.add(self._get_underlying_stream(node.stream_name))


def sends_and_receives(compute: spir.ComputeBlock) -> dict[spir.Identifier, tuple[bool, bool]]:
    """
    Returns, for each stream, whether it is used in a send or receive operation.

    :return: Dictionary mapping stream identifiers to a 2-tuple of (is_sent, is_received).
    """
    collector = _SendRecvCollector()
    collector.visit(compute)
    all_identifiers = {k for k in collector.sends | collector.receives}
    return {k: (k in collector.sends, k in collector.receives) for k in all_identifiers}


def get_kernel_stream_arguments(
        kernel: spir.Kernel) -> tuple[dict[str, dict[str, list[int] | str]], dict[str, dict[str, list[int] | str]]]:
    """
    Returns two dictionaries:
    1. A dictionary mapping input stream names to their data types and shapes.
    2. A dictionary mapping output stream names to their data types and shapes.
    """
    input_streams = {}
    output_streams = {}
    for arg in kernel.arguments:
        if arg.compiletime:
            continue

        shape = []
        if isinstance(arg.dtype, spir.ArrayType):
            for dim in arg.dtype.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    shape.append(dim.eval())

        arg_as_dict = {
            "dtype": arg.dtype.element_type.element_type.element_type.element_type.as_ir(),
            "shape": shape,
        }
        if isinstance(arg.dtype, spir.StreamType):
            arg_as_dict["buffer_size"] = arg.dtype.buffer_size.eval() if arg.dtype.buffer_size else None
        elif isinstance(arg.dtype, spir.ArrayType) and isinstance(arg.dtype.base_type, spir.StreamType):
            arg_as_dict["buffer_size"] = arg.dtype.base_type.buffer_size.eval(
            ) if arg.dtype.base_type.buffer_size else None
        else:
            arg_as_dict["buffer_size"] = None

        if arg.readonly:
            input_streams[arg.identifier.name] = arg_as_dict
        elif arg.writeonly:
            output_streams[arg.identifier.name] = arg_as_dict
        else:
            input_streams[arg.identifier.name] = arg_as_dict
            output_streams[arg.identifier.name] = arg_as_dict

    return input_streams, output_streams


def kernel_uses_memcpy_mode(kernel: spir.Kernel) -> bool:
    """
    Returns whether the kernel uses memcpy mode for any of its stream arguments.
    """
    for arg in kernel.arguments:
        if isinstance(arg.dtype, spir.StreamType) and arg.dtype.buffer_size is not None:
            return True
        if isinstance(arg.dtype, spir.ArrayType) and isinstance(arg.dtype.base_type, spir.StreamType):
            if arg.dtype.base_type.buffer_size is not None:
                return True
    return False


class StreamExtents:
    """
    Class to manage stream extents for mapping stream arguments to PE rectangles.
    """

    def __init__(self, kernel: spir.Kernel):
        self.extents: dict[spir.Identifier, list[Rectangle]] = {}
        self.argnames: set[spir.Identifier] = set(arg.identifier for arg in kernel.arguments)

    def add_extent(self, arg: spir.Identifier, rect: Rectangle):
        """
        Adds a rectangle extent for the given stream argument.
        If the argument is not already in the extents, it initializes it.
        """
        if arg not in self.argnames:
            return  # Ignore arguments not in the kernel
        if arg not in self.extents:
            self.extents[arg] = []
        self.extents[arg].append(rect)

    def is_valid(self, arg: spir.Identifier, rect: Rectangle) -> bool:
        return arg in self.extents and any(rect.is_subset_of(r) for r in self.extents[arg])


def detect_stream_argument_extents(rectangles: list[Rectangle], kernel: spir.Kernel) -> StreamExtents:
    """
    Detects the extents of stream arguments in the current kernel.
    This function collects all rectangles that contain send or receive statements
    for each stream argument and returns a ``StreamExtents`` object containing this information.
    
    Each stream argument has to correspond to a single contiguous rectangle (even if it appears as a union
    of rectangles). In case of disjoint rectangles, an exception is raised.
    """
    # Collect all extents from all rectangles
    stream_extents = StreamExtents(kernel)
    for rect in rectangles:
        compute_block: spir.ComputeBlock = rect.metadata.compute
        # Create a mapping of variable names to their positions in the compute block
        var_name_to_position = {var.identifier: i for i, var in enumerate(compute_block.variables)}
        subgrid = compute_block.get_grid_rect()

        for top_stmt in compute_block.statements:
            for stmt in top_stmt.walk():
                if isinstance(stmt, (spir.SendStatement, spir.ReceiveStatement, spir.ReceiveGenerator)):
                    stream_name = stmt.stream_name

                    # Keep track of the position order of indices in the array slice (or none if one stream is used)
                    position_order = []

                    # If array slice, ensure that the indices are valid for the rectangle
                    if isinstance(stream_name, spir.ArraySlice):
                        if stream_name.array not in stream_extents.argnames:
                            continue
                        # For 1D rectangle subsets (e.g., ``place i,j in [0:1, 0:N]`` with ``a[j]``),
                        # we need to check that the used indices correspond to valid compute block variables

                        # Check that each index in the array slice corresponds to a valid compute block variable
                        for index_expr in stream_name.indices:
                            index_name = index_expr.value
                            if not isinstance(index_name, spir.Identifier) or index_name not in var_name_to_position:
                                raise ValueError(
                                    f"Array slice {stream_name.as_ir()} uses index '{index_name.as_ir()}', "
                                    f"but compute block variables are {[var.identifier.as_ir() for var in compute_block.variables]}. "
                                    f"Index is not available in this compute block.\n  In {stream_name.lineinfo}")
                            position_order.append(var_name_to_position[index_name])
                        # If position order is not monotonically increasing, raise an error
                        if not all(position_order[i] <= position_order[i + 1] for i in range(len(position_order) - 1)):
                            raise ValueError(
                                f"Array slice {stream_name.as_ir()} uses an index order that does not match "
                                f"the compute block variables {[var.identifier.as_ir() for var in compute_block.variables]}"
                                f".\n  In {stream_name.lineinfo}")

                        stream_name = stream_name.array

                    if stream_name not in stream_extents.argnames:
                        continue

                    # For every variable name that is not in the position order, ensure the dimension is 1
                    for i, var in enumerate(compute_block.variables):
                        if i not in position_order:
                            if (subgrid[2 * i + 1] - subgrid[2 * i]) != 1:
                                raise ValueError(
                                    f"Array slice {stream_name.as_ir()} skips index '{var.identifier.as_ir()}', "
                                    f"but compute block variable '{var.identifier.as_ir()}' has shape "
                                    f"{(subgrid[2 * i + 1] - subgrid[2 * i])}. Unused index subgrids must have "
                                    f"dimension 1.\n  In {stream_name.lineinfo}")

                    stream_extents.add_extent(stream_name, rect)

    # Check for disjoint rectangles and validate that each stream argument maps to a contiguous region
    for stream_name, extents in stream_extents.extents.items():
        if len(extents) > 1:
            # Check if rectangles can be unified into a single contiguous rectangle
            # For now, we'll raise an error for disjoint rectangles as they're not supported

            # Sort rectangles by their position to check for contiguity
            extents.sort(key=lambda r: (r.x_range[0], r.y_range[0]))

            # Check if rectangles are contiguous (can be unified)
            for i in range(len(extents) - 1):
                current_rect = extents[i]
                for next_rect in extents[0:i] + extents[i + 1:]:
                    # Check if rectangles are adjacent or overlapping
                    # Two rectangles are contiguous if they share a border or overlap
                    x_adjacent = (
                        current_rect.x_range[1] == next_rect.x_range[0] or
                        current_rect.x_range[0] == next_rect.x_range[1] or
                        (current_rect.x_range[0] <= next_rect.x_range[1] and
                         next_rect.x_range[0] <= current_rect.x_range[1]))

                    y_adjacent = (
                        current_rect.y_range[1] == next_rect.y_range[0] or
                        current_rect.y_range[0] == next_rect.y_range[1] or
                        (current_rect.y_range[0] <= next_rect.y_range[1] and
                         next_rect.y_range[0] <= current_rect.y_range[1]))

                    # For rectangles to be contiguous, they must be adjacent in at least one dimension
                    # and overlap or be adjacent in the other dimension
                    if x_adjacent and y_adjacent:
                        break
                else:
                    # If we reach here, it means no adjacent rectangle was found
                    raise ValueError(f"Stream argument '{stream_name.as_ir()}' is used in disjoint rectangles. "
                                     f"Found rectangles at {current_rect.x_range}×{current_rect.y_range} and "
                                     f"{next_rect.x_range}×{next_rect.y_range}, which are not contiguous. "
                                     f"Stream arguments must correspond to a single contiguous rectangular region.")

    # Union all rectangles for each stream argument
    for stream_name, extents in stream_extents.extents.items():
        if len(extents) > 1:
            # Union the rectangles into a single rectangle
            x_min = min(r.x_range[0] for r in extents)
            x_max = max(r.x_range[1] for r in extents)
            x_step = min(r.x_range[2] for r in extents)
            y_min = min(r.y_range[0] for r in extents)
            y_max = max(r.y_range[1] for r in extents)
            y_step = min(r.y_range[2] for r in extents)

            # Create a new unified rectangle using the metadata from the first rectangle
            unified_rect = Rectangle(x_range=(x_min, x_max, x_step), y_range=(y_min, y_max, y_step), metadata=extents[0].metadata)

            # Replace the list with just the unified rectangle
            stream_extents.extents[stream_name] = [unified_rect]

    return stream_extents
