"""
Canonicalization passes for Spatial IR
"""
from collections import defaultdict
import copy
from dataclasses import dataclass
from spatialstencil.syntax.spatial_ir import irnodes as spir, analysis, passes
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle


def canonicalize_phases(kernel: spir.Kernel) -> spir.Kernel:
    """
    If the given kernel contains free dataflow/compute blocks that are not contained in phases,
    contains them in phase blocks.

    Ensures that ``kernel.body`` will only contain Phase blocks.
    """
    new_body: list[spir.Phase] = []
    current_phase = None
    for block in kernel.body:
        if isinstance(block, spir.Phase):  # Already a phase
            if current_phase is not None:  # Commit previous phase
                if len(current_phase.compute) + len(current_phase.dataflow) + len(current_phase.place) > 0:
                    new_body.append(current_phase)
                current_phase = None
            new_body.append(block)
            continue

        # Create new phase as necessary
        if current_phase is None:
            current_phase = spir.Phase([], [], [])

        if isinstance(block, spir.PlaceBlock):  # Keep placement global
            new_body.append(block)
            continue
        elif isinstance(block, spir.DataflowBlock):
            current_phase.dataflow.append(block)
        elif isinstance(block, spir.ComputeBlock):
            current_phase.compute.append(block)
        else:
            raise TypeError(f'Unrecognized kernel body IR node type "{type(block).__name__}"')

    # Final phase
    if current_phase is not None:
        if len(current_phase.compute) + len(current_phase.dataflow) + len(current_phase.place) > 0:
            new_body.append(current_phase)

    # Reassign kernel body
    return spir.Kernel(kernel.name, kernel.parameters, kernel.arguments, new_body)


def inline_phases(kernel: spir.Kernel) -> spir.Kernel:
    """
    Inlines phases into their constituent computation and dataflow blocks by adding waits and appending all streams,
    respectively.
    """
    rect_place: dict[tuple[int, int, int, int], spir.PlaceBlock] = {}
    rect_dataflow: dict[tuple[int, int, int, int], spir.DataflowBlock] = {}
    rect_compute: dict[tuple[int, int, int, int], spir.ComputeBlock] = {}
    # After canonicalize phases, kernel body can only contain phases or place blocks
    for block in kernel.body:
        rect = block.get_grid_rect()
        if isinstance(block, spir.PlaceBlock):
            if rect in rect_place:
                rect_place[rect].statements.extend(block.statements)
            else:
                rect_place[rect] = copy.deepcopy(block)
        elif isinstance(block, spir.Phase):
            # Extend place blocks
            for place in block.place:
                rect = place.get_grid_rect()
                if rect in rect_place:
                    # Replace variables in place statements with the new variables
                    rep = passes.FindAndReplace({
                        oldv.identifier: newv.identifier
                        for oldv, newv in zip(place.variables, rect_place[rect].variables)
                    })
                    stmts = [rep.visit(s) for s in place.statements]
                    rect_place[rect].statements.extend(stmts)
                else:
                    rect_place[rect] = copy.deepcopy(place)
            # Extend dataflow blocks
            for df in block.dataflow:
                rect = df.get_grid_rect()
                if rect in rect_dataflow:
                    rep = passes.FindAndReplace({
                        oldv.identifier: newv.identifier
                        for oldv, newv in zip(df.variables, rect_dataflow[rect].variables)
                    })
                    stmts = [rep.visit(s) for s in df.statements]
                    rect_dataflow[rect].statements.extend(stmts)
                else:
                    rect_dataflow[rect] = copy.deepcopy(df)
            # Concatenate compute blocks with an endphase statement
            for compute in block.compute:
                rect = compute.get_grid_rect()
                if rect in rect_compute:
                    rect_compute[rect].statements.append(spir.AwaitAllStatement())
                    rep = passes.FindAndReplace({
                        oldv.identifier: newv.identifier
                        for oldv, newv in zip(compute.variables, rect_compute[rect].variables)
                    })
                    stmts = [rep.visit(s) for s in compute.statements]
                    rect_compute[rect].statements.extend(stmts)
                else:
                    rect_compute[rect] = copy.deepcopy(compute)
        else:
            raise TypeError(f'Unexpected block type "{type(block).__name__}" in kernel. Was ``canonicalize_phases`` '
                            'called?')

    return spir.Kernel(
        name=kernel.name,
        parameters=copy.deepcopy(kernel.parameters),
        arguments=copy.deepcopy(kernel.arguments),
        body=list(rect_place.values()) + list(rect_dataflow.values()) + list(rect_compute.values()))


@dataclass
class PEBlock:
    """
    A class that represents a Processing Element equivalence class, with a canonical
    one-block place, dataflow, and compute blocks.
    """
    place: spir.PlaceBlock
    dataflow: spir.DataflowBlock
    compute: spir.ComputeBlock


def consolidate_rectangles_to_equivalence_classes(kernel: spir.Kernel) -> list[Rectangle[PEBlock]]:
    """
    Ensures dataflow/compute/place exist for each equivalence class.
    """
    # After inline_phases, there should be one block of each type for each rectangle
    result: dict[tuple[int, int, int, int], PEBlock] = defaultdict(lambda: PEBlock(None, None, None))
    rect_to_stride: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for block in kernel.body:
        rect = block.get_grid_rect()
        rect_to_stride[rect] = block.get_grid_stride()
        if isinstance(block, spir.PlaceBlock):
            assert result[rect].place is None
            result[rect].place = block
        elif isinstance(block, spir.DataflowBlock):
            assert result[rect].dataflow is None
            result[rect].dataflow = block
        elif isinstance(block, spir.ComputeBlock):
            assert result[rect].compute is None
            result[rect].compute = block

    # Fill in remainder of PEBlock with empty scopes (e.g., blocks without dataflow)
    for rect, pe in result.items():
        subgrid = spir.SubgridExpression.from_tuple(
            (rect[0], rect[1], rect_to_stride[rect][0]),
            (rect[2], rect[3], rect_to_stride[rect][1]),
        )
        if pe.place is None:
            pe.place = spir.PlaceBlock(_make_vars(), copy.deepcopy(subgrid), [])
        if pe.dataflow is None:
            pe.dataflow = spir.DataflowBlock(_make_vars(), copy.deepcopy(subgrid), [])
        if pe.compute is None:
            pe.compute = spir.ComputeBlock(_make_vars(), copy.deepcopy(subgrid), [])

    return [
        Rectangle((k[0], k[1], rect_to_stride[k][0]), (k[2], k[3], rect_to_stride[k][1]), v)
        for k, v in sorted(result.items())
    ]


def _make_vars():
    """
    Helper function that creates two unused variables for an empty block.
    """
    return [
        spir.TypedIdentifier(spir.ScalarType.u16, spir.Identifier('__i', 0)),
        spir.TypedIdentifier(spir.ScalarType.u16, spir.Identifier('__j', 0))
    ]


def reduce_streams(kernel: spir.Kernel) -> spir.Kernel:
    """
    Combines multiple streams if their colors and routing instructions overlap.
    """
    # TODO(later)
    return kernel


class _BulkCommunicationLowerer(spir.NodeTransformer):

    def __init__(self, place: spir.PlaceBlock):
        super().__init__()
        self.identifier_sizes = analysis.get_identifier_sizes(place)
        self.identifier_dtypes = analysis.get_identifier_types(place)

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        sz = node.get_size(self.identifier_sizes)
        if len(sz) == 0:  # Scalar receive
            return self.generic_visit(node)

        # Array receive, make a foreach node
        new_node = spir.ForeachStatement(
            [spir.TypedIdentifier(spir.ScalarType.u16, spir.Identifier(f'__k{i}', 0)) for i in range(len(sz))],
            [
                # ``0:size`` for every dimension
                spir.RangeExpression(
                    spir.Expression(spir.ConstantLiteral(0, spir.ScalarType.u16)),
                    spir.Expression(spir.ConstantLiteral(s, spir.ScalarType.u16))) for s in sz
            ],
            spir.TypedIdentifier(self.identifier_dtypes[node.local_array], spir.Identifier(f'__x', 0)),
            spir.ReceiveGenerator(node.stream_name),
            [
                # ``arr[__k0, ...] = __x``
                spir.AssignmentStatement(
                    spir.ArraySlice(
                        copy.deepcopy(node.local_array),
                        [spir.Expression(spir.Identifier(f'__k{i}', 0)) for i in range(len(sz))]),
                    spir.Expression(spir.Identifier(f'__x', 0))),
            ],
            node.completion_name)
        new_node.lineinfo = node.lineinfo

        return new_node

    # NOTE: No need for this, the ``.extent`` DSD field in CSL takes care of that. Additionally, the completion moving
    #       into the for loop does not make sense.
    # def visit_SendStatement(self, node: spir.SendStatement):
    #     sz = node.get_size(self.identifier_sizes)
    #     if len(sz) == 0:  # Scalar send
    #         return self.generic_visit(node)

    #     # Array send, make a for node
    #     new_node = spir.ForStatement(
    #         [spir.TypedIdentifier(spir.ScalarType.u16, spir.Identifier(f'__k{i}', 0)) for i in range(len(sz))],
    #         [
    #             # ``0:size`` for every dimension
    #             spir.RangeExpression(
    #                 spir.Expression(spir.ConstantLiteral(0, spir.ScalarType.u16)),
    #                 spir.Expression(spir.ConstantLiteral(s, spir.ScalarType.u16))) for s in sz
    #         ],
    #         [
    #             # ``send(arr[__k0, ...], stream)``
    #             spir.SendStatement(
    #                 spir.ArraySlice(
    #                     copy.deepcopy(node.local_array),
    #                     [spir.Expression(spir.Identifier(f'__k{i}', 0)) for i in range(len(sz))]), node.stream_name,
    #                 node.completion_name)
    #         ])

    #     return new_node


def lower_bulk_communication(rectangles: list[Rectangle[PEBlock]]) -> None:
    """
    Lowers top-level array ``receive`` and ``send`` operations to foreach and for loops, respectively.
    The array operations are shorthands for a row-major (C-order) loop over the communication operations.

    :param rectangles: A list of PE block rectangles to lower computations within.
    """
    for rect in rectangles:
        rect.metadata.compute = _BulkCommunicationLowerer(rect.metadata.place).visit(rect.metadata.compute)


class _MakeArraySlices(spir.NodeTransformer):

    def __init__(self, index: list[spir.Identifier], identifier_sizes: dict[spir.Identifier, list[int]]):
        super().__init__()
        self.index = index
        self.identifier_sizes = identifier_sizes

    def visit_Identifier(self, node: spir.Identifier):
        if self.identifier_sizes[node]:
            new_node = spir.ArraySlice(node, [spir.Expression(v) for v in self.index])
            new_node.lineinfo = node.lineinfo
            return new_node
        return self.generic_visit(node)


class _ArrayAssignmentLowerer(spir.NodeTransformer):

    def __init__(self, place: spir.PlaceBlock):
        super().__init__()
        self.identifier_sizes = analysis.get_identifier_sizes(place)
        self.identifier_dtypes = analysis.get_identifier_types(place)

    def visit_AssignmentStatement(self, node: spir.AssignmentStatement):
        if not isinstance(node.destination, spir.Identifier):
            return self.generic_visit(node)

        if not self.identifier_sizes[node.destination]:  # Skip scalar assignments
            return self.generic_visit(node)

        sz = self.identifier_sizes[node.destination]

        typed_variables = [
            spir.TypedIdentifier(spir.ScalarType.u16, spir.Identifier(f'__k{i}', 0)) for i in range(len(sz))
        ]
        variables = [spir.Identifier(f'__k{i}', 0) for i in range(len(sz))]
        for i in range(len(sz)):
            variables[i].lineinfo = node.lineinfo
            typed_variables[i].lineinfo = node.lineinfo
        slicemaker = _MakeArraySlices(variables, self.identifier_sizes)
        new_assignment = slicemaker.visit(node)

        # Array assignment, make a map node
        new_node = spir.MapStatement(
            typed_variables,
            [
                # ``0:size`` for every dimension
                spir.RangeExpression(
                    spir.Expression(spir.ConstantLiteral(0, spir.ScalarType.u16)),
                    spir.Expression(spir.ConstantLiteral(s, spir.ScalarType.u16))) for s in sz
            ],
            body=[
                # ``arr[__k0, ...] = a[__k0, ...] + b[__k0, ...]``
                new_assignment
            ])
        new_node.lineinfo = node.lineinfo

        return new_node


def lower_array_assignment(rectangles: list[Rectangle[PEBlock]]) -> None:
    """
    Lowers array assignments into ``map`` operations.

    :param rectangles: A list of PE block rectangles to lower computations within.
    """
    for rect in rectangles:
        rect.metadata.compute = _ArrayAssignmentLowerer(rect.metadata.place).visit(rect.metadata.compute)


class _MemCpyStreamOperatorRemover(spir.NodeTransformer):

    def __init__(self, stream_args: set[spir.Identifier]):
        super().__init__()
        self.stream_args = stream_args

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        if isinstance(node.stream_name, spir.Identifier) and node.stream_name in self.stream_args:
            return None
        if isinstance(node.stream_name, spir.ArraySlice) and node.stream_name.array in self.stream_args:
            return None
        return self.generic_visit(node)

    def visit_SendStatement(self, node: spir.SendStatement):
        if isinstance(node.stream_name, spir.Identifier) and node.stream_name in self.stream_args:
            return None
        if isinstance(node.stream_name, spir.ArraySlice) and node.stream_name.array in self.stream_args:
            return None
        return self.generic_visit(node)

    def visit_ForeachStatement(self, node: spir.ForeachStatement):
        if isinstance(node.receive_stream.stream_name,
                      spir.Identifier) and node.receive_stream.stream_name in self.stream_args:
            return None
        if isinstance(node.receive_stream.stream_name,
                      spir.ArraySlice) and node.receive_stream.stream_name.array in self.stream_args:
            return None
        return self.generic_visit(node)


def remove_memcpy_stream_operators(kernel: spir.Kernel, rectangles: list[Rectangle[PEBlock]]) -> None:
    """
    Removes receives/sends/foreach loops that involve kernel arguments from the given rectangles in memcpy mode.
    This pass is performed because memcpy mode will already copy the memory in and out outside the kernel code.

    :param kernel: The kernel to modify.
    :param rectangles: A list of PE block rectangles to modify.
    """
    stream_args: set[spir.Identifier] = set()
    for arg in kernel.arguments:
        if isinstance(arg.dtype, spir.StreamType):
            stream_args.add(arg.identifier)
        elif isinstance(arg.dtype, spir.ArrayType) and isinstance(arg.dtype.base_type, spir.StreamType):
            stream_args.add(arg.identifier)

    # TODO(later): Verify that each stream argument is used once, and then replace every occurrence of the stream
    #              argument with its internal name.
    for rect in rectangles:
        rect.metadata.compute = _MemCpyStreamOperatorRemover(stream_args).visit(rect.metadata.compute)
