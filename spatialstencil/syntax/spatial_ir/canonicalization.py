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
            if result[rect].place is not None:
                result[rect].place.statements.extend(block.statements)
                block.statements.clear()
            else:
                result[rect].place = block
        elif isinstance(block, spir.DataflowBlock):
            if result[rect].dataflow is not None:
                result[rect].dataflow.statements.extend(block.statements)
                block.statements.clear()
            else:
                result[rect].dataflow = block
        elif isinstance(block, spir.ComputeBlock):
            if result[rect].compute is not None:
                raise ValueError('Multiple compute blocks found for the same rectangle after inlining phases.')
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


class _ForeachDataTaskToLoopConverter(spir.NodeTransformer):

    def __init__(self, dtypes: dict[spir.Identifier, spir.IRType]):
        super().__init__()
        self.dtypes = dtypes

    def visit_ForeachStatement(self, node: spir.ForeachStatement):
        from spatialstencil.syntax.csl import dsd_ops
        if dsd_ops.get_dsd_op(self.dtypes, node) is not None:
            return self.generic_visit(node)

        if isinstance(self.dtypes[node.receive_stream.stream_name], spir.StreamType):
            return self.generic_visit(node)

        body_statements = [self.visit(stmt) for stmt in node.body]
        loop_variables = [copy.deepcopy(var) for var in node.variables]
        loop_ranges = [copy.deepcopy(rng) for rng in node.parameter_range]
        stream_target = copy.deepcopy(node.receive_stream.stream_name)
        if isinstance(self.dtypes[stream_target], spir.ArrayType) and loop_ranges:
            index_exprs = []
            for var in loop_variables:
                idx_identifier = copy.deepcopy(var.identifier)
                index_expr = spir.Expression(idx_identifier)
                index_expr.lineinfo = getattr(idx_identifier, 'lineinfo', node.lineinfo)
                index_exprs.append(index_expr)

            if isinstance(stream_target, spir.ArraySlice):
                stream_target.indices.extend(index_exprs)
            elif isinstance(stream_target, spir.Identifier):
                stream_target = spir.ArraySlice(stream_target, index_exprs)
            else:
                raise TypeError(
                    f'Unsupported stream target type "{type(stream_target).__name__}" in foreach to loop conversion')

        stream_target.lineinfo = getattr(stream_target, 'lineinfo', node.lineinfo)

        receive_destination = copy.deepcopy(node.stream_variable)
        receive_destination.identifier.lineinfo = getattr(receive_destination, 'lineinfo', node.lineinfo)
        receive_statement = spir.ReceiveStatement(receive_destination.identifier, stream_target)
        receive_statement.local_array = receive_destination
        receive_statement.lineinfo = node.lineinfo

        loop_body = [receive_statement] + body_statements

        loop_statement = spir.ForStatement(loop_variables, loop_ranges, loop_body)
        loop_statement.lineinfo = node.lineinfo

        # If the foreach was running asynchronously, wrap the loop in an async block
        if node.completion_name is not None:
            async_block = spir.AsyncBlock(copy.deepcopy(node.completion_name), [loop_statement])
            async_block.lineinfo = node.lineinfo
            return async_block

        return loop_statement


def convert_foreach_data_tasks_to_loops(rect: Rectangle[PEBlock], dtypes: dict[spir.Identifier, spir.IRType]) -> None:
    """
    Converts foreach blocks on input arguments to (async) loop blocks in memcpy mode.
    This pass is performed because memcpy mode will already copy the memory in and out outside the kernel code.

    :param rect: A single PE block rectangle to modify.
    :param dtypes: A mapping of identifier to its type in the given rectangle.
    """
    rect.metadata.compute = _ForeachDataTaskToLoopConverter(dtypes).visit(rect.metadata.compute)


def lower_arguments_to_extern(rectangles: list[Rectangle[PEBlock]], kernel: spir.Kernel) -> None:
    """
    Lowers stream arguments to extern field declarations in a place block or 
    extern stream declarations in a dataflow block. Scalar arguments are unaffected.

    :param rectangles: A list of PE block rectangles to modify.
    :param kernel: The kernel whose arguments are being lowered.
    :note: Modifies the kernel in-place.
    """
    # Create dataflow or place block depending on argument types
    stream_decls: list[spir.StreamDeclaration] = []
    field_decls: list[spir.FieldDeclaration] = []

    # Create declarations
    for arg in kernel.arguments:
        dtype = arg.dtype
        if isinstance(dtype, spir.ArrayType):
            dtype = dtype.base_type

        if isinstance(dtype, spir.StreamType):
            if dtype.buffer_size is not None:
                field_decls.append(
                    spir.FieldDeclaration(
                        dtype=spir.ArrayType(dtype.element_type, [dtype.buffer_size]),
                        field_name=arg.identifier,
                        is_extern=True))
            else:
                if arg.readonly or (not arg.readonly and not arg.writeonly):
                    extern_decl = spir.StreamDeclaration(
                        stream_name=arg.identifier,
                        dtype=dtype,
                        stream=spir.ExternStreamDeclaration('in', routing=spir.RoutingDeclaration()))
                    stream_decls.append(extern_decl)
                if arg.writeonly or (not arg.readonly and not arg.writeonly):
                    extern_decl = spir.StreamDeclaration(
                        stream_name=arg.identifier,
                        dtype=dtype,
                        stream=spir.ExternStreamDeclaration('out', routing=spir.RoutingDeclaration()))
                    stream_decls.append(extern_decl)

    # Replace all index expressions of our newly created extern streams/fields
    extern_names = set(decl.stream_name for decl in stream_decls) | set(decl.field_name for decl in field_decls)

    class _ArgumentReplacer(spir.NodeTransformer):

        def visit_ArraySlice(self, node: spir.ArraySlice):
            if node.array in extern_names:
                return node.array
            return self.generic_visit(node)

    replacer = _ArgumentReplacer()

    # Insert dataflow and place blocks for every compute block
    # (unused fields/streams will be pruned later)
    for rect in rectangles:
        for decl in stream_decls:
            if decl not in rect.metadata.dataflow.statements:
                rect.metadata.dataflow.statements.append(copy.deepcopy(decl))
        for decl in field_decls:
            if decl not in rect.metadata.place.statements:
                rect.metadata.place.statements.append(copy.deepcopy(decl))

        for stmt in rect.metadata.compute.statements:
            stmt = replacer.visit(stmt)

    # Remove arguments from kernel
    kernel.arguments = [arg for arg in kernel.arguments if arg.identifier not in extern_names]
