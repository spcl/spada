import copy
from collections import defaultdict
from spada.syntax.stencil_ir.irnodes import *


class RefactorForwardBackwardStencils(ScopedNodeVisitor):
    """
    For any horizontal offset in a forward/backward stencil,
    create a horizontal computation that reads all the data at the stencil offset and stores it,
    replacing the access to that field.

    ASSUMES THAT THE FIELDS BEING ACCESSED AT NON-ZERO HORIZONTAL OFFSETS
    WITHIN FORWARD/BACKWARD STENCILS ARE READONLY INPUT FIELDS
    """
    refactored: dict[Identifier, list[tuple[int, int]]]
    field_sources: list[Identifier]
    field_names: list[Identifier]
    field_offsets: list[tuple[int, int]]
    precomputation: ComputationBlock
    def visit_Subscript(self, node: Subscript):
        scope = self.get_scope()
        if isinstance(scope, ComputationBlock) and scope.schedule != ComputationType.PARALLEL:
            if node.subscript[0] != 0 or node.subscript[1] != 0:
                self._refactor_node(node)

    def _refactor_node(self, node: Subscript):
        assert node.value.version == 0, "In Forward/Backward stencils, horizontally-offset access must be to readonly fields"
        new_variable_name = f"_refactored_{node.value.name}_{node.subscript[0]}_{node.subscript[1]}"
        new_variable = Identifier(new_variable_name)
        if node.value not in self.refactored:
            if (node.subscript[0], node.subscript[1]) not in self.refactored[node.value]:
                # Note that we have to create the statement to read at the offset

                self.field_sources.append(node.value)
                self.field_offsets.append((node.subscript[0], node.subscript[1]))
                self.field_names.append(new_variable)

                # Save that we refactored this version
                self.refactored[node.value].append((node.subscript[0], node.subscript[1]))

        # Update the usage of the variable
        node.value = new_variable
        node.subscript[0] = 0
        node.subscript[1] = 0

    def pre_visit_Program(self, program: Program):
        self.refactored = defaultdict(list)
        self.field_offsets = []
        self.field_names = []
        self.field_sources = []

    def post_visit_Program(self, program: Program):
        # Generate the additional compute block and put it in the beginning of the program

        if len(self.field_names) == 0:
            return

        stmts = []
        for source, offset, new_variable in zip(self.field_sources, self.field_offsets, self.field_names):

            stmts.append(
                StatementBlock(
                    [new_variable],
                    [source],
                    [],
                    OperationType([ViewType.empty()], [ViewType.empty()]),
                    [
                        ReturnOp(
                            [Expression(Subscript(
                                source,
                                [offset[0], offset[1], 0]
                            ))],
                            OperationType(
                                [ViewType.empty()]
                            )
                        )
                    ]
                )
            )
        stmts.append(
            ReturnOp(
                [Expression(f) for f in self.field_names]
            )
        )

        comp = copy.deepcopy(ComputationBlock(
            self.field_names,
            self.field_sources,
            ComputationType.PARALLEL,
            [Interval(0, None), Interval(0, None), Interval(0, None)],
            OperationType([ViewType.empty()] * len(self.field_sources), [ViewType.empty()] * len(self.field_names)),
            stmts
        ))

        # Prepend the computation
        comps = [comp]
        comps.extend(program.computations)
        program.computations = comps
