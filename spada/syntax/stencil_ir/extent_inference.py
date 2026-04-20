import warnings
from collections import defaultdict
from typing import Sequence, Collection

import spada.syntax.stencil_ir.irnodes as sast
import spada.syntax.stencil_ir.def_use_analysis as def_use_analysis

def infer_field_extents(program: sast.Program):
    """
    Uses def-use analysis to infer the extents of a Stencil IR program by traversing it twice.

    First, we identify for each field where it is used and note that type object.
    This is done using a visitor that creates a dictionary of field names to a set of uses (field type objects).
    The field type objects collected are modified in place!

    Then, we compute the extents for each field based on the uses,
    by traversing the program backwards and accumulating the extents.
    Specifically, the extent of an input to a computation or program is the union of all its uses.
    The extent of a statement argument is the Minkowski sum of its 'local' extent and the extents of its uses.

    The 'local' extents are computed using an ExtentCollector.

    :param program: The program whose extents should be inferred (operates in-place).
    """

    # Create the def-use analysis object
    uses = dict()
    def_use = def_use_analysis.DefUseAnalysis(uses)
    def_use.visit(program)

    # Start with outputs. Extents always start at (0, 0, 0)
    _init_outputs(program.operation_type.destination)

    # Traverse the program backwards and accumulate the extents using a visitor
    extent_inference = ExtentInference(def_use, uses)
    extent_inference.visit(program)


class ExtentInference(sast.ScopedNodeVisitor):

    def __init__(self, def_use, uses):
        super().__init__(reverse=True)
        self.def_use = def_use
        self.uses = uses

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        _init_outputs(node.operation_type.destination)
        super().visit_ComputationBlock(node)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        computation = self.get_scope()
        # Get all the uses of the result within the current scope

        # We assume that all outputs are uniform, meaning that the uses are given by the union
        # of the uses of the outputs.
        use_offsets = []
        for output in node.outputs:
            offsets = self._offsets_of_uses_in_scope(computation, output)
            use_offsets.extend(offsets)

        use_offsets = list(set(use_offsets))

        # Compute local extents
        local_extent_collector = LocalExtentCollector()
        local_extent_collector.visit(node)

        # For each input, compute the Minkowski sum of the local extent and the uses
        for arg, arg_t in zip(node.inputs, node.operation_type.source):
            # Skip scalar types (they have no extent)
            if isinstance(arg_t, sast.ScalarType):
                continue

            # Compute the Minkowski sum of the local extent and the uses
            local_extent = local_extent_collector.extents[arg.name]

            if len(use_offsets) > 0:
                arg_t.extent.extents = [*_minkowski_sum(local_extent, use_offsets)]
            else:
                arg_t.extent.extents = [*local_extent]

            arg_t.extent.sort_extents()

        # For each output, the extent is the union of the uses
        for output, output_t in zip(node.outputs, node.operation_type.destination):
            output_t.extent.extents = use_offsets
            output_t.extent.sort_extents()

    def visit_IfBlock(self, node: sast.IfBlock):
        computation = self.get_scope_with_type(sast.ComputationBlock)
        assert len(node.outputs) == 1, "Only a single output is supported for now"
        # The if-block takes the predicate as input and outputs a field
        use_offsets = self._offsets_of_uses_in_scope(computation, node.outputs[0])

        # The output type is the union of the uses
        # For each output, the extent is the union of the uses
        for output, output_t in zip(node.outputs, node.operation_type.destination):
            output_t.extent.extents = use_offsets
            output_t.extent.sort_extents()

        # The input type is [0, 0, 0]
        node.operation_type.source[0].extent.extents[:] = [sast.Offset((0, 0, 0))]

        # Recurse into the if-else block
        self.generic_visit(node)

    def visit_ReturnOp(self, node: sast.ReturnOp):
        assert isinstance(node.operation_type.source[0], sast.ViewType)

        _init_outputs(node.operation_type.source)


    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        computation = self.get_scope()
        # Input is 0 0 0
        node.operation_type.source[0].extent.extents[:] = [sast.Offset((0, 0, 0))]
        # Output is given by the union of the uses.

        node.operation_type.destination[0].extent.extents[:] = self._offsets_of_uses_in_scope(computation, node.result)
        node.operation_type.destination[0].extent.sort_extents()

    def _offsets_of_uses_in_scope(self, computation: sast.ComputationBlock, identifier: sast.Identifier):
        """
        Get the offsets of the uses of a field in the current scope.

        :param computation: The current computation block
        :param identifier: The identifier
        :return: A list of offsets
        """
        uses_of_result = self.uses.get(identifier) or []
        uses_of_result = [u for u in uses_of_result if u.definition_scope == computation]

        # Concatenate all the extents from uses_of_result
        offsets = []
        for use in uses_of_result:
            # Any use is by a view type
            assert isinstance(use.field_type, sast.ViewType)
            if use.field_type.extent.is_unknown():
                raise ValueError(f"Extent of {identifier.name} is unknown in use in scope {computation.as_ir()}")
            else:
                offsets.extend(use.field_type.extent.extents)

        if len(offsets) == 0:
            warnings.warn(f"No uses of {identifier.name} found within current scope.")

        # Make sure all offsets are known
        assert all(not o.is_unknown() for o in offsets), f"Offsets of uses of {identifier.as_ir()} must be known before inference"

        return offsets


def _init_outputs(dtypes: Sequence[sast.ViewType]):
    """
    Initialize the extents of every element to be (0, 0, 0)

    :param dtypes: Field type elements to set.
    """
    for dtype in dtypes:
        if isinstance(dtype, sast.ViewType) and dtype.extent.is_unknown():
            dtype.extent.extents[:] = [sast.Offset((0, 0, 0))]


def _minkowski_sum(a: Collection[sast.Offset],
                   b: Collection[sast.Offset]) -> set[sast.Offset]:
    """
    Compute the Minkowski sum of two sets of extents.

    :param a: First set operand.
    :param b: Second set operand.
    :return: The Minkowski sum set result.
    """
    result = set()
    for a_extent in a:
        for b_extent in b:
            result.add(a_extent + b_extent)
    return result


class LocalExtentCollector(sast.NodeVisitor):

    """
    A node visitor that collects all input and output extents from field accesses in the visited blocks/statements.
    """

    # The extents are stored in a dictionary with the field name as key and a set of extents as value.
    extents: dict[str, set[tuple[int]]]

    def __init__(self):
        super().__init__()
        self.extents: dict[str, set[sast.Offset]] = defaultdict(set)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        # Visit only the body (arguments do not count as accesses)
        for b in node.body:
            self.visit(b)
    def visit_Identifier(self, node: sast.Identifier):
        # If a bare identifier (i.e., no subscript) is used, the extent (0, 0, 0) should be added
        self.extents[node.name].add(sast.Offset((0, 0, 0)))

    def visit_Subscript(self, node: sast.Subscript):
        # If a subscript is found, add its subscript to the extents.
        # Make sure not to recursively visit into the subscript to avoid adding (0, 0, 0)
        self.extents[node.value.name].add(sast.Offset((node.subscript[0], node.subscript[1], node.subscript[2])))
