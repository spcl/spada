from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir.canonicalization import PEBlock


def preprocess_rectangle(rect: PEBlock):
    """
    Performs a set of pre-processing passes on code.
    """
    _fma_fusion(rect)
    _promote_scalar_receive_targets(rect)


def _fma_fusion(rect: PEBlock):
    """
    Detects multiply-accumulate chains and converts them to ``FusedMultiplyAccumulate`` IR nodes.
    """


def _promote_scalar_receive_targets(rect: PEBlock) -> None:
    """
    Gives every scalar that receives from a fabric stream a one-element array to live in.

    A ``receive`` lowers to a copy between descriptors, and the local side of one is a memory DSD --
    a description of how to walk something indexable. A CSL scalar has nothing to index, so the copy
    used to fall back to a plain assignment whose right-hand side was the stream's name, which is not
    a variable at all. Declaring the scalar as a one-element array gives the copy a descriptor to
    write through; every use other than the transfer itself is rewritten to ``x[0]``, so it still
    reads and writes as a scalar.

    Streams that cross the kernel boundary are left alone: their contents either arrive by host
    memcpy or are delivered to a data task, and neither wants a descriptor over the target.
    """
    fabric_streams = {
        declaration.stream_name.as_ir()
        for declaration in rect.dataflow.statements
        if isinstance(declaration, spir.StreamDeclaration) and
        not isinstance(declaration.stream, spir.ExternStreamDeclaration)
    }
    scalar_fields = {
        field.field_name.as_ir(): field
        for field in rect.place.statements
        if isinstance(field, spir.FieldDeclaration) and isinstance(field.dtype, spir.ScalarType)
    }
    if not fabric_streams or not scalar_fields:
        return

    promoted: dict[str, spir.FieldDeclaration] = {}
    for statement in rect.compute.statements:
        for node in statement.walk():
            if not isinstance(node, spir.ReceiveStatement):
                continue
            stream = node.stream_name
            stream = stream.array if isinstance(stream, spir.ArraySlice) else stream
            if stream.as_ir() not in fabric_streams:
                continue
            target = transfer_local_operand(node)
            if target is not None and target.as_ir() in scalar_fields:
                promoted[target.as_ir()] = scalar_fields[target.as_ir()]

    if not promoted:
        return

    for field in promoted.values():
        field.dtype = spir.ArrayType(field.dtype, [1])

    transformer = _ScalarToSingletonArray(set(promoted))
    for statement in rect.compute.statements:
        transformer.visit(statement)


def transfer_local_operand(statement: spir.SendStatement | spir.ReceiveStatement) -> spir.Identifier | None:
    """
    Returns the identifier a transfer names as its local operand, or None if it names something else
    (an array element or a literal), which moves one value and never becomes a whole-buffer
    descriptor.
    """
    local = statement.local_array
    if isinstance(local, spir.TypedIdentifier):
        local = local.identifier
    return local if isinstance(local, spir.Identifier) else None


def local_transfer_operands(rect: PEBlock) -> set[str]:
    """
    Returns the names every ``send`` and ``receive`` in this rectangle uses as its local operand.
    """
    operands: set[str] = set()
    for statement in rect.compute.statements:
        for node in statement.walk():
            if isinstance(node, (spir.SendStatement, spir.ReceiveStatement)):
                operand = transfer_local_operand(node)
                if operand is not None:
                    operands.add(operand.as_ir())
    return operands


class _ScalarToSingletonArray(spir.NodeTransformer):
    """
    Rewrites reads and writes of a promoted scalar into accesses to element zero of its new array.

    The local operand of a transfer keeps naming the array itself: that is the one place that wants
    the whole (one-element) buffer, because it is what the memory DSD is built from.
    """

    def __init__(self, promoted: set[str]):
        super().__init__()
        self.promoted = promoted

    def visit_Identifier(self, node: spir.Identifier) -> spir.Identifier | spir.ArraySlice:
        if node.as_ir() not in self.promoted:
            return node
        element = spir.ArraySlice(node, [spir.Expression(spir.ConstantLiteral(0, spir.ScalarType.u16))])
        element.lineinfo = node.lineinfo
        return element

    def visit_ArraySlice(self, node: spir.ArraySlice) -> spir.ArraySlice:
        # A promoted scalar is never the array being sliced, only (possibly) one of the indices.
        node.indices = self.generic_visit_sequence(node.indices)
        return node

    def visit_TypedIdentifier(self, node: spir.TypedIdentifier) -> spir.TypedIdentifier:
        # A declaration names the variable; it does not read it.
        return node

    def visit_SendStatement(self, node: spir.SendStatement) -> spir.SendStatement:
        return self._visit_transfer(node)

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement) -> spir.ReceiveStatement:
        return self._visit_transfer(node)

    def _visit_transfer(self, node: spir.SendStatement | spir.ReceiveStatement):
        target = transfer_local_operand(node)
        if target is None or target.as_ir() not in self.promoted:
            node.local_array = self.visit(node.local_array)
        node.stream_name = self.visit(node.stream_name)
        return node
