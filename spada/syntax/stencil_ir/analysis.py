"""
Analysis passes on the Stencil IR.
"""
from collections import defaultdict
from spada.syntax.stencil_ir import irnodes as sast
from typing import Literal


class ExtentCollector(sast.NodeVisitor):
    """
    A node visitor that collects all input and output extents from field accesses in the visited blocks/statements.
    """

    def __init__(self):
        super().__init__()
        self._DEFAULT_INTERVAL = (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

        self.extents: dict[str, dict[tuple[int | None], set[tuple[int | None]]]] = defaultdict(lambda: defaultdict(set))
        self.interval: tuple[sast.Interval, sast.Interval, sast.Interval] = self._DEFAULT_INTERVAL

    @property
    def flat_interval(self) -> tuple[int | Literal["?"] | None]:
        return tuple(item for dim in self.interval for item in (dim.start, dim.end))

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        # Set current interval
        self.interval = node.interval

        # Skip arguments and outputs
        for stmt in node.body:
            self.visit(stmt)

        # Reset current interval
        self.interval = self._DEFAULT_INTERVAL

    def visit_StatementBlock(self, node: sast.StatementBlock):
        # Skip arguments
        for stmt in node.body:
            self.visit(stmt)
        for out in node.outputs:
            self.visit(out)

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        # Skip value
        pass

    def visit_Identifier(self, node: sast.Identifier):
        # If a bare identifier (i.e., no subscript) is used, the extent (0, 0, 0) should be added
        self.extents[node.name][self.flat_interval].add((0, 0, 0))

    def visit_Subscript(self, node: sast.Subscript):
        # If a subscript is found, add its subscript to the extents.
        # Make sure not to recursively visit into the subscript to avoid adding (0, 0, 0)
        self.extents[node.value.name][self.flat_interval].add(tuple(node.subscript))


def collect_extents(node: sast.Node) -> dict[str, set[tuple[int]]]:
    """
    Collects all input and output extents from field accesses in this block.
    
    :param node: The node to traverse.
    """
    collector = ExtentCollector()
    collector.visit(node)
    return collector.extents


class InputOutputCollector(sast.NodeVisitor):
    """
    A node visitor that collects all input and output fields in the visited blocks/statements.
    """

    def __init__(self):
        super().__init__()
        self.inputs: set[sast.Identifier] = set()
        self.outputs: set[sast.Identifier] = set()

    # Visitors that return used identifiers
    def visit_Identifier(self, node: sast.Identifier) -> set[sast.Identifier]:
        return {node}

    def visit_Expression(self, node: sast.Expression) -> set[sast.Identifier]:
        if isinstance(node.value, (int, float)):
            return set()
        return self.visit(node.value)

    def visit_Subscript(self, node: sast.Subscript) -> set[sast.Identifier]:
        return self.visit(node.value)

    def visit_UnaryOperator(self, node: sast.UnaryOperator) -> set[sast.Identifier]:
        return self.visit(node.value)

    def visit_BinaryOperator(self, node: sast.BinaryOperator) -> set[sast.Identifier]:
        return self.visit(node.left) | self.visit(node.right)

    def visit_TernaryOperator(self, node: sast.TernaryOperator) -> set[sast.Identifier]:
        return self.visit(node.test) | self.visit(node.true_value) | self.visit(node.false_value)

    def visit_MathCall(self, node: sast.MathCall) -> list[sast.Identifier]:
        return set().union(*(self.visit(arg) for arg in node.arguments))

    # Visitors that infer input/output status from returned identifiers
    def visit_ReturnOp(self, node: sast.ReturnOp):
        self.inputs |= set().union(*(self.visit(retval) for retval in node.values))

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self.inputs |= self.visit(node.value)
        self.outputs |= self.visit(node.result)

    def visit_AssignOp(self, node: sast.AssignOp):
        self.inputs |= self.visit(node.value)
        # self.outputs |= self.visit(node.result)  # Only include return values as outputs

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self.inputs |= set(node.inputs)
        self.outputs |= set(node.outputs)

    def visit_IfBlock(self, node: sast.IfBlock):
        self.inputs |= self.visit(node.condition)
        self.outputs |= set(node.outputs)

        # Recurse to children
        self.generic_visit(node)

    def visit_ElseIfBlock(self, node: sast.ElseIfBlock):
        if node.condition is not None:
            self.inputs |= self.visit(node.condition)

        # Recurse to body
        self.generic_visit(node)
