"""
Includes canonicalization routines for Stencil IR.
"""

from collections import defaultdict
from typing import Literal

from spada.syntax.stencil_ir import irnodes as sast
from spada.syntax.stencil_ir import type_inference


def canonicalize(program: sast.Program) -> sast.Program:
    """
    Canonicalizes a Stencil IR program by reordering extents to a canonical order.
    """
    program = CanonicalizeExtents().visit(program)
    return program


class CanonicalizeExtents(sast.NodeTransformer):
    """
    Sorts extents to canonicalize order.
    """
    def _modify_typeinfo(self, operation_type: sast.OperationType):
        """
        Helper function that updates the type information based on inferred types.
        """
        for src in operation_type.source:
            if isinstance(src, sast.ViewType):
                src.extent.sort_extents()

        if operation_type.destination:
            for dst in operation_type.destination:
                if isinstance(dst, sast.ViewType):
                    dst.extent.sort_extents()

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_IfBlock(self, node: sast.IfBlock):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_Program(self, node: sast.Program):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)
