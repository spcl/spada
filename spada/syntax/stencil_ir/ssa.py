import copy
from dataclasses import dataclass
from typing import Mapping

import spada.syntax.stencil_ir.irnodes as sast
from spada.syntax.stencil_ir import analysis
from spada.syntax.stencil_ir.irnodes import ComputationBlock, Program


class SSAVisitor(sast.ScopedNodeVisitor):
    """
    A node visitor that transforms a given program into SSA form.
    The visitor assigns a version to each variable in the program.
    The version is incremented each time the variable is assigned to in the current scope.

    After inference, the version of a variable is stored in the `version` attribute of the `Identifier` node,
    and it is guaranteed that the version of a variable is unique in a given scope.

    The body of a FORWARD/BACKWARD computation is the body of a sequential k-loop rather than straight-line code.
    For fields that the computation assigns to, vertically-offset reads are therefore versioned by the
    iteration they read from:

    * Levels already processed by the loop (k - n for FORWARD, k + n for BACKWARD) hold the value at the end of
      an earlier iteration, or the incoming value if that level is outside the computation's interval. Both are
      the computation's own output version (the full field after the computation), so the read refers to it.
    * Levels not yet processed (k + n for FORWARD, k - n for BACKWARD) still hold the incoming value, so the
      read refers to the version that was current before the computation.
    * Reads at the current level (offset 0) follow the straight-line rule.
    """

    __current_version: Mapping[str, int]

    def __init__(self):
        super().__init__()
        self._current_version_in_scope = dict()
        self._incoming_version: dict[str, int] = {}
        self._written_in_computation: set[str] = set()
        self._loop_carried_reads: list[sast.Identifier] = []

    def _get_version(self, name: str) -> int:
        """
        Returns the current version of a variable.
        If the variable is not defined yet, -1 is returned.

        :param name: The name of the variable.
        :return: The current version of the variable.
        """
        if name not in self._current_version_in_scope:
            return -1

        return self._current_version_in_scope[name]

    def _set_version(self, identifier: sast.Identifier, version: int):
        """
        Sets the version of a variable.
        Modifies the identifier in place and updates the internal state (current version).

        :param identifier: The identifier to set the version for.
        :param version: The version to set.
        :return:
        """
        name = identifier.name
        identifier.version = version

        self._current_version_in_scope[name] = version

    def _increment_version(self, identifier: sast.Identifier):
        """
        Increments the version of a variable in the current scope.

        :param identifier: The identifier to increment the version for.
        :return:
        """
        current_version = self._get_version(identifier.name)
        self._set_version(identifier, current_version + 1)
        assert identifier.version == current_version + 1

    def _sequential_computation(self) -> ComputationBlock | None:
        computation = self.get_scope_with_type(ComputationBlock)
        if computation is None or computation.schedule == sast.ComputationType.PARALLEL:
            return None
        return computation

    def visit_Identifier(self, node: sast.Identifier):
        # Set the version of the identifier to the current version in the current scope
        node.version = self._get_version(node.name)

    def visit_Subscript(self, node: sast.Subscript):
        computation = self._sequential_computation()
        name = node.value.name
        if computation is None or node.subscript[2] == 0 or name not in self._written_in_computation:
            self.generic_visit(node)
            return

        # A fresh identifier avoids aliasing with the statement's input list, which follows the straight-line rule
        if analysis.is_loop_carried_access(computation, node):
            # The output version is only known after the body has been visited
            node.value = sast.Identifier(name, 0)
            self._loop_carried_reads.append(node.value)
        else:
            if name not in self._incoming_version:
                raise ValueError(f"Field {name} is read at vertical offset {node.subscript[2]} before it is defined")
            node.value = sast.Identifier(name, self._incoming_version[name])

    def visit_StatementBlock(self, node: sast.StatementBlock):
        # Increment the version of all identifiers that the statement assigns to in the current scope
        # This is done AFTER visiting the nodes nested in the statement
        self.generic_visit(node)
        if self._sequential_computation() is not None:
            self._update_statement_inputs(node)
        for out in node.outputs:
            self._increment_version(out)

    def _update_statement_inputs(self, node: sast.StatementBlock):
        """
        Makes the inputs of a statement in a sequential computation match the versions accessed in its body.
        A statement may access several versions of the same field (e.g., the current version and the incoming
        version at a later level). Loop-carried reads are not inputs, as they refer to the result of the
        enclosing computation.
        """
        loop_carried = set(id(ident) for ident in self._loop_carried_reads)
        local = set(stmt.result.name for stmt in node.body if isinstance(stmt, sast.AssignOp))
        accessed: set[sast.Identifier] = set()
        for subnode in node.walk():
            if isinstance(subnode, sast.Subscript):
                if id(subnode.value) not in loop_carried:
                    accessed.add(subnode.value)
            elif isinstance(subnode, sast.Expression) and isinstance(subnode.value, sast.Identifier):
                accessed.add(subnode.value)
        accessed = {ident for ident in accessed if ident.name not in local}

        types_by_name = {inp.name: inp_t for inp, inp_t in zip(node.inputs, node.operation_type.source)}
        if set(node.inputs) == accessed or not all(ident.name in types_by_name for ident in accessed):
            return
        node.inputs = [sast.Identifier(ident.name, ident.version) for ident in sorted(accessed)]
        node.operation_type.source = [copy.deepcopy(types_by_name[ident.name]) for ident in node.inputs]

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self.generic_visit(node)
        self._increment_version(node.result)

    def visit_IfBlock(self, node: sast.IfBlock):
        # We deepcopy the outputs to avoid sharing the same identifiers between different branches
        node.outputs = [copy.deepcopy(out) for out in node.outputs]
        self.generic_visit(node)
        for out in node.outputs:
            self._increment_version(out)

    def do_visit_Program(self, program: Program):
        # Initialize the program inputs to version 0
        for inp in program.inputs:
            self._set_version(inp, 0)

        for computation in program.computations:
            self.visit(computation)

    def visit_ReturnOp(self, node: sast.ReturnOp):
        # We deepcopy the returns to avoid sharing the same identifiers between different computations
        node.values = [copy.deepcopy(val) for val in node.values]
        self.generic_visit(node)

    def pre_visit_ComputationBlock(self, computation: ComputationBlock):
        # Copy the inputs and outputs
        computation.inputs = [copy.deepcopy(inp) for inp in computation.inputs]
        computation.outputs = [copy.deepcopy(out) for out in computation.outputs]

        # Define the inputs of the computation with the current version
        for inp in computation.inputs:
            version = self._get_version(inp.name)
            self._set_version(inp, version)

        self._incoming_version = dict(self._current_version_in_scope)
        self._written_in_computation = analysis.names_written_in(computation)
        self._loop_carried_reads = []

    def post_visit_ComputationBlock(self, computation: ComputationBlock):
        # Increment the version of all identifiers that the computation assigns to in the current scope.

        for out in computation.outputs:
            self._increment_version(out)

        output_versions = {out.name: out.version for out in computation.outputs}
        for ident in self._loop_carried_reads:
            if ident.name not in output_versions:
                raise ValueError(f"Field {ident.name} is read across k-iterations in a {computation.schedule.name} "
                                 f"computation, so it must be an output of that computation")
            ident.version = output_versions[ident.name]
        self._loop_carried_reads = []
