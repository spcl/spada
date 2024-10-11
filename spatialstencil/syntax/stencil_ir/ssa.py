import copy
from dataclasses import dataclass
from typing import Mapping

import spatialstencil.syntax.stencil_ir.irnodes as sast
from spatialstencil.syntax.stencil_ir.irnodes import ComputationBlock, Program


@dataclass
class ScopedVersion:
    """
    Represents a version of a variable in a given scope.
    """
    scope: sast.ComputationBlock | sast.Program
    version: int


class SSAVisitor(sast.ScopedNodeVisitor):
    """
    A node visitor that transforms a given program into SSA form.
    The visitor assigns a version to each variable in the program.
    The version is incremented each time the variable is assigned to in the current scope.

    After inference, the version of a variable is stored in the `version` attribute of the `Identifier` node,
    and it is guaranteed that the version of a variable is unique in a given scope.
    """

    __current_version_in_scope: Mapping[str, list[ScopedVersion]]

    def __init__(self):
        super().__init__()
        self._current_version_in_scope = dict()

    def get_scope(self):
        return self._current_scope[0]

    def _get_version(self, name: str) -> int:
        """
        Returns the current version of a variable in the current scope.
        If the variable is not defined in the current scope, -1 is returned.

        :param name: The name of the variable.
        :return: The current version of the variable.
        """
        if name not in self._current_version_in_scope:
            return -1
        scope = self.get_scope()
        for version in self._current_version_in_scope[name]:
            if version.scope == scope:
                return version.version
        return -1

    def _set_version(self, identifier: sast.Identifier, version: int, scope=None):
        """
        Sets the version of a variable in the current scope.
        Modifies the identifier in place and updates the internal state (current version).

        :param identifier: The identifier to set the version for.
        :param version: The version to set.
        :param scope: The scope to set the version for. If None, the current scope is used.
        :return:
        """
        name = identifier.name
        identifier.version = version
        if scope is None:
            scope = self.get_scope()
        if name not in self._current_version_in_scope:
            self._current_version_in_scope[name] = [ScopedVersion(scope, version)]
        else:
            for i, v in enumerate(self._current_version_in_scope[name]):
                if v.scope == scope:
                    self._current_version_in_scope[name][i] = ScopedVersion(scope, version)
                    return
            self._current_version_in_scope[name].append(ScopedVersion(scope, version))

    def _increment_version(self, identifier: sast.Identifier):
        """
        Increments the version of a variable in the current scope.

        :param identifier: The identifier to increment the version for.
        :return:
        """
        current_version = self._get_version(identifier.name)
        self._set_version(identifier, current_version + 1)
        assert identifier.version == current_version + 1
        # Assert there is at most one version of the identifier in the current scope
        assert len([v for v in self._current_version_in_scope[identifier.name] if v.scope == self.get_scope()]) == 1

    def visit_Identifier(self, node: sast.Identifier):
        # Set the version of the identifier to the current version in the current scope
        node.version = self._get_version(node.name)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        # Increment the version of all identifiers that the statement assigns to in the current scope
        # This is done AFTER visiting the nodes nested in the statement
        self.generic_visit(node)
        for out in node.outputs:
            self._increment_version(out)

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
            self._set_version(inp, version, scope=computation)

    def post_visit_ComputationBlock(self, computation: ComputationBlock):
        # Increment the version of all identifiers that the computation assigns to in the current scope.

        for out in computation.outputs:
            self._increment_version(out)
